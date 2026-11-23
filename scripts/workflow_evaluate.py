"""Bounded native GPU evaluation; frozen acceptance remains locked until selection.

Imports do not load fixtures, generate outputs or reserve acceptance. The
corrected pinned scorer is unchanged. Human annotations are never fabricated.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import signal
import time

from nanochat.sft_runtime import atomic_json,sha256,validate_checkpoint
from nanochat.conversation_eval import (ChatCodec,NativeGreedyBackend,run_frozen,
    score_records,load_frozen,PINNED_HASHES)
from nanochat.conversation_eval_checkpoint import strict_model,MODEL_CONFIG,BASE_WEIGHTS,TOKENIZER
from scripts.conversation_eval import review_template

def rank_development(score):
    """Predeclared conversation-first mechanical ranking, NOT human usefulness."""
    cases=score['case_results'];responses=score['response_results']
    eligible={c['case_id'] for c in cases if c['capability']!='human_review'}
    correct=sum(r['check_pass'] is True for r in responses if r['case_id'] in eligible)
    boundaries=sum(r['boundary_ok'] is True for r in responses)
    loops=sum(r['loop_free'] is True for r in responses)
    return correct,boundaries,loops

def candidates(work):
    complete=sorted((work/'main').glob('step-*/COMPLETE.json'))
    if not complete:raise ValueError('No main SFT checkpoints')
    chosen=sorted({0,len(complete)//2,len(complete)-1})
    return [dict(kind='base',weights=str(work/'assets/base/model_133632.pt'),checkpoint_sha256=BASE_WEIGHTS)]+[
        dict(kind='sft',weights=str(complete[i].parent/'model.pt'),checkpoint_sha256=sha256(complete[i].parent/'model.pt')) for i in chosen]

def load_candidate(candidate):
    import torch
    path=Path(candidate['weights'])
    if sha256(path)!=candidate['checkpoint_sha256']:raise ValueError('Selected weights changed')
    if candidate['kind']=='sft':
        meta=json.loads((path.parent/'loop.json').read_text())
        validate_checkpoint(path.parent,meta['identity'],meta['world_size'])
        if meta['identity']['base_weights_sha256']!=BASE_WEIGHTS:raise ValueError('Wrong SFT base')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    return strict_model(state,MODEL_CONFIG).to('cuda').eval()

def arc100(model,tokenizer,path,checkpoint,end):
    """Historical zero-shot, length-normalized MC NLL, same ordered cached 100."""
    import torch
    from nanochat.core_eval import render_prompts_mc,batch_sequences_mc,stack_sequences,forward_model
    data=json.loads(Path(path).read_text());rows=[];correct=0
    if data['protocol']!='1384m-arc100-text10-20261123-v1' or len(data['examples'])!=100:raise ValueError('ARC protocol/input drift')
    with torch.inference_mode():
        for index,item in enumerate(data['examples']):
            if time.time()>=end:raise TimeoutError('ARC budget exhausted; incomplete, no reduced denominator')
            prompts=render_prompts_mc(item,data['delimiter'],[])
            tokens,starts,ends=batch_sequences_mc(tokenizer,prompts)
            overflow=[max(0,len(t)-2048) for t in tokens]
            if any(s-o<1 or e<=s for s,e,o in zip(starts,ends,overflow)):raise ValueError('Unscorable ARC item')
            ids=stack_sequences([t[o:] for t,o in zip(tokens,overflow)],tokenizer.get_bos_token_id()).cuda()
            losses,_=forward_model(model,ids)
            means=[losses[i,s-o-1:e-o-1].mean().item() for i,(s,e,o) in enumerate(zip(starts,ends,overflow))]
            if not all(math.isfinite(x) for x in means):raise ValueError('ARC nonfinite')
            predicted=min(range(len(means)),key=means.__getitem__);correct+=int(predicted==item['gold'])
            rows.append(dict(index=index,prediction=predicted,gold=item['gold'],choice_mean_nll=means))
    return dict(correct=correct,denominator=100,checkpoint_sha256=checkpoint,
        benchmark_artifact='wandb/nanochat-demo/e1384arc-benchmark-inputs:v0',
        historical_protocol_evidence='code-nanochat-1384m-arc100-text10-20261123:v0/arc_eval.py; exact scoring loop retained',
        historical_protocol='1384m-arc100-text10-20261123-v1:ARC100 component',rows=rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--end',type=float,required=True);a=p.parse_args()
    def expired(*_):raise TimeoutError('Evaluation allocation exhausted; outputs incomplete, acceptance not retried')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(a.end-time.time())))
    work=a.work;out=work/'evaluation';out.mkdir(exist_ok=False)
    codec=ChatCodec.from_directory(work/'assets/tokenizer');ledger=out/'ledger.sqlite';results=[]
    suite_dir=work/'assets/fixtures'
    # A selection rule and fixture approval are necessary BEFORE any evaluation.
    approval=json.loads((work/'approval.json').read_text())
    for candidate in candidates(work):
        number=len(results);model=load_candidate(candidate);backend=NativeGreedyBackend(model)
        identity=dict(kind='model',checkpoint_sha256=candidate['checkpoint_sha256'],tokenizer_sha256=TOKENIZER)
        records=run_frozen(suite_dir,'development',codec,backend,identity,output_path=out/f'dev-{number}.jsonl',ledger_path=ledger)
        suite,protocol=load_frozen(suite_dir,'development');score=score_records(suite,protocol,codec,records)
        atomic_json(out/f'dev-{number}-score.json',score)
        results.append(dict(candidate=candidate,rank=list(rank_development(score)),development_evidence=f'dev-{number}.jsonl'))
        del model,backend;gc.collect()
        import torch
        torch.cuda.empty_cache()
    # Stable ties prefer the initial model/earlier checkpoint; never newest by default.
    selected=max(results,key=lambda r:tuple(r['rank']))
    selection=dict(**selected,selected_by='Owner-authorized deterministic development-only policy',
        selection_rationale='Conversation mechanical turn passes (including recall/correction turns), then boundaries, then loop-free responses; stable earlier tie. Human usefulness remains unassessed.',
        human_selection=False)
    atomic_json(out/'selection.json',selection)  # Freeze before opening primary acceptance.
    model=load_candidate(selected['candidate']);backend=NativeGreedyBackend(model)
    checkpoint=selected['candidate']['checkpoint_sha256']
    identity=dict(kind='model',checkpoint_sha256=checkpoint,tokenizer_sha256=TOKENIZER)
    permit=dict(phase='acceptance_primary',selected_checkpoint_sha256=checkpoint,
        selected_by=selection['selected_by'],selection_rationale=selection['selection_rationale'],
        development_evidence=selected['development_evidence'],protocol_sha256=PINNED_HASHES['protocol.json'],
        human_fixture_review_evidence=approval['decisions']['corrected_fixtures']['evidence'])
    atomic_json(out/'selection-permit.json',permit)
    records=run_frozen(suite_dir,'acceptance_primary',codec,backend,identity,
        output_path=out/'acceptance-primary.jsonl',ledger_path=ledger,permit=permit)
    arc=arc100(model,codec.tokenizer,work/'assets/arc/benchmark-inputs.json',checkpoint,a.end)
    atomic_json(out/'arc100.json',arc)
    suite,protocol=load_frozen(suite_dir,'acceptance_primary')
    score=score_records(suite,protocol,codec,records,arc_evidence=arc)
    atomic_json(out/'acceptance-score.json',score)
    atomic_json(out/'human-review-template.json',review_template(suite,records))
    atomic_json(out/'result.json',dict(completed=True,selected=selected['candidate'],
        release_approved=False,gate_outcome=score['gate_outcome'],human_review_pending=True,
        development_candidates=len(results),acceptance_attempts=1))

if __name__=='__main__':main()