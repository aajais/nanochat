"""Deterministic CPU/data audit; no model calls, holdout payload reads or GPUs."""
import hashlib,json,random,sys,time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prepared'))
from scripts.prepare_conversation_sft import NearIndex,normalize,text_of,validate_messages
from nanochat.sft_runtime import PackedLoader,sha256
from nanochat.tokenizer import RustBPETokenizer
import torch

def main():
    torch.set_num_threads(1)
    out=ROOT/'readiness_review_v1'
    out.mkdir(exist_ok=True)
    cache=ROOT/'conversation_cache_protected_v1'
    old=json.loads((cache/'manifest.json').read_text())
    tok=RustBPETokenizer.from_directory(ROOT/'tokenizer')
    source=json.loads((ROOT/'conversation_sources/sources.json').read_text())
    source_hashes={k:sha256(ROOT/'conversation_sources'/(k+'.parquet'))==v['sha256']==v['hub_lfs_sha256'] for k,v in source['files'].items()}
    assert all(source_hashes.values())
    fixtures=ROOT/'conversation_evals_v3'
    hashes=json.loads((fixtures/'SHA256.json').read_text())
    assert all(sha256(fixtures/k)==v for k,v in hashes.items())
    protected=NearIndex();spans=set();prompt_count=0
    for name in ['development','acceptance_primary','acceptance_correction']:
        # Prompt-only contamination check. No responses generated or inspected.
        suite=json.loads((fixtures/(name+'.json')).read_text())
        for row in suite['examples']:
            for msg in row['messages']:
                norm=normalize(msg['content']);protected.add(norm);prompt_count+=1
                if len(norm.split())>=12:spans.add(norm)
    for row in json.loads((ROOT/'arc_inputs/benchmark-inputs.json').read_text())['examples']:
        for value in (row['query'],row['query'].removeprefix('Question:').strip()):
            protected.add(value)
            if len(normalize(value).split())>=12:spans.add(normalize(value))
    records={};all_ids=set();stats={}
    for split in ['train','dev']:
        path=cache/(split+'.jsonl')
        assert sha256(path)==old['files'][path.name]
        records[split]=[json.loads(x) for x in path.open()]
        ids={r['id'] for r in records[split]}
        assert len(ids)==len(records[split]) and not ids&all_ids
        all_ids|=ids
        stats[split]={'conversations':len(ids),'input_tokens':sum(len(r['ids'])-1 for r in records[split]),'assistant_tokens':sum(sum(r['mask'][1:]) for r in records[split]),'near_context_limit':sum(len(r['ids'])-1>=1843 for r in records[split])}
    by_id={r['id']:r for rows in records.values() for r in rows}
    train_ids={row['id'] for row in records['train']}
    raw={};counts=Counter();near=Counter();multi=Counter();samples=defaultdict(list)
    started=time.monotonic()
    for i,line in enumerate((ROOT/'conversation_candidates/selected.jsonl').open()):
        row=json.loads(line)
        key=hashlib.sha256(normalize(text_of(row['messages'],True)).encode()).hexdigest()
        if key not in all_ids or key in raw:continue
        ids,mask=tok.render_conversation({'messages':row['messages']},max_tokens=10_000_000)
        if ids!=by_id[key]['ids']:continue
        assert mask==by_id[key]['mask']
        assert validate_messages(row['messages']) is None
        assert ids[-1]==tok.encode_special('<|assistant_end|>') and mask[-1]==1
        assert len(ids)<=2049 and min(ids)>=0 and max(ids)<32768
        raw[key]=row
        component=row['source'].rsplit('/',1)[1]
        counts[component]+=1
        multi['multi_turn' if sum(m['role']=='user' for m in row['messages'])>1 else 'single_turn']+=1
        for msg in row['messages']:
            if msg['role'] not in ('user','system'):continue
            norm=normalize(msg['content'])
            if protected.contains(norm):near['exact_or_simhash']+=1
            if any((' '+span+' ') in (' '+norm+' ') for span in spans):near['embedded_prompt']+=1
        samples[component].append(dict(id=key,split='train' if key in train_ids else 'dev',category=row['category'],source=row['source'],source_row=row['source_row'],messages=row['messages']))
        if len(raw)%3000==0:print(json.dumps({'audited':len(raw),'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
    assert len(raw)==len(all_ids)
    assert not near,dict(near)
    train_prompt,train_full=NearIndex(),NearIndex()
    for row in records['train']:
        train_prompt.add(text_of(raw[row['id']]['messages'],True));train_full.add(text_of(raw[row['id']]['messages']))
    overlap={'prompt':0,'full_conversation':0}
    for row in records['dev']:
        overlap['prompt']+=train_prompt.contains(text_of(raw[row['id']]['messages'],True))
        overlap['full_conversation']+=train_full.contains(text_of(raw[row['id']]['messages']))
    assert not any(overlap.values()),overlap
    loader=PackedLoader(records['train'],2048,8,tok.get_bos_token_id(),old['effective_shares'],seed=20261123)
    total=0
    for _ in range(128):
        x,y=loader.next('cpu');assert tuple(x.shape)==tuple(y.shape)==(8,2048)
        total+=int((y!=-1).sum())
    assert total==sum(loader.category_tokens.values())
    selected=[]
    for component,rows in sorted(samples.items()):
        rows.sort(key=lambda row:hashlib.sha256(('review-20261123:'+row['id']).encode()).hexdigest())
        selected+=rows[:20]
    (out/'dataset-human-review-sample.json').write_text(json.dumps({'scope':'100 source-stratified retained conversations; NOT statistically representative under weighted training sampling','human_review':'pending','examples':selected},indent=2))
    md=['# Dataset review packet — human approval pending','','Twenty deterministic retained examples per source. Review accuracy, relevance, supported claims, concise English, multi-turn consistency, and unwanted/repetitive content. This sample is not proof about all rows.','']
    for row in selected:
        md+=['## '+row['source'].rsplit('/',1)[1]+' / '+row['id'][:12],f"Split: {row['split']}; source row: {row['source_row']}",'']
        for msg in row['messages']:md += ['### '+msg['role'],msg['content'],'']
        md += ['Reviewer: ______ | pass / reject / uncertain: ______ | rationale: ______','']
    (out/'DATASET_REVIEW.md').write_text('\n'.join(md))
    manifest=dict(old)
    manifest['protected_manifests']={str(ROOT/'arc_inputs/benchmark-inputs.json'):sha256(ROOT/'arc_inputs/benchmark-inputs.json'),**{str(fixtures/(s+'.json')):hashes[s+'.json'] for s in ['development','acceptance_primary','acceptance_correction']}}
    manifest['evaluation_fixture_status']='v3 wording fixes; runner pins refreshed; human fixture review pending'
    manifest['release_gate']={'training_approved':False,'completed':['all cached source SHA256 checks','all retained token/mask/template checks','ARC100 plus v3 development/primary/correction prompt screening','train/dev exact and heuristic near-duplicate checks','weighted CPU packing dry-run'],'pending':['human dataset content review','human evaluation fixture review','license/attribution acceptance','protected holdout per-example fingerprint comparison']}
    (out/'candidate-cache-manifest.json').write_text(json.dumps(manifest,indent=2))
    result={'source_shards_sha256_verified':source_hashes,'splits':stats,'source_conversation_counts':dict(counts),'turn_counts':dict(multi),'v3_protected_conversation_prompts':prompt_count,'v3_and_arc_prompt_screen_matches':sum(near.values()),'screen':'NFKC/casefold exact, SimHash Hamming<=3, embedded full normalized prompts>=12 words; not semantic proof','train_dev_heuristic_overlap':overlap,'token_ids_and_loss_masks_all_rows_verified':len(raw),'train_dev_files_unchanged':True,'packing_dry_run':{'updates':128,'input_slots':128*16384,'assistant_tokens':total,'assistant_category_shares':{k:v/total for k,v in loader.category_tokens.items()}},'human_dataset_review':'pending','human_fixture_review':'pending','holdout_fingerprints':'unavailable in protected artifact; payloads not opened','training_approved':False,'gpu_jobs_launched_this_audit':0,'model_generation_calls':0,'acceptance_generations':0}
    (out/'data-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    for component,rows in sorted(samples.items()):
        for row in rows[:2]:print(json.dumps({'ai_review_sample':component,'id':row['id'],'messages':[{**m,'content':m['content'][:1000]} for m in row['messages'][:4]]}),flush=True)

if __name__=='__main__':main()