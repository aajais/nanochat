"""Non-destructive filtered cache, unchanged split assignments and answer tokens."""
import copy,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prepared'))
from nanochat.sft_runtime import PackedLoader,sha256
from nanochat.tokenizer import RustBPETokenizer
from scripts.prepare_conversation_sft import normalize,text_of
import torch

torch.set_num_threads(1)
review=ROOT/'readiness_review_v1';original=ROOT/'conversation_cache_protected_v1'
out=ROOT/'conversation_cache_quality_v1'
out.mkdir(exist_ok=False)
audit=json.loads((review/'instruction-label-audit.json').read_text())
excluded={f['id'] for f in audit['flags']}
assert len(excluded)==1166
manifest=json.loads((review/'candidate-cache-manifest.json').read_text())
manifest=copy.deepcopy(manifest)
manifest.pop('rebuild_comparison',None)
manifest.pop('conversation_evaluation_artifact',None)
tok=RustBPETokenizer.from_directory(ROOT/'tokenizer')
all_rows={};statistics={};removed={};split_of={}
for split in ('train','dev'):
    source=original/(split+'.jsonl')
    assert sha256(source)==manifest['files'][source.name]
    kept=[];dropped=0
    with source.open() as stream,(out/source.name).open('w') as sink:
        for line in stream:
            row=json.loads(line)
            if row['id'] in excluded:dropped+=1;continue
            sink.write(line);kept.append(row);split_of[row['id']]=split
    removed[split]=dropped;all_rows[split]=kept
    statistics[split]=dict(examples=len(kept),input_tokens=sum(len(r['ids'])-1 for r in kept),assistant_tokens=sum(sum(r['mask'][1:]) for r in kept),near_context_examples=sum(len(r['ids'])>=1843 for r in kept),categories=dict(Counter(r['category'] for r in kept)))
    manifest['files'][source.name]=sha256(out/source.name)
assert sum(removed.values())==len(excluded)
assert not {r['id'] for r in all_rows['train']}&{r['id'] for r in all_rows['dev']}
manifest['statistics']=statistics
manifest['filter_counts']['summary_instruction_risk_quarantined']=len(excluded)
manifest['filter_counts']['kept']=sum(len(v) for v in all_rows.values())
manifest['quality_quarantine']={'criterion':'Any recorded summary sentence-limit heuristic flag; conservative exclusion, not adjudicated source error rate','exclusions':len(excluded),'original_data_unchanged':True,'split_assignments_preserved':True,'responses_rewritten_or_truncated':False,'audit_sha256':sha256(review/'instruction-label-audit.json')}
manifest['release_gate']['completed'].append('conservative quarantine of summary sentence-limit risks')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'quarantine-ids.json').write_text(json.dumps(sorted(excluded),indent=2))
loader=PackedLoader(all_rows['train'],2048,8,tok.get_bos_token_id(),manifest['effective_shares'],seed=20261123)
assistant=0
for _ in range(128):
    x,y=loader.next('cpu');assert x.shape==y.shape==(8,2048)
    assistant+=int((y!=-1).sum())
result={'statistics':statistics,'quarantined_by_original_split':removed,'quarantined_total':len(excluded),'original_cache_unchanged':all(sha256(original/f)==v for f,v in json.loads((original/'manifest.json').read_text())['files'].items()),'train_dev_leakage_cannot_increase':'strict subset; no re-splitting or re-encoding','packing_dry_run':{'updates':128,'input_slots':128*16384,'assistant_tokens':assistant,'assistant_category_shares':{k:v/assistant for k,v in loader.category_tokens.items()}},'training_approved':False,'all_human_reviews':'pending'}
(review/'quality-filtered-cache-audit.json').write_text(json.dumps(result,indent=2))
# Refresh the human review packet to contain only the NEW candidate's retained rows.
samples=defaultdict(list);seen=set()
for line in (ROOT/'conversation_candidates/selected.jsonl').open():
    row=json.loads(line);key=hashlib.sha256(normalize(text_of(row['messages'],True)).encode()).hexdigest()
    if key not in split_of or key in seen:continue
    seen.add(key);component=row['source'].rsplit('/',1)[1]
    samples[component].append(dict(id=key,split=split_of[key],source=row['source'],source_row=row['source_row'],messages=row['messages']))
selected=[]
for component,rows in sorted(samples.items()):
    rows.sort(key=lambda row:hashlib.sha256(('review-20261123:'+row['id']).encode()).hexdigest());selected+=rows[:20]
packet={'scope':'100 source-stratified retained conversations; NOT representative under weighted sampling','human_review':'pending','candidate_files':manifest['files'],'examples':selected}
(review/'filtered-dataset-human-review-sample.json').write_text(json.dumps(packet,indent=2))
md=['# Filtered candidate dataset review — human approval pending','','Twenty deterministic retained examples per source; no excluded rows. Judge correctness, instruction compliance, context grounding, concise English, conversational consistency and unwanted content. Do not infer all-row correctness from this sample.','']
for row in selected:
    md+=['## '+row['source'].rsplit('/',1)[1]+' / '+row['id'],f"Split: {row['split']}; source row: {row['source_row']}",'']
    for msg in row['messages']:md+=['### '+msg['role'],msg['content'],'']
    md+=['Reviewer: ______ | pass / reject / uncertain: ______ | rationale: ______','']
(review/'FILTERED_DATASET_REVIEW.md').write_text('\n'.join(md))
print(json.dumps(result,indent=2),flush=True)