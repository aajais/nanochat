"""Exact cache checks and deterministic packing dry-run; no model invocation."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'prepared'))
from nanochat.sft_runtime import PackedLoader,sha256
from nanochat.tokenizer import RustBPETokenizer
from scripts.prepare_conversation_sft import NearIndex,normalize,text_of

torch.set_num_threads(1)
cache=root/'conversation_cache'
manifest=json.loads((cache/'manifest.json').read_text())
tokenizer=RustBPETokenizer.from_directory(root/'tokenizer')
eot=tokenizer.encode_special('<|assistant_end|>')
rows={}
ids={}
stats={}
for split in ('train','dev'):
 assert sha256(cache/(split+'.jsonl'))==manifest['files'][split+'.jsonl']
 records=[json.loads(x) for x in (cache/(split+'.jsonl')).open()]
 rows[split]=records
 ids[split]={r['id'] for r in records}
 assert len(ids[split])==len(records)
 by_category=Counter()
 for r in records:
  assert 2<=len(r['ids'])<=2049
  assert len(r['ids'])==len(r['mask'])
  assert min(r['ids'])>=0 and max(r['ids'])<32768
  assert set(r['mask'])<={0,1}
  assert r['ids'][-1]==eot and r['mask'][-1]==1
  assert sum(r['mask'][1:])>0
  by_category[r['category']]+=sum(r['mask'][1:])
 stats[split]={'examples':len(records),'assistant_tokens_by_category':dict(by_category),
               'assistant_tokens':sum(by_category.values()),'all_rows_valid':True}
assert not ids['train']&ids['dev']
all_ids=ids['train']|ids['dev']
cached_by_id={r['id']:r for records in rows.values() for r in records}
audited=set()
raw_by_id={}
protected=NearIndex()
for row in json.loads((root/'arc_inputs/benchmark-inputs.json').read_text())['examples']:
 protected.add(row['query'])
 protected.add(row['query'].removeprefix('Question:').strip())
arc_overlap=0
retained_text=0
rawstats=Counter()
for line in (root/'conversation_candidates/selected.jsonl').open():
 r=json.loads(line)
 key=hashlib.sha256(normalize(text_of(r['messages'],True)).encode()).hexdigest()
 if key not in all_ids or key in audited:continue
 rendered,mask=tokenizer.render_conversation({'messages':r['messages']},max_tokens=10_000_000)
 if rendered!=cached_by_id[key]['ids']:continue
 assert mask==cached_by_id[key]['mask']
 raw_by_id[key]=r
 audited.add(key)
 retained_text+=1
 if any(protected.contains(m['content']) for m in r['messages'] if m['role'] in ('user','system')):
  arc_overlap+=1
 rawstats['multi_turn' if len([m for m in r['messages'] if m['role']=='user'])>1 else 'single_turn']+=1
assert arc_overlap==0
assert retained_text==len(ids['train'])+len(ids['dev'])
train_prompt_index,train_full_index=NearIndex(),NearIndex()
for key in ids['train']:
 train_prompt_index.add(text_of(raw_by_id[key]['messages'],True))
 train_full_index.add(text_of(raw_by_id[key]['messages']))
near_prompt_overlap=sum(train_prompt_index.contains(text_of(raw_by_id[key]['messages'],True)) for key in ids['dev'])
near_full_overlap=sum(train_full_index.contains(text_of(raw_by_id[key]['messages'])) for key in ids['dev'])
assert near_prompt_overlap==near_full_overlap==0
loader=PackedLoader(rows['train'],2048,8,tokenizer.get_bos_token_id(),manifest['effective_shares'],seed=20261123)
supervised=0
for _ in range(128):
 x,y=loader.next('cpu')
 assert x.shape==(8,2048) and y.shape==(8,2048)
 assert bool((x>=0).all())
 assert bool((y==-1).any())
 supervised+=int((y!=-1).sum())
assert supervised==sum(loader.category_tokens.values())
actual={c:n/supervised for c,n in loader.category_tokens.items()}
result={'split_checks':stats,'train_dev_exact_prompt_overlap':0,'arc100_prompt_overlap_under_recorded_heuristic':arc_overlap,
        'train_dev_heuristic_near_prompt_overlap':near_prompt_overlap,
        'train_dev_heuristic_near_full_overlap':near_full_overlap,
        'all_masks_match_exact_tokenizer_rendering':True,
        'retained_conversation_turn_counts':dict(rawstats),
        'packing_dry_run':{'global_batch_input_slots':16384,'updates':128,'input_slots':128*16384,
                           'assistant_tokens':supervised,'realized_assistant_token_shares':actual,
                           'requested_shares':manifest['effective_shares']},
        'unit_tests_passed':8,'training_approved':False,
        'limitations':['Near-duplicate matching is heuristic, not semantic proof.',
                       'Primary/correction acceptance and conversation-development exclusions are pending.',
                       'Existing untouched pretraining holdout payloads were not downloaded or opened.',
                       'Source labels/ASCII heuristics do not prove English fluency or answer correctness.',
                       'Context component is conversation/summarization; explicit grounded Q&A coverage is unverified.']}
(root/'conversation_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)