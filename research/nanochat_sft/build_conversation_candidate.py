"""Deterministic bounded selection from cached, pinned SmolTalk components."""
from collections import Counter
import hashlib,json,random,re,sys,time
from pathlib import Path
import pyarrow.parquet as pq

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'prepared'))
from scripts.prepare_conversation_sft import prepare,validate_messages


def main():
 source=root/'conversation_sources'
 metadata=json.loads((source/'sources.json').read_text())
 candidates=root/'conversation_candidates'
 candidates.mkdir(exist_ok=True)
 output=candidates/'selected.jsonl'
 if output.exists():raise ValueError('Selection already exists: refuse overwrite')
 selection={}
 allowed_general={'information-seeking','advice-seeking','planning','brainstorming','editing','creative-writing'}
 caps={'smol-magpie-ultra':7000,'smol-constraints':3000,'everyday-conversations':2260,'smol-summarize':4000,'metamathqa-50k':3000}
 with output.open('w') as sink:
  for component,cap in caps.items():
   counters=Counter()
   table=pq.read_table(source/(component+'.parquet'))
   indices=list(range(len(table)))
   # Stable seed, no random network/dataset streaming in preparation or training.
   random.Random(20261123).shuffle(indices)
   rows=table.to_pylist()
   for idx in indices:
    row=rows[idx];counters['scanned']+=1
    messages=row['messages']
    error=validate_messages(messages)
    if error:counters[error]+=1;continue
    if component=='smol-magpie-ultra':
     if row['category'] not in allowed_general or row['quality'] not in ('good','excellent') or row['difficulty'] not in ('very easy','easy','medium'):
      counters['excluded_source_metadata']+=1;continue
    if component=='metamathqa-50k' and not row['type'].startswith('GSM_'):
     counters['excluded_advanced_math']+=1;continue
    text='\n'.join(m['content'] for m in messages)
    # Transparent heuristic only; does not establish English fluency or answer correctness.
    letters=[c for c in text if c.isalpha()]
    if letters and sum(ord(c)<128 for c in letters)/len(letters)<.90:
     counters['excluded_non_ascii_letter_heavy']+=1;continue
    if len(text)>24000 or len(messages)>12:
     counters['excluded_oversize_or_long_dialogue']+=1;continue
    if any('```' in m['content'] or re.search(r'<\|(?:tool|python)|<tool_call|<function_call|\[TOOL_CALLS\]',m['content'],re.I) for m in messages):
     counters['excluded_code_or_tool_syntax']+=1;continue
    if any(len(m['content'].split())>650 for m in messages if m['role']=='assistant'):
     counters['excluded_verbose_answer']+=1;continue
    category=('general' if component in ('smol-magpie-ultra','smol-constraints') else
              'reasoning' if component=='metamathqa-50k' else 'context')
    record={'messages':messages,'category':category,
      'source':metadata['repo']+'/'+component,'revision':metadata['revision'],
      'source_row':idx,'source_file_sha256':metadata['files'][component]['sha256']}
    if component=='metamathqa-50k':record['source_type']=row['type']
    if component=='smol-magpie-ultra':record['source_category']=row['category']
    sink.write(json.dumps(record,ensure_ascii=False)+'\n')
    counters['selected']+=1
    if counters['selected']>=cap:break
   selection[component]={'cap':cap,**dict(counters)}
   print(json.dumps({'phase':'select','source':component,'counts':selection[component]}),flush=True)
 (candidates/'selection.json').write_text(json.dumps(selection,indent=2))
 protected=[root/'arc_inputs/benchmark-inputs.json']
 started=time.monotonic()
 manifest=prepare([output],root/'conversation_cache',root/'tokenizer',protected,
   progress=lambda result:print(json.dumps({'phase':'prepare','elapsed_s':round(time.monotonic()-started),**result}),flush=True))
 manifest['release_gate']={'training_approved':False,'pending':[
   'freeze_and_exclude_conversation_development_and_both_acceptance_sets',
   'independent_dataset_quality_and_license_review',
   'untouched_pretraining_holdout_overlap_check_without_opening_holdout']}
 manifest['source_cache']=metadata
 manifest['selection']=selection
 manifest['selection_seed']=20261123
 manifest['scope']='bounded deterministic sample, not the full SmolTalk corpus'
 (root/'conversation_cache/manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'phase':'complete','statistics':manifest['statistics'],'filter_counts':manifest['filter_counts'],
                   'effective_shares':manifest['effective_shares'],'training_approved':False}),flush=True)


if __name__=='__main__':main()