"""Archive blocked data/code preparation, with server readback. No GPU work."""
import hashlib,json,os,time
from pathlib import Path
import wandb

root=Path(__file__).resolve().parent
scope='wandb/nanochat-demo'
manifest=json.loads((root/'conversation_cache/manifest.json').read_text())
audit=json.loads((root/'conversation_audit.json').read_text())
assert manifest['release_gate']['training_approved'] is False
assert audit['unit_tests_passed']==8 and audit['all_masks_match_exact_tokenizer_rendering'] is True

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

dataset_files={p.name:p for p in (root/'conversation_cache').iterdir() if p.is_file()}
for name in ('sources.json','SMOLTALK_CARD.md'):
 dataset_files['sources/'+name]=root/'conversation_sources'/name
dataset_files['sources/selection.json']=root/'conversation_candidates/selection.json'
dataset_files['audit/conversation_audit.json']=root/'conversation_audit.json'
code_files={}
for p in (root/'prepared').rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.toml','.lock','.txt','.json','.md'):
  code_files[str(p.relative_to(root/'prepared'))]=p
for name in ('cache_conversations.py','build_conversation_candidate.py','audit_conversation_candidate.py',
             'PREPARATION_STATUS.md','base_verification.json','preparation_tests.json','conversation_audit.json'):
 code_files['preparation/'+name]=root/name
hashes={'dataset':{n:digest(p) for n,p in dataset_files.items()},
        'code':{n:digest(p) for n,p in code_files.items()}}
(root/'conversation_candidate_sha256.json').write_text(json.dumps(hashes,indent=2))
dataset_files['audit/sha256.json']=root/'conversation_candidate_sha256.json'
code_files['preparation/sha256.json']=root/'conversation_candidate_sha256.json'
provenance={key:os.environ[name] for key,name in
 [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if name in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',job_type='sft-data-preparation',
 name='conversation-cache-candidate-no-gpu',dir=str(root),save_code=False,
 config={'_wb_agent':provenance,'scope':'CPU preparation only; no GPU launch',
         'launch_ready':False,'training_approved':False,'source_revision':manifest['source_cache']['revision']})
run.use_artifact(scope+'/b1384b1-checkpoint:v2')
run.use_artifact(scope+'/pr1-smoke-tokenizer:v0')
run.use_artifact(scope+'/code-nanochat-1384m-8l40-startup3:v0')
refs={}
for kind,name,files in [('dataset','conversation-sft-cache-candidate',dataset_files),
                        ('code','conversation-sft-preparation',code_files)]:
 meta={'launch_ready':False,'training_approved':False,'gpu_jobs_launched':0,'unit_tests_passed':8,
       'native_gpu_resume':'not_run','real_conversation_dataset':'provisional',
       'immutable_container':'missing','source_revision':manifest['source_cache']['revision']}
 if kind=='dataset':meta.update({'statistics':manifest['statistics'],'release_gate':manifest['release_gate']})
 artifact=wandb.Artifact(name,type=kind,metadata=meta)
 for target,path in files.items():artifact.add_file(str(path),name=target)
 logged=run.log_artifact(artifact,aliases=['candidate-not-launch-ready']).wait()
 refs[kind]={'ref':scope+'/'+logged.name,'digest':logged.digest,'files':len(files)}
 print(json.dumps({'phase':'uploaded',kind:refs[kind]}),flush=True)
run.summary.update({'launch_ready':False,'training_approved':False,'gpu_jobs_launched':0,
                    'unit_tests_passed':8,'train_conversations':17298,'dev_conversations':886,
                    'train_assistant_tokens':8879264,'native_gpu_resume':'not_run'})
run_id=run.id
run_url=run.url
run.finish()
api=wandb.Api()
for kind,files in [('dataset',dataset_files),('code',code_files)]:
 entry=refs[kind]
 readback=api.artifact(entry['ref'],type=kind)
 assert readback.digest==entry['digest']
 assert readback.metadata['training_approved'] is False
 assert set(readback.manifest.entries)==set(files)
 entry['url']=run_url.split('/runs/')[0]+f'/artifacts/{kind}/{readback.name.split(":")[0]}/{readback.version}'
 entry['verified_from_server']=True
state=api.run(scope+'/'+run_id).state
result={'run_url':run_url,'run_id':run_id,'run_state':state,'artifacts':refs,
        'launch_ready':False,'training_approved':False,'gpu_jobs_launched':0}
(root/'published_conversation_candidate.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)