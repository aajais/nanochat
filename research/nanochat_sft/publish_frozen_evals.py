"""Publish frozen fixtures, protected provisional data, code; never run a model."""
import hashlib,json,os
from pathlib import Path
import wandb

root=Path(__file__).resolve().parent
scope='wandb/nanochat-demo'
evaldir=root/'conversation_evals_v1'
cachedir=root/'conversation_cache_protected_v1'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

hashes=json.loads((evaldir/'SHA256.json').read_text())
for name,value in hashes.items():assert sha(evaldir/name)==value
manifest=json.loads((cachedir/'manifest.json').read_text())
assert len(manifest['protected_manifests'])==4
assert all(v['byte_identical'] for v in manifest['rebuild_comparison'].values())
assert manifest['release_gate']['training_approved'] is False
assert manifest['filter_counts'].get('protected_overlap',0)==0
for rel in ('scripts/base_train.py','tests/test_training_process.py'):
 assert sha(root/'prepared'/rel)==sha(root/'training_source'/rel)
test_evidence={'targeted_tests':{'sft_preparation':8,'eval_fixtures':3,'passed':11},
 'repository_discovery':{'run':16,'errors':2,'skipped':1,
   'errors_detail':'Historical TestAccumulationTelemetry AST execution lacks require_finite_loss; both files match captured training source.'},
 'protected_screens':{'ARC100_snapshot':True,'behavioral_development':True,'primary_acceptance':True,
                      'correction_acceptance':True,'new_matches_under_exact_heuristic_screen':0,
                      'cache_train_dev_byte_identical':True},
 'model_evaluations_run':0,'gpu_jobs_launched':0,
 'existing_untouched_holdout_payloads_opened':False,'human_fixture_review':'pending'}
(root/'frozen_eval_test_evidence.json').write_text(json.dumps(test_evidence,indent=2))
prov={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',name='frozen-conversation-evals-no-gpu',
 job_type='sft-eval-preparation',dir=str(root),save_code=False,
 config={'_wb_agent':prov,'scope':'fixtures and CPU exclusion checks only','launch_ready':False})
for ref in ('conversation-sft-cache-candidate:v0','conversation-sft-preparation:v1',
            'pr1-smoke-tokenizer:v0','e1384arc-benchmark-inputs:v0'):
 run.use_artifact(scope+'/'+ref)
api=wandb.Api()
published={}

def publish(name,kind,files,metadata,aliases):
 artifact=wandb.Artifact(name,type=kind,metadata=metadata)
 for path,target in files:artifact.add_file(str(path),name=target)
 logged=run.log_artifact(artifact,aliases=aliases).wait()
 ref=scope+'/'+logged.name
 reread=api.artifact(ref,type=kind)
 assert reread.digest==logged.digest
 assert set(reread.manifest.entries)=={target for _,target in files}
 assert reread.metadata['launch_ready'] is False
 url=run.url.split('/runs/')[0]+f'/artifacts/{kind}/{name}/{reread.version}'
 published[name]={'ref':ref,'url':url,'digest':reread.digest,'file_count':len(files),'verified_from_server':True}
 print(json.dumps({'published':published[name]}),flush=True)
 return ref

evalfiles=[(p,p.name) for p in evaldir.iterdir() if p.is_file()]
evalfiles.append((root/'frozen_eval_test_evidence.json','test_evidence.json'))
evalref=publish('nanochat-conversation-demo-evals','dataset',evalfiles,
 {'launch_ready':False,'status':'frozen-not-evaluated','cases_per_suite':100,'responses_per_suite':180,
  'suite_count':3,'model_outputs_generated':0,'human_fixture_review_required':True,
  'shared_task_templates':True,'access_control_sealed':False},['frozen-v1-not-evaluated'])
manifest['conversation_evaluation_artifact']=evalref
(cachedir/'manifest.json').write_text(json.dumps(manifest,indent=2))
cachefiles=[(p,p.name) for p in cachedir.iterdir() if p.is_file()]
cachefiles.extend([(root/'conversation_sources/sources.json','sources/sources.json'),
                   (root/'conversation_sources/SMOLTALK_CARD.md','sources/SMOLTALK_CARD.md'),
                   (root/'conversation_candidates/selection.json','sources/selection.json'),
                   (root/'frozen_eval_test_evidence.json','audit/protected_exclusions.json')])
dataref=publish('conversation-sft-cache-candidate','dataset',cachefiles,
 {'launch_ready':False,'training_approved':False,'evaluation_artifact':evalref,
  'statistics':manifest['statistics'],'release_gate':manifest['release_gate'],
  'new_protected_matches':0,'train_dev_bytes_changed':False},['candidate-not-launch-ready'])
codefiles=[(p,str(p.relative_to(root/'prepared'))) for p in (root/'prepared').rglob('*')
 if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.toml','.lock','.txt','.json','.md')]
for name in ('freeze_conversation_evals.py','rebuild_protected_conversations.py','PREPARATION_STATUS.md',
             'frozen_eval_test_evidence.json','base_verification.json','preparation_tests.json',
             'cache_conversations.py','build_conversation_candidate.py','audit_conversation_candidate.py'):
 codefiles.append((root/name,'preparation/'+name))
codehashes={target:sha(path) for path,target in codefiles}
(root/'frozen_eval_code_sha256.json').write_text(json.dumps(codehashes,indent=2))
codefiles.append((root/'frozen_eval_code_sha256.json','preparation/SHA256.json'))
publish('conversation-sft-preparation','code',codefiles,
 {'launch_ready':False,'unit_tests_passed':11,'repository_suite_errors':2,'gpu_jobs_launched':0,
  'evaluation_artifact':evalref,'dataset_artifact':dataref,'native_gpu_resume':'not_run'},['candidate-not-launch-ready'])
run.summary.update({'launch_ready':False,'targeted_tests_passed':11,'repository_test_errors':2,
 'gpu_jobs_launched':0,'model_evaluations_run':0,'protected_matches':0,
 'evaluation_cases_total':300,'planned_responses_per_suite':180,'human_fixture_review':'pending'})
runid,url=run.id,run.url
run.finish()
state=api.run(scope+'/'+runid).state
result={'run_url':url,'run_id':runid,'run_state':state,'artifacts':published,
        'launch_ready':False,'model_evaluations_run':0,'gpu_jobs_launched':0}
(root/'published_frozen_evals.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)