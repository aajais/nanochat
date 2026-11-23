"""Submit only the explicitly approved bounded single-GPU plumbing smoke."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,'/workspace/skills/wbagent/scripts')
import launch_helpers as launch
import wandb

ROOT=Path('/workspace/nanochat_sft')
PACKAGE=ROOT/'gpu_smoke_job_v1'
RECEIPT=ROOT/'gpu_smoke_submission.json'
prepare_only=os.environ.get('PREPARE_ONLY')=='1'
if RECEIPT.exists() and not prepare_only:
    raise RuntimeError('Submission receipt already exists; inspect it instead of duplicating GPU work')
image=json.loads((ROOT/'pinned_training_image.json').read_text())['image']
assert image=='wandb/autoresearch-base@sha256:05578cc98ed0629b6518482bfe8de57935e5b852ed6057446b2a09a7f60edeca'
files={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
       and 'smoke_work' not in p.parts and p.name!='SMOKE_PACKAGE.json'}
manifest={'image':image,'source_files_sha256':files,
          'authorized_queue':'nanochat-demo-k8s','gpus':1,'max_job_seconds':900,
          'smoke_only':True,'main_sft_auto_launch':False,
          'synthetic_fixture':True,'no_acceptance_evaluation':True}
(PACKAGE/'SMOKE_PACKAGE.json').write_text(json.dumps(manifest,indent=2))
# Exercise the actual entrypoint's source validation before publishing or spending GPU time.
from unittest.mock import patch
import importlib.util
spec=importlib.util.spec_from_file_location('smoke_supervisor_preflight',PACKAGE/'gpu_smoke_supervisor.py')
supervisor=importlib.util.module_from_spec(spec)
with patch('pathlib.Path.mkdir'),patch('signal.alarm'):
    spec.loader.exec_module(supervisor)
class ReachedInitialization(Exception): pass
old_cwd=os.getcwd()
try:
    os.chdir(PACKAGE)
    with patch.object(supervisor.wandb,'init',side_effect=ReachedInitialization) as initialize:
        try:
            supervisor.main()
        except ReachedInitialization:
            pass
        else:
            raise AssertionError('Expected initialization boundary was not reached')
        initialize.assert_called_once()
    with patch.object(supervisor,'sha256',return_value='invalid'),patch.object(supervisor.wandb,'init') as initialize:
        try:
            supervisor.main()
        except AssertionError as error:
            assert 'Code hash mismatch' in str(error)
        else:
            raise AssertionError('Hash mismatch failed to block initialization')
        initialize.assert_not_called()
finally:
    os.chdir(old_cwd)
print('Packaged supervisor preflight passed; tampered source rejected before initialization',flush=True)
artifact,action,aliases=launch._create_job(
    api=launch.LaunchApi(), job_type='code', path=str(PACKAGE),entity='wandb',project='nanochat-demo',
    name='nanochat-one-gpu-resume-smoke',entrypoint='python -u gpu_smoke_supervisor.py',base_image=image)
job_ref=f'wandb/nanochat-demo/{artifact.name}'
print(json.dumps({'job':job_ref,'action':action}),flush=True)
if prepare_only:
    (ROOT/'gpu_smoke_corrected_job.json').write_text(json.dumps({
        'job_ref':job_ref,'job_artifact_digest':artifact.digest,'manifest':manifest,
        'local_entrypoint_contract_passed':True,'source_tamper_rejected':True,
        'gpu_resubmitted':False},indent=2))
    raise SystemExit(0)
resources=launch._resolve_resource_args(entity='wandb',queue_name='nanochat-demo-k8s',gpus=1,cpu=8,memory='80Gi')
kube=resources['kubernetes']
kube['spec']['activeDeadlineSeconds']=900
kube['spec']['backoffLimit']=0
kube['spec']['template']['spec']['restartPolicy']='Never'
kube['spec']['template']['spec']['terminationGracePeriodSeconds']=15
container=kube['spec']['template']['spec']['containers'][0]
assert container['resources']['limits']['nvidia.com/gpu']=='1'
assert kube['spec']['activeDeadlineSeconds']==900 and kube['spec']['backoffLimit']==0
config=launch._with_wb_agent_provenance({
    'mode':'single-gpu-training-save-fresh-process-restore-smoke',
    'approved_gpu_count':1,'approved_smoke_minutes':15,'image':image,
    'context':2048,'microbatch':1,'global_input_tokens':16384,
    'gradient_accumulation':8,'synthetic_data_only':True,
    'no_automatic_retry':True,'no_main_sft_auto_launch':True})
queued=launch.launch_add(job=job_ref,entity='wandb',project='nanochat-demo',
                         queue_name='nanochat-demo-k8s',entry_point=['python','-u','gpu_smoke_supervisor.py'],
                         resource_args=resources,config={'overrides':{'run_config':config}})
receipt={'queue_item_id':queued.id,'queue':'nanochat-demo-k8s','job_ref':job_ref,
         'job_artifact_digest':artifact.digest,'manifest':manifest,
         'hard_deadline_seconds':900,'backoff_limit':0,'gpus':1}
RECEIPT.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
status=launch.check_launch('wandb','nanochat-demo','nanochat-demo-k8s',queued.id)
(ROOT/'gpu_smoke_last_status.json').write_text(json.dumps(status,indent=2,default=str))
print(json.dumps(status,default=str),flush=True)