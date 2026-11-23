"""Publish only preparation code/evidence; never submits a compute job."""
import hashlib,json,os
from pathlib import Path
import wandb

root=Path(__file__).resolve().parent
files={}
for section in ('nanochat','scripts','tasks','tests'):
 for p in sorted((root/'prepared'/section).rglob('*.py')):
  files[str(p.relative_to(root/'prepared'))]=p
for name in ('pyproject.toml','uv.lock'):
 p=root/'prepared'/name
 if p.is_file():files[name]=p
for name in ('PREPARATION_STATUS.md','base_verification.json','preparation_tests.json'):
 files['evidence/'+name]=root/name
hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
(root/'bundle_sha256.json').write_text(json.dumps(hashes,indent=2))
files['evidence/bundle_sha256.json']=root/'bundle_sha256.json'
provenance={key:os.environ[name] for key,name in
 [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if name in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',job_type='sft-preparation',
 name='conversation-sft-preparation-no-gpu',config={'_wb_agent':provenance,
 'scope':'CPU preparation only; no GPU launch','launch_ready':False,
 'base_checkpoint':'wandb/nanochat-demo/b1384b1-checkpoint:v2'},save_code=False)
run.use_artifact('wandb/nanochat-demo/b1384b1-checkpoint:v2')
run.use_artifact('wandb/nanochat-demo/pr1-smoke-tokenizer:v0')
artifact=wandb.Artifact('conversation-sft-preparation',type='code',metadata={
 'launch_ready':False,'gpu_jobs_launched':0,'unit_tests_passed':6,
 'native_single_rank_exact_resume':'CPU pass','native_multi_gpu_resume':'untested',
 'real_conversation_dataset':'missing','immutable_container':'missing',
 'source_artifact':'code-nanochat-1384m-8l40-startup3:v0'})
for name,path in files.items():artifact.add_file(str(path),name=name)
logged=run.log_artifact(artifact,aliases=['candidate-not-launch-ready']).wait()
ref=f'wandb/nanochat-demo/{logged.name}'
run.summary.update({'preparation_status':'candidate-not-launch-ready','gpu_jobs_launched':0,
                    'base_weight_structure_and_finiteness':'pass','cpu_forward':'pass',
                    'unit_tests_passed':6,'native_cpu_resume_exact':'pass',
                    'two_rank_protocol_only_resume_exact':'pass','native_gpu_resume':'not_run'})
run_url=run.url
run_id=run.id
run.finish()
api=wandb.Api()
readback=api.artifact(ref,type='code')
assert readback.digest==logged.digest
assert readback.metadata['launch_ready'] is False
assert set(files)==set(readback.manifest.entries)
for name,digest in hashes.items():
 assert name in readback.manifest.entries
read_run=api.run(f'wandb/nanochat-demo/{run_id}')
assert read_run.state=='finished'
result={'artifact':ref,'digest':readback.digest,'file_count':len(files),
 'run_url':run_url,'artifact_url':f'{run_url.split("/runs/")[0]}/artifacts/code/conversation-sft-preparation/{readback.version}',
 'verified_from_server':True,'launch_ready':False}
(root/'published_preparation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)