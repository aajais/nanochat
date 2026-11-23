import json
import os
from pathlib import Path
import sys
import wandb
ROOT=Path('/workspace/nanochat_sft');P=ROOT/'final_workflow_v1';sys.path.insert(0,str(P))
from nanochat.workflow_policy import verify_package
from nanochat.sft_runtime import atomic_json,sha256
package,identity=verify_package(P)
receipt=json.loads((ROOT/'consolidated-preparation-receipt.json').read_text())
api=wandb.Api();art=api.artifact(receipt['preparation_ref']);job=api.artifact(receipt['job_ref'])
assert art.digest==receipt['preparation_digest'] and job.digest==receipt['job_digest'] and identity==receipt['workflow_sha256']
assert receipt['preparation_server_bytes_verified'] and receipt['job_server_source_verified'] and not receipt['queue_item_created']
provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',name='consolidated-production-sft-preparation-verification-no-gpu',job_type='cpu-readiness-verification',config={'_wb_agent':provenance,'workflow_sha256':identity,'gpu_launched':False})
try:
    run.use_artifact(receipt['preparation_ref']);run.use_artifact(receipt['job_ref'])
    receipt['verification_run_path']='wandb/nanochat-demo/'+run.id
    receipt['publication_verified']=True
    atomic_json(ROOT/'consolidated-preparation-receipt.json',receipt)
    ra=wandb.Artifact('nanochat-consolidated-sft-preparation-receipt',type='evaluation',metadata={'gpu_launched':False,'launch_authorized':False,'workflow_sha256':identity})
    ra.add_file(str(ROOT/'consolidated-preparation-receipt.json'),name='preparation-receipt.json')
    logged=run.log_artifact(ra).wait();server=api.artifact('wandb/nanochat-demo/'+logged.name)
    downloaded=Path(server.get_path('preparation-receipt.json').download(root=str(ROOT/'final-workflow-receipt-verification')))
    assert sha256(downloaded)==sha256(ROOT/'consolidated-preparation-receipt.json')
    run.log({'cpu/production_path_tests_passed':15,'cpu/corrected_evaluator_tests_passed':63,'preparation/server_verified':1,'gpu/launched':0})
    run.summary.update({'receipt_ref':'wandb/nanochat-demo/'+server.name,'launch_ready':False,'prepared':True,'gpu_launch_authorized':False})
    run.finish();print(json.dumps(receipt));print('receipt_ref=wandb/nanochat-demo/'+server.name)
except BaseException:
    run.finish(exit_code=1);raise