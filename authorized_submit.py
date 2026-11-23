"""ARIA workbench: validate only by default; explicit --submit required later.

No launch at import. Pending decisions are rejected before any queue call.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from nanochat.workflow_policy import verify_package,authorize,CAP
from nanochat.sft_runtime import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--submit',action='store_true')
    p.add_argument('--submission-receipt',type=Path,default=Path('/workspace/nanochat_sft/consolidated-submission.json'));a=p.parse_args()
    root=Path(__file__).resolve().parent;package,identity=verify_package(root)
    approval=authorize(a.approval,identity,package['assets'])
    receipt=json.loads(a.receipt.read_text())
    if receipt['workflow_sha256']!=identity:raise PermissionError('Receipt/workflow mismatch')
    if a.submission_receipt.exists():raise FileExistsError('Allocation already submitted; do not duplicate')
    if not a.submit:
        print(json.dumps({'all_decisions_explicitly_accepted':True,'submitted':False}));return
    sys.path.insert(0,'/workspace/skills/wbagent/scripts')
    import launch_helpers as launch
    import wandb
    job=wandb.Api().artifact(receipt['job_ref'])
    if job.digest!=receipt['job_digest']:raise ValueError('Job digest changed')
    temp=Path('/workspace/nanochat_sft/authorized-job-check');temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
    info=launch.download_code_artifact(receipt['job_ref'])
    _,remote_identity=verify_package(info['code_dir'])
    if remote_identity!=identity or info['base_image']!=package['image'] or info['entrypoint']!=package['entrypoint']:
        raise ValueError('Launch job is not the approved workflow')
    capacity=approval['decisions']['gpu_execution'].get('capacity_evidence',{})
    if capacity.get('eight_l40_one_node_confirmed') is not True or not capacity.get('source') or not capacity.get('observed_at_utc'):
        raise PermissionError('Current one-node/eight-L40 schedulability evidence missing; historical runs are insufficient')
    resources=launch._resolve_resource_args(entity='wandb',queue_name='nanochat-demo-k8s',gpus=8,cpu=8,memory='80Gi')
    spec=resources['kubernetes']['spec'];spec.update(activeDeadlineSeconds=CAP,backoffLimit=0)
    spec['template']['spec'].update(restartPolicy='Never',terminationGracePeriodSeconds=15)
    # Charge queue delay/startup conservatively; this timestamp NEVER resets.
    start=time.time()
    config=launch._with_wb_agent_provenance(dict(approval=approval,allocation_started_unix=start,gpus=8,
        image=package['image'],workflow_sha256=identity,total_seconds_including_prior=7200,prior_seconds=87))
    # Exclusive submission intent prevents a lost response becoming a duplicate.
    a.submission_receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.submission_receipt.open('x') as stream:
        json.dump(dict(status='submission_intent_no_automatic_retry',allocation_started_unix=start,job_ref=receipt['job_ref']),stream)
    queued=launch.launch_add(job=receipt['job_ref'],entity='wandb',project='nanochat-demo',queue_name='nanochat-demo-k8s',
        entry_point=package['entrypoint'],resource_args=resources,config={'overrides':{'run_config':config}})
    atomic_json(a.submission_receipt,dict(status='queued',queue_item_id=queued.id,allocation_started_unix=start,
        allocation_ends_unix=start+CAP,job_ref=receipt['job_ref'],workflow_sha256=identity))
    print(json.dumps({'queued':queued.id,'allocation_ends_unix':start+CAP}))

if __name__=='__main__':main()