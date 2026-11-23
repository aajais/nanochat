"""Publish and verify preparation/job only; no launch_add or GPU submission."""
import json
import os
from pathlib import Path
import sys
import tempfile
import wandb
ROOT=Path('/workspace/nanochat_sft');P=ROOT/'final_workflow_v2'
sys.path.insert(0,str(P));sys.path.insert(0,'/workspace/skills/wbagent/scripts')
from nanochat.workflow_policy import verify_package,authorize
from nanochat.sft_runtime import atomic_json,sha256
import launch_helpers as launch

def main():
    package,identity=verify_package(P)
    try:authorize(P/'APPROVAL_TEMPLATE.json',identity,package['assets'])
    except PermissionError:pass
    else:raise AssertionError('Pending template must not authorize anything')
    # _create_job may finish its active bookkeeping run: perform before ours.
    job,_,_=launch._create_job(api=launch.LaunchApi(),job_type='code',path=str(P),entity='wandb',project='nanochat-demo',
        name='nanochat-consolidated-conversation-sft',entrypoint='python -u workflow.py',base_image=package['image'])
    jobref='wandb/nanochat-demo/'+job.name
    api=wandb.Api();confirmed=api.artifact(jobref)
    if confirmed.digest!=job.digest:raise ValueError('Job server digest mismatch')
    temp=ROOT/'eight-l40-job-source-verification';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
    source=launch.download_code_artifact(jobref);_,remoteidentity=verify_package(source['code_dir'])
    if remoteidentity!=identity or source['entrypoint']!=package['entrypoint'] or source['base_image']!=package['image']:
        raise ValueError('Job server bytes/entrypoint/image mismatch')
    provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
    run=wandb.init(entity='wandb',project='nanochat-demo',name='eight-l40-consolidated-sft-preparation-no-gpu',
        job_type='cpu-readiness-preparation',config={'_wb_agent':provenance,'workflow_sha256':identity,
        'gpu_launched':False,'target_gpu_count':8,'current_capacity_verified':False,'budget_basis':'shared_wall_clock'})
    try:
        for pin in package['assets'].values():run.use_artifact(pin['ref'])
        run.use_artifact(jobref)
        metadata=dict(workflow_sha256=identity,gpu_launched=False,launch_authorized=False,
            target_gpu_count=8,current_capacity_verified=False,production_cpu_tests_passed=15,
            eight_rank_cpu_contract_tests_passed=10,evaluator_cpu_tests_passed=63)
        artifact=wandb.Artifact('nanochat-consolidated-sft-preparation',type='code',metadata=metadata)
        names=list(package['source_sha256'])+['WORKFLOW.json','APPROVAL_TEMPLATE.json']
        for name in names:artifact.add_file(str(P/name),name=name)
        logged=run.log_artifact(artifact).wait();ref='wandb/nanochat-demo/'+logged.name
        server=api.artifact(ref)
        if server.digest!=logged.digest or set(server.manifest.entries)!=set(names):raise ValueError('Preparation server manifest differs')
        downloaded=Path(server.download(root=str(ROOT/'eight-l40-preparation-server-verification')))
        for name in names:
            if sha256(downloaded/name)!=sha256(P/name):raise ValueError('Preparation server bytes differ: '+name)
        receipt=dict(preparation_ref=ref,preparation_digest=server.digest,job_ref=jobref,job_digest=confirmed.digest,
            workflow_sha256=identity,run_path='wandb/nanochat-demo/'+run.id,
            job_server_source_verified=True,preparation_server_bytes_verified=True,gpu_launched=False,
            queue_item_created=False,launch_authorized=False,current_capacity_verified=False,
            target_gpu_count=8,world_size=8,budget_basis='shared_wall_clock',budget=package['budget'],
            production_cpu_tests_passed=15,eight_rank_contract_tests_passed=10,evaluator_tests_passed=63)
        path=ROOT/'eight-l40-preparation-receipt.json';atomic_json(path,receipt)
        ra=wandb.Artifact('nanochat-consolidated-sft-preparation-receipt',type='evaluation',metadata=metadata)
        ra.add_file(str(path),name='preparation-receipt.json');published=run.log_artifact(ra).wait()
        rserver=api.artifact('wandb/nanochat-demo/'+published.name)
        rpath=rserver.get_path('preparation-receipt.json').download(root=str(ROOT/'eight-l40-receipt-verification'))
        if sha256(rpath)!=sha256(path):raise ValueError('Receipt readback differs')
        run.log({'cpu/production_path_tests_passed':15,'cpu/eight_rank_contract_tests_passed':10,
            'cpu/evaluator_tests_passed':63,'preparation/server_verified':1,'gpu/launched':0})
        run.summary.update({'prepared':True,'launch_ready':False,'current_eight_l40_capacity_verified':False,
            'job_ref':jobref,'preparation_ref':ref,'receipt_ref':'wandb/nanochat-demo/'+rserver.name})
        run.finish();print('PUBLICATION_RECEIPT='+json.dumps(receipt));print('RECEIPT_REF=wandb/nanochat-demo/'+rserver.name)
    except BaseException:
        run.finish(exit_code=1);raise

if __name__=='__main__':main()