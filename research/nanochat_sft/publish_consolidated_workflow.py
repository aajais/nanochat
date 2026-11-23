"""Publish preparation evidence and an UNQUEUED job. No launch_add call here."""
import json
import os
from pathlib import Path
import sys
import tempfile
import wandb

ROOT=Path('/workspace/nanochat_sft');P=ROOT/'final_workflow_v1'
sys.path.insert(0,str(P));sys.path.insert(0,'/workspace/skills/wbagent/scripts')
from nanochat.workflow_policy import verify_package,authorize
from nanochat.sft_runtime import atomic_json,sha256
import launch_helpers as launch

def main():
    package,identity=verify_package(P)
    try:authorize(P/'APPROVAL_TEMPLATE.json',identity,package['assets'])
    except PermissionError:pass
    else:raise AssertionError('Preparation must not contain a launch approval')
    provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
    run=wandb.init(entity='wandb',project='nanochat-demo',name='consolidated-production-sft-readiness-preparation-no-gpu',
        job_type='cpu-readiness-preparation',config={'_wb_agent':provenance,'workflow_sha256':identity,'gpu_launched':False,'launch_approved':False})
    for pin in package['assets'].values():run.use_artifact(pin['ref'])
    try:
        artifact=wandb.Artifact('nanochat-consolidated-sft-preparation',type='code',metadata={'workflow_sha256':identity,'gpu_launched':False,'launch_approved':False,'cpu_tests_passed':15,'corrected_evaluator_tests_passed':63})
        names=list(package['source_sha256'])+['WORKFLOW.json','APPROVAL_TEMPLATE.json']
        for name in names:artifact.add_file(str(P/name),name=name)
        logged=run.log_artifact(artifact).wait();ref='wandb/nanochat-demo/'+logged.name
        server=wandb.Api().artifact(ref)
        if server.digest!=logged.digest or set(server.manifest.entries)!=set(names):raise ValueError('Preparation server manifest mismatch')
        destination=Path(server.download(root=str(ROOT/'final-workflow-published-verification')))
        for name in names:
            if sha256(destination/name)!=sha256(P/name):raise ValueError('Preparation server bytes differ: '+name)
        job,action,_=launch._create_job(api=launch.LaunchApi(),job_type='code',path=str(P),entity='wandb',project='nanochat-demo',
            name='nanochat-consolidated-conversation-sft',entrypoint='python -u workflow.py',base_image=package['image'])
        jobref='wandb/nanochat-demo/'+job.name
        confirmed=wandb.Api().artifact(jobref)
        if confirmed.digest!=job.digest:raise ValueError('Job not verified on server')
        temp=ROOT/'final-workflow-job-server-verification';temp.mkdir(exist_ok=True)
        tempfile.tempdir=str(temp)
        info=launch.download_code_artifact(jobref)
        _,jobidentity=verify_package(info['code_dir'])
        if jobidentity!=identity or info['base_image']!=package['image'] or info['entrypoint']!=['python','-u','workflow.py']:
            raise ValueError('Published job source/image/entrypoint differs')
        receipt=dict(preparation_ref=ref,preparation_digest=server.digest,workflow_sha256=identity,
            job_ref=jobref,job_digest=confirmed.digest,job_source_ref=info['source_artifact_name'],
            run_path='wandb/nanochat-demo/'+run.id,preparation_server_bytes_verified=True,job_server_source_verified=True,
            gpu_launched=False,queue_item_created=False,launch_authorized=False,budget=package['budget'],
            required_decisions=list(json.loads((P/'APPROVAL_TEMPLATE.json').read_text())['decisions']))
        atomic_json(ROOT/'consolidated-preparation-receipt.json',receipt)
        receipt_art=wandb.Artifact('nanochat-consolidated-sft-preparation-receipt',type='evaluation',metadata={'gpu_launched':False,'launch_authorized':False})
        receipt_art.add_file(str(ROOT/'consolidated-preparation-receipt.json'),name='preparation-receipt.json')
        published=run.log_artifact(receipt_art).wait();rserver=wandb.Api().artifact('wandb/nanochat-demo/'+published.name)
        downloaded=Path(rserver.get_path('preparation-receipt.json').download(root=str(ROOT/'final-workflow-receipt-verification')))
        if sha256(downloaded)!=sha256(ROOT/'consolidated-preparation-receipt.json'):raise ValueError('Receipt not verified')
        run.log({'cpu/production_path_tests_passed':15,'cpu/corrected_evaluator_tests_passed':63,'preparation/server_verified':1,'gpu/launched':0})
        run.finish();print(json.dumps(receipt))
    except BaseException:
        run.finish(exit_code=1);raise

if __name__=='__main__':main()