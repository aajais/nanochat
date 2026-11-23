"""Publish CPU-only candidate/audit versions, never a Launch job."""
import hashlib
import json
import os
from pathlib import Path
import wandb

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'cpu_remediation_v2'
SCOPE='wandb/nanochat-demo'
RECEIPT=ROOT/'published_cpu_remediation.json'
if RECEIPT.exists():raise FileExistsError('Existing publication receipt: inspect, do not duplicate runs/artifacts')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def write_receipt():RECEIPT.write_text(json.dumps(receipt,indent=2))

result=json.loads((OUT/'remediation-result.json').read_text())
assert result['mandatory_quarantined']==13 and result['additional_quarantined']==854
assert result['training_approved'] is False
assert result['statistics']['train']['examples']==15375 and result['statistics']['dev']['examples']==789
assert 'Ran 24 tests' in (OUT/'cpu-unit-tests.log').read_text() and '\nOK\n' in (OUT/'cpu-unit-tests.log').read_text()
provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',name='cpu-content-remediation-shared-budget-no-gpu',
    job_type='sft-cpu-remediation',dir=str(ROOT),save_code=False,
    config={'_wb_agent':provenance,'responsible_owner':'Aayush Jaiswal','intended_use':'internal demo',
            'training_approved':False,'gpu_jobs_launched':0,'model_calls':0,
            'scope':'Strict-subset data remediation and CPU-tested shared final-workflow budget only'})
receipt=dict(run_id=run.id,run_url=run.url,artifacts={},training_approved=False,gpu_jobs_launched=0,model_calls=0)
write_receipt()
api=wandb.Api()
for ref in ['conversation-sft-cache-candidate:v2','nanochat-sft-bounded-gate-review:v0','nanochat-sft-holdout-comparison:v0','nanochat-conversation-demo-evals:v1','pr1-smoke-tokenizer:v0']:
    run.use_artifact(SCOPE+'/'+ref)

def publish(name,kind,files,metadata):
    hashes={target:sha(path) for path,target in files}
    hashfile=OUT/(name+'-SHA256.json')
    hashfile.write_text(json.dumps(hashes,indent=2))
    files=files+[(hashfile,'package-SHA256.json')]
    art=wandb.Artifact(name,type=kind,metadata=dict(training_approved=False,gpu_jobs_launched=0,**metadata))
    for path,target in files:art.add_file(str(path),name=target)
    logged=run.log_artifact(art,aliases=['cpu-remediated-not-approved']).wait()
    ref=SCOPE+'/'+logged.name
    receipt['artifacts'][name]=dict(ref=ref,digest=logged.digest,server_bytes_verified=False)
    write_receipt()
    server=wandb.Api().artifact(ref)
    assert server.digest==logged.digest
    assert server.metadata['training_approved'] is False
    assert set(server.manifest.entries)=={target for _,target in files}
    local=Path(server.download(root=str(OUT/'server-verification'/name/server.version)))
    assert json.loads((local/'package-SHA256.json').read_text())==hashes
    for target,expected in hashes.items():assert sha(local/target)==expected,target
    url=run.url.split('/runs/')[0]+f'/artifacts/{kind}/{name}/{server.version}'
    record=dict(ref=ref,digest=server.digest,url=url,files=len(files),server_bytes_verified=True)
    receipt['artifacts'][name]=record;write_receipt()
    print(json.dumps({'published_and_verified':record}),flush=True)
    return ref

datafiles=[(p,p.name) for p in sorted((OUT/'candidate').iterdir()) if p.is_file()]
datafiles += [(OUT/'README.md','README.md'),(ROOT/'conversation_sources/sources.json','sources/sources.json')]
candidate=publish('conversation-sft-cache-candidate','dataset',datafiles,
    dict(train_conversations=15375,dev_conversations=789,additional_quarantined=854,
         total_quarantined=2020,source_candidate='conversation-sft-cache-candidate:v2',
         human_review='pending',owner='Aayush Jaiswal',intended_use='internal demo only'))
linkfile=OUT/'published-candidate.json';linkfile.write_text(json.dumps(receipt['artifacts']['conversation-sft-cache-candidate'],indent=2))
auditfiles=[(p,p.name) for p in sorted(OUT.iterdir()) if p.is_file() and not p.name.endswith('-SHA256.json')]
for name in ['remediation_checks.py','remediate_candidate_cpu.py','test_remediation_checks.py',
             'final_workflow_budget.py','test_final_workflow_budget.py','finalize_cpu_remediation.py','publish_cpu_remediation.py']:
    auditfiles.append((ROOT/name,'source/'+name))
for path in sorted((ROOT/'gpu_readiness_stage_v2').rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        auditfiles.append((path,'staged-helper/'+path.relative_to(ROOT/'gpu_readiness_stage_v2').as_posix()))
auditfiles += [(ROOT/'bounded_gate_review_v1/LICENSE_AND_ATTRIBUTION_REVIEW.md','prior-license-inventory.md')]
publish('nanochat-sft-cpu-remediation','evaluation',auditfiles,
    dict(exact_candidate=candidate,cpu_unit_tests_passed=24,production_sft_gpu_proof='unverified',
         independent_human_content_review='pending',gpu_total_budget_seconds=7200,
         first_stage_budget_seconds=780,main_stage_can_start_this_turn=False))
for k,v in dict(additional_quarantined=854,total_quarantined=2020,mandatory_packet_quarantined=13,
                train_conversations=15375,dev_conversations=789,cpu_unit_tests_passed=24,
                training_approved=False,gpu_jobs_launched=0,model_calls=0).items():run.summary[k]=v
run.finish()
server_run=wandb.Api().run(SCOPE+'/'+receipt['run_id'])
assert server_run.state=='finished',server_run.state
assert server_run.config['training_approved'] is False
receipt.update(run_state_verified=server_run.state,run_config_verified=True,
               summary_readback=dict(server_run.summary),exact_candidate=candidate)
write_receipt()
print(json.dumps(receipt,indent=2),flush=True)