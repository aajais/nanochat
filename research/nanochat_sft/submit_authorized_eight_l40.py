"""One explicitly user-authorized submission; unchanged production job/safeguards.

Capacity is established by scheduler allocation/runtime hardware verification,
not fabricated operator attestation. No retries or alternate training configs.
"""
import json,sys,tempfile,time
from pathlib import Path
import wandb
ROOT=Path('/workspace/nanochat_sft');P=ROOT/'final_workflow_v2';N=ROOT/'authoritative_notices'
sys.path.insert(0,str(P));sys.path.insert(0,'/workspace/skills/wbagent/scripts')
import launch_helpers as launch
from nanochat.workflow_policy import verify_package,authorize,CAP
from nanochat.sft_runtime import atomic_json

def main():
 receipt_path=ROOT/'eight-l40-authorized-submission.json'
 if receipt_path.exists():raise FileExistsError('Submission already attempted; inspect receipt, never duplicate')
 package,identity=verify_package(P)
 approval=json.loads((N/'EXECUTION_ACCEPTANCE.json').read_text())
 assert approval['workflow_sha256']==identity
 direction={'type':'user_message','time_utc':'2026-11-23T19:56:58Z','text':'launch the gpu training',
   'context':'Repeated explicit launch instruction after disclosure of MetaMath notice gap and unverified capacity; no human content or legal review claimed.'}
 approval['launch_direction']=direction
 licensing=approval['decisions']['license_notice_disposition']
 licensing.update(status='accepted',reviewer='ARIA: documented publisher-license basis for user-authorized internal research only',
  evidence={'basis':'Use the first-party pinned dataset MIT/Apache declarations for internal research, preserve every supplied notice; not a certification of all upstream rights or complete publisher attribution.',
  'notice_gap_retained':True,'human_legal_review_performed':False,'user_launch_direction':direction,
  'prior_evidence_artifact':'wandb/nanochat-demo/nanochat-sft-authoritative-notice-disposition:v0'},
  notice_bundle={
   'Apache-2.0.txt':(N/'apache2.txt').read_text(),
   'OpenAI-GSM8K-MIT.txt':(N/'gsm8k-license.txt').read_text(),
   'Hendrycks-MATH-repository-MIT.txt':(N/'math-license.txt').read_text(),
   'Nanochat-current-upstream-MIT.txt':(N/'nanochat-license.txt').read_text(),
   'Llama-3.1-LICENSE.txt':(N/'llama31-license.txt').read_text(),
   'Llama-3.1-AUP.txt':(N/'llama31-use-policy.txt').read_text(),
   'MetaMathQA-pinned-publisher-declaration.md':(N/'metamath-card.txt').read_text(),
   'SmolTalk-pinned-publisher-declaration.md':(N/'smoltalk-card.txt').read_text(),
   'Everyday-pinned-publisher-declaration.md':(N/'everyday-card.txt').read_text(),
   'RESEARCH_USE_AND_UNRESOLVED_ATTRIBUTION.md':
    '# Llama-nanochat-research-demo — internal research only, human release hold\n\nBuilt with Llama.\n'
    'Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n\n'
    'The pinned MetaMathQA publisher declares the dataset MIT but supplies no separate attributable full MIT notice. '
    'The linked code repository Apache license is not substituted for that dataset declaration. '
    'No copyright holder has been invented. The exact publisher declaration is preserved. '
    'This unresolved attribution detail remains in the record and must be resolved before any approved release. '
    'The current upstream nanochat MIT notice is not represented as proof of the exact captured fork commit. '
    'Constituent source/version and image-rights limitations remain as documented in '
    'wandb/nanochat-demo/nanochat-sft-authoritative-notice-disposition:v0. '
    'No human content/legal review, blanket legal clearance, or external redistribution permission is claimed.\n'
  })
 approval.update(record_type='user_directed_internal_research_execution',launch_submitted=False,
  startup_status='authorized_submission_pending',human_legal_review_performed=False,human_content_review_performed=False)
 approval['decisions']['gpu_execution'].update(evidence=direction,
  capacity_verification_mode='scheduler_allocation_then_exact_eight_L40_runtime_guard',
  preallocation_capacity_verified=False)
 approval_file=ROOT/'eight-l40-execution-approval.json';atomic_json(approval_file,approval)
 authorize(approval_file,identity,package['assets'])
 receipt=json.loads((ROOT/'eight-l40-preparation-receipt.json').read_text())
 api=wandb.Api();job=api.artifact(receipt['job_ref'])
 if job.digest!=receipt['job_digest']:raise ValueError('Job digest mismatch')
 temp=ROOT/'authorized-launch-source-check';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
 source=launch.download_code_artifact(receipt['job_ref'])
 _,remote=verify_package(source['code_dir'])
 if remote!=identity or source['entrypoint']!=package['entrypoint'] or source['base_image']!=package['image']:
  raise ValueError('Remote immutable job identity mismatch')
 resources=launch._resolve_resource_args(entity='wandb',queue_name='nanochat-demo-k8s',gpus=8,cpu=8,memory='80Gi')
 spec=resources['kubernetes']['spec'];spec.update(activeDeadlineSeconds=CAP,backoffLimit=0)
 spec['template']['spec'].update(restartPolicy='Never',terminationGracePeriodSeconds=15)
 limits=spec['template']['spec']['containers'][0]['resources']
 assert str(limits['limits']['nvidia.com/gpu'])=='8' and str(limits['requests']['nvidia.com/gpu'])=='8'
 start=time.time()
 config=launch._with_wb_agent_provenance(dict(approval=approval,allocation_started_unix=start,gpus=8,
  image=package['image'],workflow_sha256=identity,total_seconds_including_prior=7200,prior_seconds=87,
  human_release_review_required=True,preallocation_capacity_verified=False))
 intent=dict(status='submission_intent_no_retry',allocation_started_unix=start,allocation_ends_unix=start+CAP,
  job_ref=receipt['job_ref'],job_digest=job.digest,workflow_sha256=identity,queue='nanochat-demo-k8s',
  gpu_count_requested=8,preallocation_capacity_verified=False)
 with receipt_path.open('x') as f:json.dump(intent,f)
 # Explicit Launch submission only; no cluster/control-plane commands.
 queued=launch.launch_add(job=receipt['job_ref'],entity='wandb',project='nanochat-demo',queue_name='nanochat-demo-k8s',
  entry_point=package['entrypoint'],resource_args=resources,config={'overrides':{'run_config':config}})
 intent.update(status='submitted',queue_item_id=queued.id);atomic_json(receipt_path,intent)
 print('SUBMITTED='+json.dumps(intent),flush=True)
 status=launch.check_launch('wandb','nanochat-demo','nanochat-demo-k8s',queued.id)
 if status:
  compact={k:v for k,v in status.items() if k!='run'}
  atomic_json(ROOT/'eight-l40-startup-status.json',compact)
  print('STARTUP_STATUS='+json.dumps(compact,default=str),flush=True)

if __name__=='__main__':main()