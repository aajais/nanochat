"""Publish provisional readiness evidence, NOT a GPU submission or approval."""
import hashlib,json,os
from pathlib import Path
import wandb
ROOT=Path(__file__).resolve().parent
R=ROOT/'readiness_review_v1';B=ROOT/'readiness_bundle_v1';C=ROOT/'conversation_cache_quality_v1'
SCOPE='wandb/nanochat-demo'
receipt_path=ROOT/'published_readiness_resolution.json'
if receipt_path.exists():raise FileExistsError('Existing receipt: inspect it; do not duplicate publication')
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()
evidence=json.loads((R/'test-evidence.json').read_text())
assert evidence['targeted_eval_and_preparation_tests_passed']==71
assert evidence['fingerprint_utility_synthetic_tests_passed']==3
assert evidence['gpu_jobs_launched']==evidence['acceptance_generations']==0
assert evidence['dataset_training_approved'] is False
provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',name='dataset-evaluation-readiness-no-gpu',
               job_type='sft-readiness-audit',dir=str(ROOT),save_code=False,
               config={'_wb_agent':provenance,'launch_ready':False,'gpu_jobs_launched':0,
                       'scope':'hash checks, content quarantine, CPU tests and review packet'})
api=wandb.Api();receipt={'run_id':run.id,'run_url':run.url,'artifacts':{},'launch_ready':False,'gpu_jobs_launched':0,'acceptance_generations':0}
receipt_path.write_text(json.dumps(receipt,indent=2))
for ref in ['b1384b1-checkpoint:v2','pr1-smoke-tokenizer:v0','conversation-sft-cache-candidate:v1','nanochat-conversation-demo-evals:v0','nanochat-conversation-evaluation-runner:v1','e1384arc-benchmark-inputs:v0']:
    run.use_artifact(SCOPE+'/'+ref)

def publish(name,kind,files,metadata):
    # Per-file SHA256 allow direct verification of bytes downloaded from server.
    hashes={target:sha(path) for path,target in files}
    hashpath=R/(name+'-SHA256.json');hashpath.write_text(json.dumps(hashes,indent=2))
    files=files+[(hashpath,'package-SHA256.json')]
    artifact=wandb.Artifact(name,type=kind,metadata={'launch_ready':False,**metadata})
    for path,target in files:artifact.add_file(str(path),name=target)
    logged=run.log_artifact(artifact,aliases=['provisional-readiness-reviewed-not-approved']).wait()
    ref=SCOPE+'/'+logged.name;server=api.artifact(ref)
    assert server.digest==logged.digest
    assert set(server.manifest.entries)=={target for _,target in files}
    assert server.metadata['launch_ready'] is False
    # Download this exact version and check all bytes, not merely the mutation response.
    local=Path(server.download(root=str(ROOT/'readiness_server_verification'/name/server.version)))
    assert json.loads((local/'package-SHA256.json').read_text())==hashes
    for target,value in hashes.items():assert sha(local/target)==value
    url=run.url.split('/runs/')[0]+f'/artifacts/{kind}/{name}/{server.version}'
    result={'ref':ref,'digest':server.digest,'url':url,'files':len(files),'all_file_bytes_verified_from_server':True}
    receipt['artifacts'][name]=result;receipt_path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'published':result}),flush=True)
    return ref

fixtures=[(p,p.name) for p in sorted((B/'conversation_evals_v3').iterdir()) if p.is_file()]
evalref=publish('nanochat-conversation-demo-evals','dataset',fixtures,
                {'revision':'v3','human_review':'pending','model_outputs_generated':0,'references_thresholds_decoding_unchanged':True})
manifest=json.loads((C/'manifest.json').read_text())
manifest['conversation_evaluation_artifact']=evalref
(C/'manifest.json').write_text(json.dumps(manifest,indent=2))
datafiles=[(p,p.name) for p in sorted(C.iterdir()) if p.is_file()]
datafiles += [(ROOT/'conversation_sources/sources.json','sources/sources.json'),(ROOT/'conversation_sources/SMOLTALK_CARD.md','sources/SMOLTALK_CARD.md'),(R/'quality-filtered-cache-audit.json','audit/quality-filtered-cache-audit.json')]
dataref=publish('conversation-sft-cache-candidate','dataset',datafiles,
                {'training_approved':False,'train_conversations':16184,'dev_conversations':834,
                 'quarantined_sentence_limit_risks':1166,'human_review':'pending','untouched_holdout_comparison':'pending'})
evidence['source_artifact_publication']='published package verified by accompanying receipt; not execution approval'
(R/'test-evidence.json').write_text(json.dumps(evidence,indent=2))
codefiles=[]
for folder in ['prepared','conversation_evals_v1','conversation_evals_v3','tokenizer']:
    for path in sorted((B/folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.pkl','.md','.txt','.pt'):
            codefiles.append((path,path.relative_to(B).as_posix()))
codefiles += [(B/'freeze_conversation_evals.py','freeze_conversation_evals.py'),(B/'base-evaluation-manifest-v3.json','base-evaluation-manifest-v3.json'),(R/'test-evidence.json','evidence/test-evidence.json'),(R/'READINESS.md','READINESS.md')]
coderef=publish('nanochat-conversation-evaluation-runner','code',codefiles,
                {'fixture_revision':'v3','eval_and_preparation_tests_passed':71,'real_base_forward_calls_this_audit':0,'cpu_only_evaluator':True})
receipt['exact_candidate']=dataref;receipt['exact_fixtures']=evalref;receipt['exact_runner']=coderef
evidence['published_refs']={'candidate':dataref,'fixtures':evalref,'runner':coderef}
(R/'test-evidence.json').write_text(json.dumps(evidence,indent=2))
reviewfiles=[(path,path.name) for path in sorted(R.iterdir()) if path.is_file() and path.suffix in ('.md','.json','.py','.log') and not path.name.endswith('-SHA256.json')]
for name in ['resolve_data_readiness.py','check_readiness_sources.py','check_instruction_labels.py','quarantine_summary_risks.py','finalize_readiness_evidence.py','publish_readiness_resolution.py']:
    reviewfiles.append((ROOT/name,'audit-source/'+name))
publish('nanochat-sft-readiness-review','evaluation',reviewfiles,
        {'training_approved':False,'human_review':'pending','holdout_payloads_opened':False,'artifact_refs':evidence['published_refs']})
run.summary.update({'cpu_eval_and_preparation_tests_passed':71,'fingerprint_synthetic_tests_passed':3,
                    'train_conversations':16184,'dev_conversations':834,'quarantined_conversations':1166,
                    'gpu_jobs_launched':0,'acceptance_generations':0,'launch_ready':False})
runid=run.id;run.finish()
server_run=api.run(SCOPE+'/'+runid)
assert server_run.state=='finished' and server_run.summary['launch_ready'] is False
receipt['run_state']=server_run.state;receipt_path.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)