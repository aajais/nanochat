"""Finalize verification of the existing audit; never start a second run."""
import hashlib,json
from pathlib import Path
import wandb
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'bounded_gate_review_v1'
api=wandb.Api()
scope='wandb/nanochat-demo'
run_id='o45m5d4h'
ref=scope+'/nanochat-sft-bounded-gate-review:v0'
artifact=api.artifact(ref)
assert artifact.logged_by().id==run_id
assert artifact.metadata['decision']=='NO-GO' and artifact.metadata['training_approved'] is False
assert artifact.digest=='0a10d18d3c3c79ab0350a70eade914e6'
folder=Path(artifact.download(root=str(OUT/'server-verification')))
hashes=artifact.metadata['file_sha256']
assert len(hashes)==len(artifact.manifest.entries)==16
for name,digest in hashes.items():
 assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
run=api.run(scope+'/'+run_id)
assert run.state=='finished'
# Initial run upload left the public API summary empty despite local summary.json.
# One public API update did not make it visible. Do not repeat the mutation or
# claim that run-summary persistence was verified; use the verified artifact.
again=wandb.Api().run(scope+'/'+run_id)
summary=dict(again.summary)
candidate=api.artifact(scope+'/conversation-sft-cache-candidate:v2')
assert candidate.digest=='3f356c1f3cb87a99d5c95dce19dc1d66' and candidate.metadata['training_approved'] is False
receipt={'run_id':run_id,'run_url':f'https://forge.coreweave.com/wandb/{scope}/runs/{run_id}',
 'artifact_ref':ref,'artifact_url':f'https://forge.coreweave.com/wandb/{scope}/artifacts/evaluation/nanochat-sft-bounded-gate-review/v0',
 'artifact_digest':artifact.digest,'published_file_count':16,'all_published_files_server_byte_verified':True,
 'server_run_state':again.state,'run_summary_readback':summary,
 'run_summary_persistence_verified':summary.get('decision')=='NO-GO',
 'run_summary_note':('Initial public API summary readbacks were empty; a later independent read verified the intended NO-GO fields.' if summary.get('decision')=='NO-GO' else 'Public API readback stayed empty after initial upload and one update; no summary success claimed. Verified artifact files and metadata are the authoritative decision record.'),
 'candidate_approval_unchanged':True,'training_approved':False,'gpu_jobs_launched':0}
(ROOT/'published_bounded_gate_review.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)