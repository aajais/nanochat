"""Publish CPU-only integration evidence as a new version; no queue operations."""
import hashlib
import json
import os
from pathlib import Path
import wandb

root = Path(__file__).resolve().parent
evidence_dir = root/'checkpoint_integration_evidence_v2'
evidence = json.loads((evidence_dir/'test-evidence.json').read_text())
assert evidence['targeted_tests_passed'] == 66
assert evidence['tests']['conversation_eval_checkpoint']['run'] == 26
assert all(row['returncode'] == 0 for row in evidence['tests'].values())
assert evidence['verified_base_load']['status'] == 'verified_CPU_load_only_no_forward'
assert evidence['base_model_generation_calls'] == evidence['gpu_jobs_launched'] == 0
assert evidence['model_evaluations_run'] == evidence['acceptance_generations'] == 0
for name, row in evidence['tests'].items():
    assert hashlib.sha256((evidence_dir/(name+'.log')).read_bytes()).hexdigest() == row['log_sha256']

files = []
for folder in ('prepared', 'conversation_evals_v1', 'tokenizer'):
    for path in sorted((root/folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.json', '.pkl', '.md', '.txt', '.pt'):
            files.append((path, path.relative_to(root).as_posix()))
for name in ('CHECKPOINT_INTEGRATION.md', 'EVALUATION_RUNNER.md', 'freeze_conversation_evals.py',
             'validate_checkpoint_integration.py', 'validate_evaluation_runner.py',
             'publish_checkpoint_integration.py', 'base_verification.json',
             'base_metadata/meta_133632.json'):
    files.append((root/name, name))
for path in sorted(evidence_dir.rglob('*')):
    if path.is_file() and path.name != 'package-SHA256.json':
        files.append((path, 'evidence/'+path.relative_to(evidence_dir).as_posix()))
hashes = {target: hashlib.sha256(path.read_bytes()).hexdigest() for path, target in files}
manifest = evidence_dir/'package-SHA256.json'
manifest.write_text(json.dumps(hashes, indent=2))
files.append((manifest, 'SHA256.json'))

provenance = {key: os.environ[var] for key, var in
              [('thread_id', 'WANDB_ARIA_THREAD_ID'), ('turn_id', 'WANDB_ARIA_TURN_ID')]
              if var in os.environ}
run = wandb.init(entity='wandb', project='nanochat-demo',
                 name='conversation-checkpoint-integration-no-gpu',
                 job_type='evaluation-runner-preparation', dir=str(root), save_code=False,
                 config={'_wb_agent': provenance, 'scope': 'CPU load/integration; no real-model generation',
                         'training_approved': False, 'launch_ready': False,
                         'checkpoint_manifest_sha256': evidence['checkpoint_manifest_sha256']})
for ref in ('b1384b1-checkpoint:v2', 'nanochat-conversation-evaluation-runner:v0',
            'nanochat-conversation-demo-evals:v0', 'pr1-smoke-tokenizer:v0'):
    run.use_artifact('wandb/nanochat-demo/'+ref)
metadata = dict(targeted_cpu_tests_passed=66, checkpoint_loader_tests_passed=26,
                base_checkpoint_CPU_load_verified=True, base_model_forward_calls=0,
                base_model_generation_calls=0, model_evaluations_run=0,
                gpu_jobs_launched=0, acceptance_generations=0,
                fixture_files_changed=False, fixture_review='AI_review_identified_two_blocking_families',
                human_fixture_review='pending', training_approved=False,
                launch_ready=False, release_approved=False,
                checkpoint_manifest_sha256=evidence['checkpoint_manifest_sha256'])
artifact = wandb.Artifact('nanochat-conversation-evaluation-runner', type='code', metadata=metadata)
for path, target in files:
    artifact.add_file(str(path), name=target)
logged = run.log_artifact(artifact, aliases=['cpu-loader-tested-fixture-review-blocked']).wait()
ref = 'wandb/nanochat-demo/'+logged.name
expected_digest = logged.digest
run.summary.update(dict(metadata, runner_artifact=ref, fixture_blocking_families=2,
                        development_cases_with_scoring_ambiguities=8))
run_id, run_url = run.id, run.url
run.finish()

# Independent server re-read, not the just-written in-memory objects.
api = wandb.Api(timeout=45)
saved = api.artifact(ref, type='code')
record = api.run('wandb/nanochat-demo/'+run_id)
assert saved.digest == expected_digest
assert set(saved.manifest.entries) == {target for _, target in files}
assert saved.metadata['targeted_cpu_tests_passed'] == 66
assert saved.metadata['human_fixture_review'] == 'pending'
assert record.state == 'finished'
assert record.summary['runner_artifact'] == ref
assert record.summary['targeted_cpu_tests_passed'] == 66
assert record.summary['model_evaluations_run'] == record.summary['gpu_jobs_launched'] == 0
assert record.summary['launch_ready'] is False
receipt = dict(run_id=run_id, run_url=run_url, run_state=record.state,
               artifact_ref=ref, digest=saved.digest, file_count=len(files),
               artifact_url=run_url.split('/runs/')[0]+'/artifacts/code/nanochat-conversation-evaluation-runner/'+saved.version,
               checkpoint_manifest_sha256=evidence['checkpoint_manifest_sha256'],
               verified_from_server=True)
(root/'published_checkpoint_integration.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)