"""Publish CPU-only runner code/evidence. Never starts model generation or a queue."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import wandb

root = Path(__file__).resolve().parent
scope = 'wandb/nanochat-demo'
evidence_dir = root/'evaluation_runner_evidence_v2'
evidence = json.loads((evidence_dir/'test_evidence.json').read_text())
assert evidence['targeted_tests_passed'] == 40
assert evidence['tests']['conversation_eval_runner']['run'] == 29
assert all(t['returncode'] == 0 for t in evidence['tests'].values())
assert evidence['model_evaluations_run'] == evidence['gpu_jobs_launched'] == 0
assert evidence['synthetic_smoke']['release_decision'] == 'synthetic_not_release_evidence'
assert evidence['synthetic_smoke']['scripted_turns'] == 180
for name, row in evidence['tests'].items():
    assert hashlib.sha256((evidence_dir/(name+'.log')).read_bytes()).hexdigest() == row['log_sha256']

files = []
for folder in ('prepared', 'conversation_evals_v1', 'tokenizer'):
    for path in (root/folder).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.txt', '.json', '.md', '.pkl', '.pt'):
            # No base weights, data cache or credentials are inside these folders.
            files.append((path, path.relative_to(root).as_posix()))
for name in ('freeze_conversation_evals.py', 'validate_evaluation_runner.py',
             'publish_evaluation_runner.py', 'PREPARATION_STATUS.md', 'EVALUATION_RUNNER.md'):
    files.append((root/name, name))
for path in evidence_dir.rglob('*'):
    if path.is_file() and 'scratch' not in path.relative_to(evidence_dir).parts:
        files.append((path, 'evidence/'+path.relative_to(evidence_dir).as_posix()))
hashes = {target: hashlib.sha256(path.read_bytes()).hexdigest() for path, target in files}
(root/'evaluation_runner_SHA256.json').write_text(json.dumps(hashes, indent=2))
files.append((root/'evaluation_runner_SHA256.json', 'SHA256.json'))

provenance = {key: os.environ[var] for key, var in
              [('thread_id', 'WANDB_ARIA_THREAD_ID'), ('turn_id', 'WANDB_ARIA_TURN_ID')]
              if var in os.environ}
runtime = {'python': platform.python_version(), **{name: importlib.metadata.version(name)
           for name in ('torch', 'numpy', 'tiktoken', 'wandb')}}
run = wandb.init(entity='wandb', project='nanochat-demo',
                 name='conversation-evaluation-runner-synthetic-no-gpu',
                 job_type='evaluation-runner-preparation', dir=str(root), save_code=False,
                 config={'_wb_agent': provenance, 'scope': 'CPU synthetic tests, no real-model evaluation',
                         'launch_ready': False, 'training_approved': False, 'runtime_tested': runtime})
for ref in ('nanochat-conversation-demo-evals:v0', 'conversation-sft-preparation:v2', 'pr1-smoke-tokenizer:v0'):
    run.use_artifact(scope+'/'+ref)

artifact = wandb.Artifact('nanochat-conversation-evaluation-runner', type='code',
    metadata={'launch_ready': False, 'training_approved': False, 'release_approved': False,
              'targeted_tests_passed': 40, 'runner_tests_passed': 29,
              'synthetic_cases': 100, 'synthetic_responses': 180,
              'model_evaluations_run': 0, 'gpu_jobs_launched': 0,
              'native_checkpoint_generation_tested': False,
              'human_fixture_review': 'pending', 'ledger_is_local_not_access_control': True,
              'fixture_artifact': scope+'/nanochat-conversation-demo-evals:v0',
              'runtime_tested': runtime, 'immutable_training_container_pinned': False})
for path, target in files:
    artifact.add_file(str(path), name=target)
logged = run.log_artifact(artifact, aliases=['synthetic-tested-not-launch-ready']).wait()
ref, expected_digest = scope+'/'+logged.name, logged.digest
run.summary.update({'cpu_targeted_tests_passed': 40, 'runner_tests_passed': 29,
                    'synthetic_cases': 100, 'synthetic_responses': 180,
                    'model_evaluations_run': 0, 'gpu_jobs_launched': 0,
                    'launch_ready': False, 'release_approved': False,
                    'runner_artifact': ref})
run_id, run_url = run.id, run.url
run.finish()

# New client: never rely on the just-mutated object's in-memory state.
api = wandb.Api(timeout=45)
server = api.artifact(ref, type='code')
assert server.digest == expected_digest
assert set(server.manifest.entries) == {target for _, target in files}
assert server.metadata['targeted_tests_passed'] == 40
assert server.metadata['model_evaluations_run'] == 0
assert server.metadata['launch_ready'] is False
record = api.run(scope+'/'+run_id)
assert record.state == 'finished', record.state
assert record.summary['cpu_targeted_tests_passed'] == 40
assert record.summary['runner_artifact'] == ref
url = run_url.split('/runs/')[0]+'/artifacts/code/nanochat-conversation-evaluation-runner/'+server.version
published = {'run_id': run_id, 'run_url': run_url, 'run_state': record.state,
             'artifact_ref': ref, 'artifact_url': url, 'digest': server.digest,
             'file_count': len(files), 'verified_from_server': True}
(root/'published_evaluation_runner.json').write_text(json.dumps(published, indent=2))
print(json.dumps(published), flush=True)