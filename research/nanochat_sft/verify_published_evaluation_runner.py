"""Read-only server verification after publication; write local verification receipt."""
import hashlib
import json
from pathlib import Path
import wandb

root = Path(__file__).resolve().parent
api = wandb.Api(timeout=30)
ref = 'wandb/nanochat-demo/nanochat-conversation-evaluation-runner:v0'
artifact = api.artifact(ref, type='code')
run = api.run('wandb/nanochat-demo/5ntx0oan')
assert run.state == 'finished'
assert run.summary['cpu_targeted_tests_passed'] == 40
assert run.summary['runner_tests_passed'] == 29
assert run.summary['runner_artifact'] == ref
assert run.summary['model_evaluations_run'] == run.summary['gpu_jobs_launched'] == 0
assert run.summary['launch_ready'] is False
assert artifact.digest == '0a4fe6fbfd1ec95744367cb056c26f79'
assert artifact.metadata['targeted_tests_passed'] == 40
assert artifact.metadata['launch_ready'] is artifact.metadata['release_approved'] is False
target = root/'checks/published_runner_v0'
manifest = json.loads(Path(artifact.get_path('SHA256.json').download(root=str(target))).read_text())
assert set(artifact.manifest.entries) == set(manifest) | {'SHA256.json'}
assert manifest == json.loads((root/'evaluation_runner_SHA256.json').read_text())
verified = []
for name in ('prepared/nanochat/conversation_eval.py', 'EVALUATION_RUNNER.md', 'evidence/test_evidence.json'):
    path = Path(artifact.get_path(name).download(root=str(target)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name]
    verified.append(name)
receipt = {'run_id': run.id, 'run_url': run.url, 'run_state': run.state,
           'artifact_ref': ref,
           'artifact_url': run.url.split('/runs/')[0]+'/artifacts/code/nanochat-conversation-evaluation-runner/v0',
           'digest': artifact.digest, 'file_count': len(artifact.manifest.entries),
           'manifest_verified_from_server': True, 'downloaded_files_verified': verified,
           'summary_verified_from_server': True, 'targeted_cpu_tests_passed': 40,
           'gpu_jobs_launched': 0, 'model_evaluations_run': 0, 'launch_ready': False}
(root/'published_evaluation_runner.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)