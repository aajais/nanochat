"""Verify publication; repair only missing summary on this preparation run."""
import hashlib
import json
from pathlib import Path
import wandb

root = Path(__file__).resolve().parent
scope = 'wandb/nanochat-demo'
ref = scope+'/nanochat-conversation-evaluation-runner:v1'
api = wandb.Api(timeout=45)
artifact = api.artifact(ref, type='code')
run = api.run(scope+'/h9w52524')
assert run.name == 'conversation-checkpoint-integration-no-gpu'
assert run.state == 'finished'
assert artifact.digest == '91e0698e437f5fc049d89ef31a85e1d1'
assert artifact.metadata['targeted_cpu_tests_passed'] == 66
assert artifact.logged_by().id == run.id
expected = json.loads((root/'checkpoint_integration_evidence_v2/package-SHA256.json').read_text())
assert set(artifact.manifest.entries) == set(expected) | {'SHA256.json'}
values = dict(artifact.metadata, runner_artifact=ref, fixture_blocking_families=2,
              development_cases_with_scoring_ambiguities=8)
if not dict(run.summary):
    run.summary.update(values)

fresh = wandb.Api(timeout=45)
record = fresh.run(scope+'/h9w52524')
assert record.state == 'finished'
assert record.summary['runner_artifact'] == ref
assert record.summary['targeted_cpu_tests_passed'] == 66
assert record.summary['model_evaluations_run'] == record.summary['gpu_jobs_launched'] == 0
assert record.summary['training_approved'] is record.summary['launch_ready'] is False
target = root/'checks/published_checkpoint_runner_v1'
downloaded = []
for name in ('SHA256.json', 'CHECKPOINT_INTEGRATION.md',
             'prepared/nanochat/conversation_eval_checkpoint.py',
             'evidence/test-evidence.json', 'evidence/base-evaluation-manifest.json'):
    path = Path(artifact.get_path(name).download(root=str(target)))
    if name == 'SHA256.json':
        assert json.loads(path.read_text()) == expected
    else:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[name]
    downloaded.append(name)
receipt = dict(run_id=record.id, run_url=record.url, run_state=record.state,
               artifact_ref=ref, digest=artifact.digest,
               artifact_url=record.url.split('/runs/')[0]+'/artifacts/code/nanochat-conversation-evaluation-runner/v1',
               file_count=len(artifact.manifest.entries), downloaded_files_verified=downloaded,
               targeted_cpu_tests_passed=66,
               checkpoint_manifest_sha256=artifact.metadata['checkpoint_manifest_sha256'],
               verified_from_server=True, human_fixture_review='pending',
               model_evaluations_run=0, gpu_jobs_launched=0,
               release_approved=False, training_approved=False)
(root/'published_checkpoint_integration.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)