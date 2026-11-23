"""Finalize local CPU evidence and hash-pinned, non-launchable runner preparation."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
from final_workflow_budget import specification

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'cpu_remediation_v2'
STAGE=ROOT/'gpu_readiness_stage_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2))

image=json.loads((ROOT/'pinned_training_image.json').read_text())['image']
stage_files={str(p.relative_to(STAGE)):sha(p) for p in sorted(STAGE.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.name!='SMOKE_PACKAGE.json'}
manifest=dict(image=image,source_files_sha256=stage_files,authorized_queue='nanochat-demo-k8s',gpus=1,
    max_job_seconds=780,supervisor_alarm_seconds=750,internal_budget_seconds=730,
    aggregate_gpu_total_seconds=7200,prior_failed_attempt_charge_seconds=87,
    smoke_only=True,main_sft_auto_launch=False,synthetic_fixture=True,
    no_acceptance_evaluation=True,launch_approved=False,
    purpose='Bounded helper probe only; actual production SFT entrypoint proof still required')
write(STAGE/'SMOKE_PACKAGE.json',manifest)
# Assert the actual staged supervisor no longer contains the old timeout values.
tree=ast.parse((STAGE/'gpu_smoke_supervisor.py').read_text())
numbers={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) is int}
assert not numbers & {850,870,900}, numbers
assert {730,750,780} <= numbers
old_supervisor=ROOT/'gpu_smoke_job_v1/gpu_smoke_supervisor.py'
old_manifest=json.loads((ROOT/'gpu_smoke_job_v1/SMOKE_PACKAGE.json').read_text())
assert sha(old_supervisor)==old_manifest['source_files_sha256']['gpu_smoke_supervisor.py']

class ReachedInit(Exception): pass
old_cwd=os.getcwd()
try:
    os.chdir(STAGE)
    spec=importlib.util.spec_from_file_location('staged_remediation_probe',STAGE/'gpu_smoke_supervisor.py')
    sup=importlib.util.module_from_spec(spec)
    with patch.dict(os.environ,{'NANOCHAT_ALLOCATION_STARTED_UNIX':str(time.time()-90)}),patch('signal.alarm') as alarm,patch('pathlib.Path.mkdir'):
        spec.loader.exec_module(sup)
    assert 650 <= alarm.call_args.args[0] <= 660
    assert 480 <= sup.budget() <= 490  # 730 - 90 startup - 150 reserve.
    with patch.object(sup.wandb,'init',side_effect=ReachedInit) as initialize:
        try:sup.main()
        except ReachedInit:pass
        else:raise AssertionError('Initialization boundary not reached')
        initialize.assert_called_once()
    with patch.object(sup,'sha256',return_value='tampered'),patch.object(sup.wandb,'init') as initialize:
        try:sup.main()
        except AssertionError as error:assert 'Code hash mismatch' in str(error)
        else:raise AssertionError('Tampered source accepted')
        initialize.assert_not_called()
finally:os.chdir(old_cwd)

owner=dict(responsible_owner='Aayush Jaiswal',intended_use='internal demo only',
    source='Explicit user message 2026-11-23T17:06:36Z',
    owner_and_scope_confirmed=True,
    permission_for_cpu_remediation=True,gpu_launch_permission_this_turn=False,
    explicit_disposition_of_outstanding_license_notice_questions='not supplied; naming the owner does not fabricate a legal conclusion',
    outstanding=['authoritative MIT attribution/copyright notice','GSM8K/MATH constituent-source terms and notices','applicable generator/output-term disposition','approval of attribution bundle for internal use'],
    license_inventory_ref='wandb/nanochat-demo/nanochat-sft-bounded-gate-review:v0',
    no_public_serving_or_weights_or_dataset_redistribution_authorized=True)
write(OUT/'owner-and-use-record.json',owner)
write(OUT/'final-workflow-budget.json',specification())
write(OUT/'runner-preparation-evidence.json',dict(
    staged_helper_source_hashes_validated=True, tampered_source_rejected_before_run_init=True,
    startup_time_subtracted=True, staged_helper_cap_seconds=780,alarm_seconds=750,internal_budget_seconds=730,
    old_published_helper_unchanged=True,new_helper_not_enqueued=True,
    final_budget_handoff_and_failure_contract_unit_tested=True,
    production_sft_gpu_save_resume_shutdown_upload='UNVERIFIED',
    production_entrypoint_stage_integration='Still required before launch; helper replay does not satisfy it',
    executable_final_gpu_job_created=False,gpu_jobs_launched=0))
print(json.dumps(dict(stage_manifest_validated=True,owner_recorded=True,budget=specification()),indent=2),flush=True)