"""Fail-closed authorization and one clock for the complete GPU workflow."""
import json
import math
from pathlib import Path
import time
from nanochat.sft_runtime import sha256

TOTAL=7200
PRIOR=87
CAP=TOTAL-PRIOR
# Source proposal only: republishing and fresh explicit approval are required.
# Borrow from main SFT, never extend the shared total or downstream reserves.
READINESS=2400
EVALUATION=1500
PACKAGING=600
EXIT=60
TRAIN_SAVE=600
REQUIRED_DECISIONS=('retained_content','corrected_fixtures','license_notice_disposition','selection_policy','gpu_execution')

def readiness_projection(result, elapsed_seconds, remaining_seconds):
    """Measured planning estimate, never proof or permission to omit a probe."""
    costs = [entry.get('checkpoint_elapsed_seconds') for entry in result.get('uploads', [])]
    if len(costs) < 2 or any(not isinstance(x,(int,float)) or isinstance(x,bool)
            or not math.isfinite(x) or x <= 0 for x in costs):
        raise ValueError('Measured checkpoint publication timings required')
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x < 0
           for x in (elapsed_seconds, remaining_seconds)):
        raise ValueError('Invalid readiness elapsed/remaining time')
    if sum(costs) > elapsed_seconds + 1:
        raise ValueError('Checkpoint timings exceed stage elapsed time')
    overhead = max(15., elapsed_seconds-sum(costs))
    # Three publications (replay/signal/deadline), four launches (incl failure),
    # pause + verification allowance and 25% margin. Still execute every probe.
    estimate = 1.25*(3*max(costs)+4*overhead+30+60)
    return dict(required_remaining_seconds=estimate, available_seconds=remaining_seconds,
        fits=remaining_seconds >= estimate, checkpoint_seconds=max(costs),
        non_checkpoint_worker_seconds=overhead, remaining_publications=3,
        remaining_worker_launches=4, safety_multiplier=1.25,
        estimate_only=True, required_probes_unchanged=True)


def verify_package(root):
    root=Path(root)
    path=root/'WORKFLOW.json'
    package=json.loads(path.read_text())
    for name,expected in package['source_sha256'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root.resolve()) or sha256(target)!=expected:
            raise ValueError('Workflow source mismatch: '+name)
    return package,sha256(path)

def authorize(path,package_sha,asset_pins):
    a=json.loads(Path(path).read_text())
    if a.get('workflow_sha256')!=package_sha or a.get('assets')!=asset_pins:
        raise PermissionError('Approval must bind this exact workflow and pinned assets')
    if a.get('scope')!='internal demo only':raise PermissionError('Internal scope required')
    for key in REQUIRED_DECISIONS:
        decision=a.get('decisions',{}).get(key,{})
        if decision.get('status')!='accepted' or not decision.get('reviewer') or not decision.get('evidence'):
            raise PermissionError('Unresolved explicit decision: '+key)
    if a.get('owner')!='Aayush Jaiswal':raise PermissionError('Responsible owner differs')
    if a['decisions']['selection_policy'].get('policy')!='mechanical_development_then_human_release_hold':
        raise PermissionError('Explicit bounded selection policy approval required')
    licensing=a['decisions']['license_notice_disposition']
    required=('apache_components','metamath_mit','gsm8k_underlying','math_underlying','llama31_outputs','source_dependencies')
    if any(not licensing.get('dispositions',{}).get(k) for k in required):
        raise PermissionError('Specific license/notice dispositions unresolved')
    if not licensing.get('notice_bundle'):raise PermissionError('Authoritative applicable notice text missing')
    if a['decisions']['gpu_execution'].get('total_seconds_including_prior')!=TOTAL:
        raise PermissionError('Exact shared GPU budget must be accepted')
    if a['decisions']['gpu_execution'].get('gpu_count')!=8 or a['decisions']['gpu_execution'].get('budget_basis')!='shared_wall_clock':
        raise PermissionError('Explicit eight-L40 shared wall-clock allocation acceptance required')
    return a

class Allocation:
    def __init__(self,started_unix,now=time.time):
        self.now=now
        self.started=float(started_unix)
        delta=self.now()-self.started
        if not math.isfinite(delta) or delta<0 or delta>=CAP:
            raise TimeoutError('Missing, future or exhausted allocation start')
        self.end=self.started+CAP
    def end_for(self,stage):
        ends={'readiness':min(self.started+READINESS,self.end-EVALUATION-PACKAGING-EXIT),
              'main':self.end-EVALUATION-PACKAGING-EXIT,
              'evaluation':self.end-PACKAGING-EXIT,'packaging':self.end-EXIT,'exit':self.end}
        end=ends[stage]
        if self.now()>=end:raise TimeoutError('No time remains for '+stage)
        return end
    def spec(self):
        return dict(total_seconds=TOTAL,prior_seconds=PRIOR,remaining_allocation_seconds=CAP,
            readiness_cap_seconds=READINESS,evaluation_reserved_seconds=EVALUATION,
            packaging_reserved_seconds=PACKAGING,final_exit_seconds=EXIT,
            main_cap_if_full_readiness_seconds=CAP-READINESS-EVALUATION-PACKAGING-EXIT,
            main_save_upload_seconds_within_main=TRAIN_SAVE,retries=0,
            allocation_started_unix=self.started,allocation_ends_unix=self.end)

def approved_worker(args,data_manifest):
    if args.cpu_fixture:
        if args.device_type!='cpu':raise PermissionError('CPU fixture can never run on GPU')
        return
    package,identity=verify_package(Path(__file__).resolve().parents[1])
    authorize(args.approval_file,identity,package['assets'])
    limits=Allocation(args.allocation_started_unix)
    if args.deadline_unix>limits.end_for(args.stage):
        raise ValueError('Worker may not consume later-stage reserves')
    if args.dataset_artifact!=package['assets']['candidate']['ref']:
        raise ValueError('Worker candidate is not pinned v3')
    if sha256(Path(args.data_dir)/'manifest.json')!=package['candidate_manifest_sha256']:
        raise ValueError('Candidate manifest changed; approval does not rewrite dataset')
    if args.stage=='main' and (args.stop_after_step or args.pause_after_step or args.inject_failure_step or args.fixed_horizon or args.checkpoint_steps):
        raise PermissionError('Readiness controls forbidden in main')
    if args.stage=='main':
        proof=json.loads(Path(args.readiness_evidence).read_text())
        if proof.get('passed') is not True or proof.get('workflow_sha256')!=identity or proof.get('allocation_started_unix')!=limits.started:
            raise PermissionError('Main requires successful same-allocation readiness evidence')
        if proof.get('world_size')!=8 or proof.get('same_world_size_verified') is not True or proof.get('optimizer_sharding_verified') is not True:
            raise PermissionError('One-rank proof cannot establish eight-rank readiness')
        plan=json.loads(Path(args.training_plan_file).read_text())
        if plan.get('world_size')!=8 or plan.get('readiness_sha256')!=sha256(args.readiness_evidence) or plan.get('allocation_started_unix')!=limits.started or plan.get('workflow_sha256')!=identity or plan.get('reported_before_handoff') is not True:
            raise PermissionError('Verified, reported eight-rank training-token plan required')
        for key in ('production_entrypoint','native_optimizer','finite_bf16_update','periodic_complete_checkpoint',
                    'fresh_process_exact_replay','real_sigterm','real_deadline','actual_server_uploads_verified',
                    'nonzero_worker_failure_propagated'):
            if proof.get(key) is not True:raise PermissionError('Missing readiness proof: '+key)
        if not limits.started<=proof.get('completed_unix',0)<=limits.started+READINESS:
            raise PermissionError('Readiness completed outside original cap')
