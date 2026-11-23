"""One conservative GPU allocation budget for readiness + main SFT.

No queue or cluster access. The final supervisor must create exactly one Budget
at allocation start, include startup, and pass its absolute deadline into every
worker. Readiness success never resets the clock. Old standalone smoke budgets
are not additive authorizations.
"""
from dataclasses import dataclass
import math
import time

TOTAL_SECONDS = 7200
PRIOR_ATTEMPT_CHARGE_SECONDS = 87  # Previous inclusive queue/startup upper bound.
ALLOCATION_CAP_SECONDS = TOTAL_SECONDS - PRIOR_ATTEMPT_CHARGE_SECONDS
READINESS_CAP_SECONDS = 780  # Startup, check, shutdown, verified upload and exit.
READINESS_EXIT_RESERVE_SECONDS = 150
MAIN_EXIT_RESERVE_SECONDS = 600
OUTER_GRACE_SECONDS = 15
REQUIRED_PROOFS = (
    'exact_base_and_tokenizer_loaded', 'production_sft_entrypoint_exercised',
    'native_bf16_update_finite', 'periodic_complete_checkpoint',
    'fresh_process_restore', 'next_update_matches_uninterrupted',
    'optimizer_loader_rng_schedule_restored', 'real_signal_shutdown_completed',
    'real_deadline_shutdown_completed', 'checkpoint_upload_server_verified',
    'worker_failure_propagates',
)

@dataclass
class Budget:
    allocation_started_unix: float
    allocation_started_monotonic: float
    monotonic: object = time.monotonic

    @classmethod
    def start(cls):
        return cls(time.time(), time.monotonic())

    def elapsed(self):
        elapsed = float(self.monotonic()) - self.allocation_started_monotonic
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError('Invalid clock; fail closed')
        return elapsed

    def remaining(self):
        return max(0., ALLOCATION_CAP_SECONDS-self.elapsed())

    def stage_limits(self, stage):
        if stage not in ('readiness','main'):
            raise ValueError(stage)
        elapsed = self.elapsed()
        hard_remaining = min(READINESS_CAP_SECONDS-elapsed, ALLOCATION_CAP_SECONDS-elapsed) if stage == 'readiness' else ALLOCATION_CAP_SECONDS-elapsed
        reserve = READINESS_EXIT_RESERVE_SECONDS if stage == 'readiness' else MAIN_EXIT_RESERVE_SECONDS
        work = hard_remaining-reserve-OUTER_GRACE_SECONDS
        if work <= 0:
            raise TimeoutError('No safe work time remains; do not start stage')
        hard_end = self.allocation_started_unix+elapsed+hard_remaining
        return dict(stage=stage, remaining_hard_seconds=hard_remaining,
            work_seconds_before_reserve=work,
            worker_deadline_unix=hard_end-OUTER_GRACE_SECONDS,
            stop_work_unix=hard_end-OUTER_GRACE_SECONDS-reserve,
            save_upload_exit_reserve_seconds=reserve,
            absolute_allocation_deadline_unix=self.allocation_started_unix+ALLOCATION_CAP_SECONDS,
            total_authorized_seconds=TOTAL_SECONDS,
            charged_prior_attempt_seconds=PRIOR_ATTEMPT_CHARGE_SECONDS)

    def authorize_main(self, evidence):
        # This validates a report's contract, not its truth. The production
        # supervisor must verify source/asset identities, server receipts and
        # actual child exit status before constructing that report.
        missing = [key for key in REQUIRED_PROOFS if evidence.get(key) is not True]
        if missing or evidence.get('passed') is not True:
            raise RuntimeError('Readiness failed/unverified; no main training: '+','.join(missing))
        if evidence.get('completed_elapsed_seconds', float('inf')) > READINESS_CAP_SECONDS or self.elapsed() > READINESS_CAP_SECONDS:
            raise TimeoutError('Readiness exceeded shared cap; no main training')
        return self.stage_limits('main')


def gated_stages(budget, readiness, main):
    """Only preverified/approved final supervisor may call; no retries or reset."""
    report = readiness(budget.stage_limits('readiness'))
    limits = budget.authorize_main(report)  # A failed/partial check aborts here.
    return main(limits)


def specification():
    return dict(total_seconds_including_prior_attempt=TOTAL_SECONDS,
        prior_failed_attempt_upper_bound_charged_seconds=PRIOR_ATTEMPT_CHARGE_SECONDS,
        remaining_allocation_hard_cap_seconds=ALLOCATION_CAP_SECONDS,
        readiness_first_stage_total_cap_seconds=READINESS_CAP_SECONDS,
        readiness_internal_work_cap_before_startup_seconds=READINESS_CAP_SECONDS-READINESS_EXIT_RESERVE_SECONDS-OUTER_GRACE_SECONDS,
        readiness_save_upload_exit_reserve_seconds=READINESS_EXIT_RESERVE_SECONDS,
        main_save_upload_exit_reserve_seconds=MAIN_EXIT_RESERVE_SECONDS,
        main_seconds_if_readiness_uses_full_cap=ALLOCATION_CAP_SECONDS-READINESS_CAP_SECONDS,
        stage_failure='Nonzero and stop before main; no automatic retry/correction',
        shared_absolute_deadline=True, clock_reset_between_stages=False,
        startup_and_upload_included=True, gpu_count=1, queue='nanochat-demo-k8s',
        old_standalone_900_870_850_second_smoke_package_superseded_not_launchable=True,
        kubernetes_required=dict(activeDeadlineSeconds=ALLOCATION_CAP_SECONDS,backoffLimit=0,
            restartPolicy='Never',terminationGracePeriodSeconds=OUTER_GRACE_SECONDS),
        required_readiness_proofs=list(REQUIRED_PROOFS),
        production_runner_proof='pending; CPU contract tests are not GPU evidence',
        executable_gpu_workflow_approved=False)