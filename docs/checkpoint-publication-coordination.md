# Checkpoint publication without an outstanding GPU collective

Readiness worker `9f54d075` reached step 11, then rank zero waited for
checkpoint publication while the other ranks waited in the post-publication
NCCL barrier. That barrier exhausted the communication timeout before remote
publication and full server-byte verification completed. Finite training loss
and a committed artifact were not sufficient to establish a resumable result.

The production checkpoint path now completes its existing local shard save,
then calls `coordinate_publication` on every rank. Rank zero stages, publishes,
downloads, and verifies the checkpoint using the existing manifest and SHA-256
checks. Other ranks poll an atomic status file on the already-required shared
filesystem; they do not enqueue a GPU collective while remote I/O is pending.

The status file is outside the checkpoint directory, unique to its output
directory and step, and bound to the absolute stage deadline. Rank zero publishes
success only after verification and the timed phase have completed. Peers reject
failure, stale status, or an expired deadline. A missing publisher also expires.
Only successful coordination allows RNG restoration and another training update.

On the Linux worker, a POSIX alarm bounds the publication callback without
replacing SIGTERM/SIGINT save handling or extending an existing alarm. Its private
deadline exception escapes ordinary SDK retry handlers. The existing supervisor
remains the hard process-level deadline backstop: Python signals cannot interrupt
every long-running C extension or filesystem operation. Local shard-save barriers
are unchanged by this fix. No communication timeout or allocation is increased.

## Validation

`tests/test_checkpoint_publication.py` runs two actual CPU/Gloo processes with a
one-second collective timeout and a two-second publication delay. Both must
receive the verified receipt and successfully perform the next all-reduce.
Additional tests cover publisher errors, a stalled SDK-style retry loop, missing
publishers, expired deadlines, stale receipts, alarm preservation, and a deadline
boundary that must never expose and subsequently revoke success.

Run the production-path checks with:

```bash
python -m pytest -q tests/test_checkpoint_publication.py test_workflow.py \
  test_eight_rank_contract.py test_readiness_timing.py
```

The existing workflow test suite also exercises tiny-model checkpoint/replay,
SIGTERM/deadline saves, corrupt restores, worker failure propagation, and the
actual publication path with mocked W&B transport. CPU/Gloo success is not a
claim of eight-L40/NCCL or real remote-upload readiness.

## Deployment

This source change requires a newly published job artifact and an approval record
bound to its new workflow hash. Old artifacts remain unchanged. Do not reuse the
failed run's approval or assert that its committed checkpoint passed server-byte
verification or restore replay. This PR does not submit a job or renew a budget.
