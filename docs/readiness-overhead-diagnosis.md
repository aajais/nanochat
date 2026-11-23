# Readiness timeout diagnosis and source-only remediation

Parent `h1fkgo63` failed during fresh-process restore; child `26bb573f` only
completed the initial twelve-update readiness segment. No main-SFT handoff.
Evidence: `wandb/nanochat-demo/h1fkgo63-failure-evidence:v0`, child history and
timestamped console logs. No private approval or checkpoint payload is included.

## Measured timeline (2026-11-23 UTC)

| Event | Time |
| --- | --- |
| Step 11 logged | 20:00:52.668 |
| Artifact staging completed, checkpoint 11 | 20:01:08.232 |
| Published checkpoint 11 download began | 20:05:46.726 |
| Download completed | 20:05:50.070 |
| Step 12 logged | 20:05:57.441 |
| Artifact staging completed, checkpoint 12 | 20:06:11.258 |
| Published checkpoint 12 download began | 20:10:52.202 |
| Download completed | 20:10:57.159 |
| Readiness hard deadline | 20:11:18.806 |

Each complete checkpoint is ~9.97GB (4.23GB model, eight ~717MB rank files,
small metadata). Publication/finalization after staging cost ~279–281s each;
downloads took ~3.3s and ~5s. The first checkpoint step-to-step gap was ~304.8s,
whereas all twelve timed updates totalled ~7.93s. The dominant observed interval
was publication/finalization, not compute or download. These logs do not isolate
network transfer versus backend finalization/queueing inside that interval.

## Proposed changes, not applied to the failed run

- Flushed per-rank phase start/end/error timings for save, artifact staging,
  publication wait, download, server-byte hashing, restore, base loading/hashing,
  W&B initialization/finalization and cleanup; supervisor child-process timings.
  Records contain no arguments, environment values or model responses.
- Each upload receipt carries checkpoint elapsed time and byte count.
- Before spawning the restored worker, estimate the remaining three complete
  publications and four launches with measured overhead, pause allowance and
  25% margin. Record and reject insufficient headroom; never omit a probe.
- Repartition the unchanged shared budget: readiness ceiling 780s ->2400s,
  main cap at full readiness 4173s ->2553s. Evaluation1500s, packaging600s,
  exit60s and main's internal save/upload600s remain reserved. Thus at full
  readiness, main has 1953s before its save reserve and other overhead. This is
  an explicit training-time tradeoff, not a speedup or guaranteed fitting budget.

All checkpoint bytes/shards, server roundtrip checks, exact replay, actual
signal/deadline tests and nonzero failure propagation remain mandatory.
There is no automatic retry or allocation-clock reset. An estimate is not GPU
readiness or quality evidence. No GPU was launched for this patch.

Future execution requires fresh publication, source-hash-bound approval and
accounting for already-spent compute; the 87s historical constant is not a
license to erase the failed allocation or authorize a new one. The proposed
repartition is code-only until separately approved. Existing runs are unchanged.