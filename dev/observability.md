# Forge observability and remote rollouts

Opt-in integrations; `--run dummy` and local rollouts remain the defaults.
Install the existing environment first, then layer on the integrations:

```bash
uv sync --extra gpu --group dev
uv pip install -r requirements-observability.txt
export WANDB_ENTITY=your-team WANDB_PROJECT=nanochat
export NANOCHAT_WEAVE=1
python -m scripts.chat_eval --source sft --task-name GSM8K --max-problems 8 --run eval-smoke
```

Use your runtime secret mechanism for `WANDB_API_KEY` and `CWSANDBOX_API_KEY`;
never put credentials in CLI arguments, images, snapshots, or source control.
The optional requirements are separate to preserve the baseline `uv.lock`.
Rerunning `uv sync` can remove them; reinstall the requirements afterward.

## What is logged

* **Pretraining/SFT:** existing loss, validation BPB, CORE/ChatCORE accuracy,
  token throughput, MFU (existing percentage scale), step time, total FLOPs/time;
  plus raw microbatch loss, resolved model/CLI config, effective batch/gradient
  accumulation, dtype/device/world size, optimizer group LRs, rank-zero gradient
  norm/nonfinite count, and CUDA allocated/reserved/peak memory. Diagnostics use
  existing logging cadence (100 pretrain / 10 SFT steps), after unscaling and
  before optimizer mutation. No `wandb.watch` or per-step tensor histograms.
* **RL:** existing global reward/pass@k/sequence length, plus rank-zero policy
  objective, generation/reward/batch-preparation time, total update wall time,
  gradient/memory/LR diagnostics. Remote mode separately logs weight-sync time.
* **Standalone base/chat eval:** checkpoint/model config, BPB/accuracy/CORE,
  per-task wall time and bounded per-example tables. Stable task/example/sample
  IDs support comparisons across runs and repeated evaluation steps.
* **Eval Tables:** `wandb.EvalTable` on the optional SDK version; old base SDKs
  fall back to `wandb.Table` with a warning. The feature is in Preview.
  Defaults: `NANOCHAT_EVAL_ROWS=64`, `NANOCHAT_TRACE_EXAMPLES=8` per task/eval,
  rank zero only. Table averages cover the logged sample, not the full dataset.
  Full-dataset scalar metrics still use the existing distributed reductions.
* **Weave:** `nanochat.evaluate` with sampled generation/categorical-forward,
  CORE example, scoring, and remote code-execution spans, correlated by W&B run
  ID. Remote rollout request spans capture policy version, worker ID, duration,
  and completion-token count. No model tensors or per-token forward traces.
  These are Calls traces, not synthetic Agent Lens conversations.
* **Privacy:** prompt/reference/completion content is off by default. Set
  `NANOCHAT_LOG_CONTENT=1` only for approved datasets. IDs, correctness, counts
  and timing remain available without content. Generation seconds in each
  sample row is the whole batched generation time, not per-sequence latency.
* **Artifacts:** `NANOCHAT_LOG_CHECKPOINTS=1` logs model weights + metadata at
  existing save points. Upload/storage costs apply. Optimizer shards and the
  tokenizer are intentionally excluded; this isn't a full resume bundle.
  Promote reviewed versions to your Registry through the usual artifact UI;
  no automatic production promotion or extra project/view is created.

W&B's normal system monitor provides CPU/GPU utilization where supported.
Gradient and timing metrics explicitly labeled `rank0` are not global maxima.

## Warm GPU rollout pool (GSM8K RL only)

Build/publish the worker image from this revision using
`docker build -f dev/Dockerfile.rollout -t YOUR_REGISTRY/nanochat-rollout .`.
Set `NANOCHAT_ROLLOUT_IMAGE` to the **published `@sha256:` digest**, not a tag.
System libraries and Python packages belong in that image; build tools are
included through the PyTorch development base. Pin/customize `BASE_IMAGE` for
your reviewed runtime. No runtime package downloads or public ports are used.

```bash
# Uses any eligible internal CKS runner; optionally pin one or more runner IDs.
python -m scripts.chat_rl --run rl-remote \
  --rollout-workers 2 --rollout-placement cks \
  --rollout-save-snapshot --rollout-lifetime-seconds 10800
```

`--rollout-runner-id` is repeatable and requires strict CKS placement.
Without pinning, `--rollout-placement cks_then_serverless` allows SDK placement
spillover on eligible creation failures; `serverless` selects serverless only.
CKS/spillover uses `CWSANDBOX_API_KEY`; serverless also supports W&B auth.
Resource limits, GPU/Snapshot enablement and runner policy still apply.

Each training rank owns `--rollout-workers` persistent one-GPU workers, bounded
to 32 per rank. Total rollout GPU reservation is workers × training ranks,
**in addition to training GPUs**. CPU/memory and timeouts have explicit flags.
Pick a lifetime longer than the job; expiry fails the run rather than silently
creating replacement capacity. Normal exit/request failure stops the pool;
hard termination relies on the configured lifetime cap. If stop fails, the
warning includes the sandbox ID for manual cleanup in Forge Sandboxes.

At each optimizer step, every worker receives the current state dict (safe
`weights_only=True` load), then examples execute in parallel across workers.
Only one request per worker is outstanding. IDs, policy version, sample counts,
prompt prefixes, token bounds and masks are checked before training consumes
the result. There is no optimizer update between sync and rollout consumption.
Worker failures propagate; there is no fallback to unsynchronized/local output.
Model weights, tokenizer, and source image are trusted inputs. Calculator tool
expressions use a bounded arithmetic/string-count AST interpreter in workers,
never Python `eval`; unsupported expressions produce no tool result.

### Snapshot once, reuse the environment

`--rollout-save-snapshot` captures one worker's assets/caches after the first
completed rollout, on rank zero only. Its READY ID is printed and stored in
the W&B summary at `rollout/environment_snapshot_id`. Future runs reuse it:

```bash
python -m scripts.chat_rl --run rl-restored \
  --rollout-workers 2 --rollout-placement cks \
  --rollout-snapshot-id "$NANOCHAT_ROLLOUT_SNAPSHOT_ID"
```

Each worker restores an independent fork of the same immutable snapshot.
`--rollout-snapshot-size` defaults to `20Gi`. The mount holds the tokenizer and
HF/Triton/TorchInductor/CUDA filesystem caches. Restore checks the image digest,
dtype, and tokenizer hash. Rebuild after changing any of these. Caches may be
recompiled on a different GPU/driver; use homogeneous runners when comparing
numerics or latency. The worker reports its actual GPU and attention backend.

FSS captures **files inside the mount only**, not process/GPU memory, packages
installed outside the mount, KV cache, or a live model. Warm workers eliminate
repeated startup *within* a run; the snapshot helps subsequent pool creation.
Policy weights and IPC are outside the snapshot, so every new pool must sync
fresh weights before generating. Snapshots remain until explicitly removed by
their owner; the integration never deletes saved snapshots.

Full weight transfer every update can outweigh rollout parallelism for a small
model/batch. Compare `rollout/weight_sync_seconds_rank0`, rollout wall time and
total training wall time against local mode; this PR does not assert a speedup.

## Generated Python (HumanEval)

Set `NANOCHAT_EXECUTION_BACKEND=cwsandbox` to run each generated program in a
fresh **serverless** CPU sandbox with denied ingress/egress, bounded CPU/memory,
timeout/lifetime, and no forwarded trainer secrets. Service failures propagate;
generated code never falls back to the trainer. `NANOCHAT_SANDBOX_IMAGE` selects
the Python image (default `python:3.11-slim`; pin an approved digest in production).
Fresh per-program isolation prioritizes security over startup speed; don't reuse
these sandboxes/snapshots between untrusted programs. The existing local backend
is retained for compatibility and is explicitly **not** a security boundary.
CKS GPU containers are for trusted generation only, not untrusted Python.

## Validation

```bash
python -m unittest discover -s tests -p 'test_observability.py' -v
python -m compileall -q nanochat scripts
```

CPU contracts use actual EvalTable serialization, mocked Weave/Sandboxes, and a
tiny Torch model. Production receipt/UI rendering, image build, GPU inference,
snapshot restore timing, throughput and multi-rank behavior need runtime smoke
tests on the internal runners. Start with one rank/worker and a short RL run,
check table rows/trace children and snapshot ID in Forge, then repeat with
`--rollout-snapshot-id` before scaling. No production run or sandbox is started
by the tests.

## API references

* [Eval Tables](https://docs.coreweave.com/products/wandb/evaltables/create-an-evaluation-table)
* [Manual Weave Calls](https://docs.coreweave.com/products/wandb/weave/guides/tracking/create-call)
* [GPU Sandboxes](https://docs.coreweave.com/products/sandboxes/gpu-sandboxes)
* [Placement/spillover](https://docs.coreweave.com/products/sandboxes/placement)
* [Filesystem snapshots](https://docs.coreweave.com/products/sandboxes/file-system-snapshots)
* [Sandbox SDK](https://docs.coreweave.com/products/sandboxes/client/ref/core/sandbox)