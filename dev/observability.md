# Forge observability and remote rollouts

Opt-in integrations; `--run dummy` and local rollouts remain the defaults.
The full integration stack requires **Python 3.11+** (`cwsandbox`'s minimum).
The repo defaults to 3.11; baseline training still supports 3.10. Use a separate
environment so an existing Python 3.10 environment is not replaced:

```bash
uv venv --python 3.11 .venv-forge
source .venv-forge/bin/activate
uv sync --active --extra gpu --group dev
uv pip install -r requirements-observability.txt
export WANDB_ENTITY=your-team WANDB_PROJECT=nanochat
export NANOCHAT_WEAVE=1
python -m scripts.chat_eval --source sft --task-name GSM8K --max-problems 8 --run eval-smoke
```

Use your runtime secret mechanism for `WANDB_API_KEY` and `CWSANDBOX_API_KEY`;
never put credentials in CLI arguments, images, snapshots, or source control.
The optional requirements are separate to preserve the baseline `uv.lock`.
Rerunning `uv sync --active` can remove them; reinstall the requirements afterward.

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
  IDs support comparisons across runs and repeated evaluation steps. Chat eval
  uses the loaded checkpoint's metadata step for metrics, tables, artifacts and
  traces, including when `--step` is omitted (latest checkpoint).
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
  existing save points. `NANOCHAT_LOG_CODE=1` snapshots allowlisted source and
  dependency manifests. `NANOCHAT_LOG_EVAL_ARTIFACTS=1` versions the same bounded,
  privacy-filtered evaluation rows as durable evaluation artifacts, with task,
  step, settings, timing, and verified input provenance. Input versions can be
  checksum-verified and recorded as consuming edges (see the demo below).
  Upload/storage costs apply. Checkpoints exclude optimizer shards/tokenizer;
  publish the tokenizer separately. This isn't a full distributed resume bundle.
  Promote reviewed versions to your Registry through the usual artifact UI;
  no automatic production promotion or extra project/view is created.

W&B's normal system monitor provides CPU/GPU utilization where supported.
Gradient and timing metrics explicitly labeled `rank0` are not global maxima.

## Artifact lineage demo: approved assets → run → model/evaluation outputs

The publisher uploads **only explicitly listed files**, never recursively scans
your cache, and waits for upload completion before printing its versioned path.
Use a dedicated demo project via `WANDB_PROJECT` and group the runs with
`WANDB_RUN_GROUP=observability-demo`. Do not enable content capture for private
datasets just to populate the demo.

```bash
export NANOCHAT_LOG_CODE=1 NANOCHAT_LOG_CHECKPOINTS=1 NANOCHAT_LOG_EVAL_ARTIFACTS=1
export WANDB_RUN_GROUP=observability-demo
python -m scripts.artifact_assets publish --name demo-tokenizer --type tokenizer \
  --file "tokenizer.pkl=$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" \
  --file "token_bytes.pt=$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt"
```

Set `NANOCHAT_BASE_DIR` to your real cache directory before this command. Use
the same publisher for approved pretraining shards (`--type dataset`, repeat
`--file` for **every** shard used, including the held-out last shard), or an
existing checkpoint (`--type model`, explicitly include its model and metadata
files). Dataset uploads can be large and may be subject to licensing/privacy
restrictions. Task loaders backed by Hugging Face are **not** automatically
captured or pinned by this integration: do not label their data as versioned
unless you have actually staged and configured those inputs.

For reuse on a fresh runner, stage the printed immutable `entity/project/name:vN`
into an **absent or empty** directory. This command creates no W&B run, does not
overwrite existing data, and verifies checksums after download:

```bash
python -m scripts.artifact_assets stage \
  --artifact "$TOKENIZER_ARTIFACT_VERSION" --root "$NANOCHAT_BASE_DIR/tokenizer"
```

Before running training/eval, create a JSON manifest matching the directories
the normal loaders actually read. Example structure (substitute the printed
version and your absolute path):

```json
[
  {"artifact": "your-team/nanochat/demo-tokenizer:v0", "root": "/actual/nanochat-cache/tokenizer"}
]
```

Set `NANOCHAT_INPUT_MANIFEST=/absolute/path/to/inputs.json`. Each enabled rank-zero
run checks the **entire** listed directory against the immutable artifact before
calling `run.use_artifact()`. Mutable aliases and unverifiable external references
are rejected, as are missing, modified or extra files. No download/overwrite
happens inside training. This records verified staged inputs, not proof that
every file was used by the model: the caller must point the real loaders at those
roots. Multi-node runs must stage identical inputs on every node; the verification
here is rank-zero only. Do not modify those directories while a run is active.

Run the normal training or evaluation command with an online `--run` name; no
synthetic training metrics are generated. In Forge, inspect:

1. **Run config / system charts:** resolved settings and hardware metrics.
2. **Workspace:** loss/BPB/reward, gradient/memory/throughput curves and Eval Tables.
3. **Artifacts graph:** tokenizer/dataset/checkpoint inputs → run → code/model/eval
   outputs. Checkpoint/evaluation metadata repeats verified input versions/digests.
4. **Weave Calls:** sampled eval children correlated with the producing run ID.
5. **Sandboxes:** actual GPU workers, timings and reusable environment snapshot ID.
6. **Registry:** optionally link a reviewed model version to an existing collection
   via the UI. `latest` is not a quality guarantee; no automated approval or
   production alias change is performed.

Native workspace panels and reports can present these same run metrics without
logging duplicate plots as images. Alerts/automations, a persistent report, and
Registry governance need a chosen team/project and approval criteria; they are
not silently created by this instrumentation. Inference/Post-Training/Agent Lens
aren't added: this repo trains its own model and logs eval Calls, not hosted-agent
traffic. This is a coherent demonstration, not a claim that every Forge product
or every possible metric is applicable.

## Warm GPU rollout pool (GSM8K RL only)

Worker startup uses the SDK's positional command API:
`Sandbox.run("python", "-u", "-m", "scripts.rollout_worker", **options)`;
`command=`/`args=` are not supported by `Sandbox.run`.
See [Sandbox lifecycle](https://docs.coreweave.com/products/sandboxes/client/guides/sandbox-lifecycle#sandbox-run--immediate-start).

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
* [Create artifacts](https://docs.coreweave.com/products/wandb/artifacts/construct-an-artifact)
* [Input lineage and checksum verification](https://docs.coreweave.com/products/wandb/artifacts/download-and-use-an-artifact)
* [Artifact tables](https://docs.coreweave.com/products/wandb/track/log/log-tables#artifact-tables)
* [Manual Weave Calls](https://docs.coreweave.com/products/wandb/weave/guides/tracking/create-call)
* [GPU Sandboxes](https://docs.coreweave.com/products/sandboxes/gpu-sandboxes)
* [Placement/spillover](https://docs.coreweave.com/products/sandboxes/placement)
* [Filesystem snapshots](https://docs.coreweave.com/products/sandboxes/file-system-snapshots)
* [Sandbox SDK](https://docs.coreweave.com/products/sandboxes/client/ref/core/sandbox)