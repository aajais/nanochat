"""Synchronous on-policy rollout pool using warm, single-GPU CoreWeave Sandboxes.

Credentials stay on the trainer. Workers receive only trusted weights/tokenizer,
token IDs, and generation settings over the authenticated Sandboxes file API.
"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import hashlib
import os
from pathlib import Path
import tempfile
import time
import uuid
import warnings

from nanochat.observability import active_run, trace_span

ROOT = "/workspace/nanochat-rollout"
ASSETS = "/workspace/nanochat-assets"


def compute_dtype_name():
    from nanochat.common import COMPUTE_DTYPE
    return str(COMPUTE_DTYPE).removeprefix("torch.")


@dataclass(frozen=True)
class RolloutConfig:
    image: str
    workers: int = 1
    placement: str = "cks_then_serverless"
    cpu: str = "4"
    memory: str = "16Gi"
    lifetime_seconds: int = 3600
    startup_seconds: int = 600
    request_seconds: int = 300
    snapshot_id: str = ""
    save_snapshot: bool = False
    snapshot_size: str = "20Gi"
    runner_ids: tuple = ()

    def __post_init__(self):
        if not self.image or "@sha256:" not in self.image:
            raise ValueError("rollout image must be pinned by @sha256 digest")
        if not 1 <= self.workers <= 32:
            raise ValueError("rollout workers must be between 1 and 32 (per training rank)")
        if self.placement not in ("cks", "serverless", "cks_then_serverless"):
            raise ValueError("unsupported rollout placement")
        if self.runner_ids and self.placement != "cks":
            raise ValueError("runner IDs require strict CKS placement (no spillover)")
        if min(self.startup_seconds, self.request_seconds) <= 0:
            raise ValueError("rollout timeouts must be positive")
        if self.lifetime_seconds <= self.startup_seconds + self.request_seconds:
            raise ValueError("rollout lifetime must exceed startup + request timeout")


def sandbox_options(config):
    from cwsandbox import AuthStrategy, NetworkOptions, ResourceOptions, FileSystemSnapshotOptions
    if config.placement != "serverless" and not os.getenv("CWSANDBOX_API_KEY"):
        raise ValueError("CKS/spillover requires CWSANDBOX_API_KEY with Sandbox User access")
    auth = AuthStrategy.COREWEAVE_API_KEY if os.getenv("CWSANDBOX_API_KEY") else AuthStrategy.WANDB
    options = dict(
        auth=auth, container_image=config.image,
        placement_mode="serverless" if config.placement == "serverless" else "cks",
        placement_spillover="cks_then_serverless" if config.placement == "cks_then_serverless" else "strict",
        resources=ResourceOptions(requests={"cpu": config.cpu, "memory": config.memory},
                                  limits={"cpu": config.cpu, "memory": config.memory},
                                  gpu={"count": 1}),
        network=NetworkOptions(deny_egress=True, deny_ingress=True),
        max_lifetime_seconds=config.lifetime_seconds,
        request_timeout_seconds=config.startup_seconds,
        environment_variables={"NANOCHAT_BASE_DIR": "/workspace/nanochat-assets",
                               "NANOCHAT_DTYPE": compute_dtype_name(),
                               "HF_HUB_OFFLINE": "1", "HF_HOME": f"{ASSETS}/cache/huggingface",
                               "TRITON_CACHE_DIR": f"{ASSETS}/cache/triton",
                               "TORCHINDUCTOR_CACHE_DIR": f"{ASSETS}/cache/torchinductor",
                               "CUDA_CACHE_PATH": f"{ASSETS}/cache/cuda"},
    )
    if config.runner_ids:
        options["runner_ids"] = list(config.runner_ids)
    if config.snapshot_id or config.save_snapshot:
        options["file_system_snapshot"] = FileSystemSnapshotOptions(
            mount_path=ASSETS, size=config.snapshot_size,
            file_system_snapshot_id=config.snapshot_id or None)
    # No runner/GPU type pin: the same 1-GPU request must fit both placements.
    return options


def validate_response(response, request):
    if response.get("request_id") != request["request_id"] or response.get("version") != request["version"]:
        raise RuntimeError("Rejected stale/mismatched rollout response")
    if response.get("error"):
        raise RuntimeError(f"Rollout worker failed: {response['error']}")
    results = response.get("results", [])
    if len(results) != len(request["jobs"]):
        raise RuntimeError("Incomplete rollout response")
    for result, job in zip(results, request["jobs"]):
        if result.get("job_id") != job["job_id"]:
            raise RuntimeError("Rollout job order/identity mismatch")
        sequences, masks = result["sequences"], result["masks"]
        if len(sequences) != job["num_samples"] or len(masks) != len(sequences):
            raise RuntimeError("Rollout sample count mismatch")
        for sequence, mask in zip(sequences, masks):
            if (sequence[:len(job["tokens"])] != job["tokens"] or len(mask) != len(sequence)
                    or len(sequence) > len(job["tokens"]) + job["max_tokens"]
                    or any(type(t) is not int or not 0 <= t < request["model_config"]["vocab_size"] for t in sequence)
                    or any(m not in (0, 1) for m in mask)
                    or any(mask[:len(job["tokens"])])):
                raise RuntimeError("Invalid rollout tokens/masks")
    return results


class RolloutPool:
    def __init__(self, config, tokenizer_path):
        from cwsandbox import Sandbox
        self.config = config
        self.workers = []
        self.version = None
        self.model_config = None
        self.snapshot_id = None
        self.worker_info = {}
        self.manifest = {"image": config.image,
                         "dtype": compute_dtype_name(),
                         "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()}
        self.executor = ThreadPoolExecutor(max_workers=config.workers)
        atexit.register(self.close)
        try:
            options = sandbox_options(config)
            def start(_):
                sandbox = Sandbox.run("python", "-u", "-m", "scripts.rollout_worker", **options)
                # Track immediately so partial startup failures still get cleaned up.
                self.workers.append(sandbox)
                sandbox.wait(timeout=config.startup_seconds)
                if config.snapshot_id:
                    manifest = json.loads(sandbox.read_file(f"{ASSETS}/manifest.json").result())
                    if manifest != self.manifest:
                        raise ValueError("Rollout snapshot image/dtype/tokenizer mismatch; rebuild the snapshot")
                sandbox.exec(["mkdir", "-p", ROOT, f"{ASSETS}/tokenizer"], check=True).result()
                self._upload(sandbox, tokenizer_path, f"{ASSETS}/tokenizer/tokenizer.pkl")
                return sandbox
            list(self.executor.map(start, range(config.workers)))
            # A replacement pool has new IDs/settings within the same W&B run.
            # Telemetry is part of construction: failures must stop workers too.
            run = active_run()
            if run is not None:
                run.config.update({"rollout_pool": asdict(config),
                                   "rollout_sandbox_ids_rank0": [w.sandbox_id for w in self.workers]},
                                  allow_val_change=True)
        except BaseException:
            self.close()
            raise

    def _upload(self, sandbox, local_path, remote_path):
        def chunks():
            with open(local_path, "rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk
        sandbox.write_file_streaming(remote_path, chunks(),
                                     timeout_seconds=self.config.request_seconds).result()

    def sync(self, model, step):
        """No optimizer step may run between sync() and generate_many()."""
        import torch
        from nanochat.common import get_base_dir
        start = time.perf_counter()
        self.version = None
        version = f"{step}-{uuid.uuid4().hex}"
        self.model_config = asdict(model.config)
        with tempfile.TemporaryDirectory(prefix="rollout-", dir=get_base_dir()) as directory:
            path = Path(directory) / "policy.pt"
            # weights_only load on workers; no optimizer state or executable objects.
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
            list(self.executor.map(lambda w: self._upload(w, path, f"{ROOT}/policy.pt"), self.workers))
        self.version = version
        return time.perf_counter() - start

    def _request(self, worker, jobs):
        request = {"request_id": uuid.uuid4().hex, "version": self.version,
                   "model_config": self.model_config, "jobs": jobs}
        start = time.perf_counter()
        with trace_span("nanochat.remote_rollout", {"version": self.version,
                        "sandbox_id": worker.sandbox_id, "jobs": len(jobs)}) as output:
            # Stage then rename: the long-running worker never sees partial JSON.
            worker.write_file(f"{ROOT}/request.tmp", json.dumps(request).encode()).result()
            worker.exec(["python", "-m", "scripts.rollout_worker", "--submit", request["request_id"],
                         "--timeout", str(self.config.request_seconds)],
                        check=True, timeout_seconds=self.config.request_seconds + 10).result()
            response = json.loads(worker.read_file(f"{ROOT}/response.json").result())
            results = validate_response(response, request)
            if "worker" in response:
                output["worker"] = response["worker"]
                self.worker_info[worker.sandbox_id] = response["worker"]
            output.update({"wall_seconds": time.perf_counter() - start,
                           "completion_tokens": sum(len(s) - len(j["tokens"])
                               for r, j in zip(results, jobs) for s in r["sequences"])})
        return results

    def generate_many(self, jobs):
        if self.version is None:
            raise RuntimeError("Synchronize the current policy before requesting rollouts")
        # At most one outstanding request per worker; no weight updates during work.
        assignments = [jobs[i::len(self.workers)] for i in range(len(self.workers))]
        futures = [self.executor.submit(self._request, worker, subset)
                   for worker, subset in zip(self.workers, assignments) if subset]
        try:
            results = [r for future in futures for r in future.result()]
            if self.config.save_snapshot and self.snapshot_id is None:
                worker = self.workers[0]
                worker.write_file(f"{ASSETS}/manifest.json", json.dumps(self.manifest).encode()).result()
                # Only immutable environment assets/caches, never IPC or stale policy weights.
                snapshot_id = worker.snapshot(wait_for_ready=True).result()
                from cwsandbox import Sandbox
                record = Sandbox.get_snapshot(snapshot_id, auth=sandbox_options(self.config)["auth"]).result()
                status = getattr(record.status, "value", record.status)
                if str(status).upper() != "READY":
                    raise RuntimeError(f"Snapshot {snapshot_id} is not READY: {status}")
                self.snapshot_id = snapshot_id
                print(f"Reusable rollout environment snapshot: {snapshot_id}", flush=True)
                run = active_run()
                if run is not None:
                    run.summary["rollout/environment_snapshot_id"] = snapshot_id
        except BaseException:
            self.close()
            raise
        by_id = {r["job_id"]: r for r in results}
        if len(by_id) != len(jobs):
            raise RuntimeError("Duplicate or missing rollout jobs")
        run = active_run()
        if run is not None:
            run.summary["rollout/worker_runtime"] = dict(self.worker_info)
        return [by_id[j["job_id"]] for j in jobs]

    def close(self):
        # Wait for requests/startups before stopping, including partial construction.
        self.executor.shutdown(wait=True)
        remaining = []
        for worker in self.workers:
            try:
                worker.stop(missing_ok=True).result()
            except Exception:
                warnings.warn(f"Could not stop rollout sandbox {worker.sandbox_id}; stop it manually")
                remaining.append(worker)
        self.workers = remaining