"""Bounded, rank-zero telemetry. No model/optimizer tensors are sent to Weave."""

import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import os
from pathlib import Path
import time
import warnings

import wandb

_client = None
_evaluation = ContextVar("nanochat_evaluation", default=None)
_parent = ContextVar("nanochat_trace_parent", default=None)
_suppressed = ContextVar("nanochat_trace_suppressed", default=False)


def init_run(project, name, config, enabled=True):
    from nanochat.common import DummyWandb
    if not enabled:
        return DummyWandb()
    run = wandb.init(project=os.getenv("WANDB_PROJECT", project), name=name, config=config)
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    for key in ("train/loss", "val/bpb"):
        run.define_metric(key, summary="min")
    for key in ("core_metric", "chatcore_metric", "chatcore_cat", "reward", "pass@*"):
        run.define_metric(key, summary="max")
    if os.getenv("NANOCHAT_WEAVE") == "1":
        if run.settings.mode != "online":
            warnings.warn("Weave tracing skipped: W&B run is not online")
        else:
            import weave
            global _client
            _client = weave.init(f"{run.entity}/{run.project}")
            atexit.register(weave.finish)
    return run


def active_run():
    return wandb.run if int(os.getenv("RANK", "0")) == 0 else None


def log_model_config(model, args, **resolved):
    run = active_run()
    if run is not None:
        import torch
        from nanochat.common import COMPUTE_DTYPE
        run.config.update({
            "resolved": {**vars(args), **resolved},
            "model": asdict(model.config),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "runtime": {"torch": torch.__version__, "dtype": str(COMPUTE_DTYPE),
                        "device": str(model.get_device()),
                        "world_size": int(os.getenv("WORLD_SIZE", "1")),
                        "execution_backend": os.getenv("NANOCHAT_EXECUTION_BACKEND", "local")},
        }, allow_val_change=True)


def optimizer_metrics(model, optimizer):
    """Call at existing log cadence, after unscale and before zero_grad. Rank-local."""
    if active_run() is None:
        return {}
    import torch
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    metrics = {f"optim/lr_group_{i}": float(g["lr"]) for i, g in enumerate(optimizer.param_groups)}
    if grads:
        norms = torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])
        metrics["train/grad_norm_rank0"] = torch.linalg.vector_norm(norms).item()
        metrics["train/grad_nonfinite_rank0"] = int((~torch.isfinite(norms)).sum().item())
    device = model.get_device()
    if device.type == "cuda":
        metrics.update({"memory/allocated_bytes": torch.cuda.memory_allocated(device),
                        "memory/reserved_bytes": torch.cuda.memory_reserved(device),
                        "memory/peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    return metrics


def log_checkpoint(checkpoint_dir, step):
    """Opt-in model + metadata versioning; intentionally excludes optimizer shards."""
    run = active_run()
    if run is None or os.getenv("NANOCHAT_LOG_CHECKPOINTS") != "1":
        return
    artifact = wandb.Artifact(f"{run.id}-checkpoint", type="model",
                              metadata={"step": step, "includes_optimizer": False})
    for filename in (f"model_{step:06d}.pt", f"meta_{step:06d}.json"):
        artifact.add_file(str(Path(checkpoint_dir) / filename))
    run.log_artifact(artifact, aliases=["latest", f"step-{step}"])


@contextmanager
def trace_span(name, inputs, enabled=True):
    output = {}
    if _client is None or not enabled or _suppressed.get():
        token = _suppressed.set(True)
        try:
            yield output
        finally:
            _suppressed.reset(token)
        return
    run = active_run()
    call = _client.create_call(name, inputs=inputs, parent=_parent.get(), use_stack=False,
                               attributes={"wandb_run_id": run.id} if run else {})
    token = _parent.set(call)
    try:
        yield output
    except BaseException as exc:
        _client.finish_call(call, exception=exc)
        raise
    else:
        _client.finish_call(call, output=output)
    finally:
        _parent.reset(token)


class Evaluation:
    def __init__(self, task, step, settings):
        self.task, self.step, self.settings = task, step, settings
        self.rows = []
        self.row_limit = max(0, int(os.getenv("NANOCHAT_EVAL_ROWS", "64")))
        self.trace_limit = max(0, int(os.getenv("NANOCHAT_TRACE_EXAMPLES", "8")))
        self.content = os.getenv("NANOCHAT_LOG_CONTENT") == "1"

    def trace_example(self, problem_id):
        # Rank zero owns indices 0, world_size, 2*world_size, ...
        return problem_id < self.trace_limit * int(os.getenv("WORLD_SIZE", "1"))

    def record(self, problem_id, sample_id, conversation, completion, correct,
               prompt_tokens, completion_tokens, latency_s):
        prompt = conversation["messages"][:-1]
        reference = conversation["messages"][-1]["content"]
        self.record_row(problem_id, sample_id, prompt, reference, completion, correct,
                        prompt_tokens, completion_tokens, latency_s)

    def record_row(self, problem_id, sample_id, prompt, reference, completion, correct,
                   prompt_tokens=None, completion_tokens=None, latency_s=None):
        if active_run() is None or len(self.rows) >= self.row_limit:
            return
        self.rows.append([
            f"{self.task}/{problem_id}/{sample_id}",
            prompt if self.content else None, reference if self.content else None,
            completion if self.content else None, bool(correct), prompt_tokens,
            completion_tokens, latency_s,
        ])

    def log(self, elapsed):
        run = active_run()
        if run is None:
            return
        data = {"step": self.step, f"eval/{self.task}/wall_seconds": elapsed,
                f"eval/{self.task}/logged_rows_rank0": len(self.rows)}
        if self.rows:
            columns = ["example_id", "prompt", "reference", "completion", "correct",
                       "prompt_tokens", "completion_tokens", "generation_seconds"]
            if hasattr(wandb, "EvalTable"):
                table = wandb.EvalTable(columns=columns, data=self.rows,
                    input_columns=["example_id", "prompt", "reference"],
                    output_columns=["completion"],
                    score_columns=["correct", "prompt_tokens", "completion_tokens", "generation_seconds"])
            else:
                warnings.warn("wandb.EvalTable unavailable; logging a standard W&B Table. Upgrade wandb for Eval Tables.")
                table = wandb.Table(columns=columns, data=self.rows)
            data[f"eval/{self.task}/examples"] = table
        run.log(data)


@contextmanager
def evaluation(task, step=0, **settings):
    report = Evaluation(task, step, settings)
    token = _evaluation.set(report)
    start = time.perf_counter()
    try:
        with trace_span("nanochat.evaluate", {"task": task, "step": step, **settings}) as output:
            yield output
        report.log(time.perf_counter() - start)
    finally:
        _evaluation.reset(token)


def generate_eval(engine, tokens, problem_id, **kwargs):
    report = _evaluation.get()
    enabled = report is not None and report.trace_example(problem_id)
    start = time.perf_counter()
    with trace_span("nanochat.generate", {"problem_id": problem_id, "prompt_tokens": len(tokens), **kwargs}, enabled) as output:
        results, masks = engine.generate_batch(tokens, **kwargs)
        elapsed = time.perf_counter() - start
        output.update({"completion_tokens": [len(r) - len(tokens) for r in results],
                       "wall_seconds": elapsed})
    return results, masks, elapsed


def forward_eval(model, prompt_ids, problem_id):
    report = _evaluation.get()
    enabled = report is not None and report.trace_example(problem_id)
    with trace_span("nanochat.categorical_forward",
                    {"problem_id": problem_id, "batch_shape": list(prompt_ids.shape)}, enabled):
        return model(prompt_ids)


def trace_core_example(problem_id):
    report = _evaluation.get()
    return trace_span("nanochat.core_example", {"problem_id": problem_id},
                      report is not None and report.trace_example(problem_id))


def record_core_result(problem_id, item, prediction, correct):
    report = _evaluation.get()
    if report is not None:
        if report.content and len(report.rows) < report.row_limit and hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        report.record_row(problem_id, 0, item, item.get("gold"), prediction, correct)


def score_eval(task, conversation, completion, problem_id, sample_id=0,
               prompt_tokens=None, completion_tokens=None, latency_s=None):
    report = _evaluation.get()
    inputs = {"problem_id": problem_id, "sample_id": sample_id}
    if report is not None and report.content:
        inputs.update({"conversation": conversation, "completion": completion})
    enabled = report is not None and report.trace_example(problem_id)
    with trace_span("nanochat.score", inputs, enabled) as output:
        correct = task.evaluate(conversation, completion)
        output["correct"] = bool(correct)
    if report is not None:
        report.record(problem_id, sample_id, conversation, completion, correct,
                      prompt_tokens, completion_tokens, latency_s)
    return correct