"""Explicit, opt-in artifacts; never crawl datasets or upload credentials by default."""

import json
import os
from pathlib import Path
import re

import wandb


def track_inputs(run):
    """Verify staged files before declaring them as run inputs; never overwrite files."""
    manifest_path = os.getenv("NANOCHAT_INPUT_MANIFEST")
    if not manifest_path:
        return
    entries = json.loads(Path(manifest_path).read_text())
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ValueError("Input manifest must list 1..32 artifact/root pairs")
    verified = []
    api = wandb.Api()
    for entry in entries:
        ref, root = entry["artifact"], Path(entry["root"]).expanduser().resolve(strict=True)
        if not re.fullmatch(r"[^/:]+/[^/:]+/[^/:]+:v\d+", ref) or not root.is_dir():
            raise ValueError("Inputs require entity/project/name:vN and an existing directory")
        artifact = api.artifact(ref)
        if any(item.ref for item in artifact.manifest.entries.values()):
            raise ValueError("External references cannot be checksum-verified; use uploaded files")
        artifact.verify(root=str(root))
        # Verify before recording the consuming edge; a failed check aborts the run.
        run.use_artifact(artifact)
        verified.append({"artifact": ref, "digest": artifact.digest, "root": str(root)})
    run.config.update({"artifact_inputs_verified_rank0": verified})


def log_source(run):
    if os.getenv("NANOCHAT_LOG_CODE") != "1":
        return
    root = Path(__file__).resolve().parents[1]
    artifact = wandb.Artifact(f"{run.id}-source", type="code")
    paths = [p for folder in ("nanochat", "scripts", "tasks", "tests")
             for p in (root / folder).glob("*.py")]
    paths += [root / p for p in ("pyproject.toml", "uv.lock", "requirements-observability.txt", "dev/Dockerfile.rollout")]
    for path in sorted(paths):
        if path.is_file() and not path.is_symlink():
            artifact.add_file(str(path), name=path.relative_to(root).as_posix())
    run.log_artifact(artifact)


def log_evaluation(run, task, step, columns, rows, settings, elapsed):
    if os.getenv("NANOCHAT_LOG_EVAL_ARTIFACTS") != "1":
        return
    # Same bounded/redacted rows as the UI table, not an additional content export.
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", task)
    artifact = wandb.Artifact(f"{run.id}-eval-{slug}", type="evaluation",
        metadata={"task": task, "step": step, "settings": settings,
                  "wall_seconds": elapsed, "sampled_rows_rank0": len(rows),
                  "inputs": run.config.get("artifact_inputs_verified_rank0", [])})
    artifact.add(wandb.Table(columns=columns, data=rows), "examples")
    run.log_artifact(artifact, aliases=["latest", f"step-{step}"])