"""Publish explicit approved files, or stage a pinned version without overwriting data."""

import argparse
import os
from pathlib import Path, PurePosixPath
import re

import wandb

from nanochat.observability import init_run


def publish(run, name, kind, files):
    artifact = wandb.Artifact(name, type=kind, metadata={"explicit_files_only": True})
    seen = set()
    for specification in files:
        target, source = specification.split("=", 1)
        target_path, source_path = PurePosixPath(target), Path(source).expanduser()
        if (not target or target_path.is_absolute() or ".." in target_path.parts
                or "\\" in target or target in seen or not source_path.is_file()):
            raise ValueError("Files require unique relative artifact names and existing files")
        seen.add(target)
        artifact.add_file(str(source_path), name=target)
    if not seen:
        raise ValueError("At least one explicitly approved file is required")
    logged = run.log_artifact(artifact)
    logged.wait()
    return f"{run.entity}/{run.project}/{logged.name}"


def stage(ref, root):
    if not re.fullmatch(r"[^/:]+/[^/:]+/[^/:]+:v\d+", ref):
        raise ValueError("Stage requires an immutable entity/project/name:vN")
    root = Path(root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Stage destination must be absent or empty; existing files are never overwritten")
    artifact = wandb.Api().artifact(ref)
    if any(item.ref for item in artifact.manifest.entries.values()):
        raise ValueError("Only uploaded files can be checksum-verified")
    artifact.download(root=str(root))
    artifact.verify(root=str(root))
    return str(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("publish")
    upload.add_argument("--name", required=True)
    upload.add_argument("--type", required=True, choices=["dataset", "tokenizer", "model"])
    upload.add_argument("--file", action="append", required=True, help="artifact/path=local/path; repeat for each approved file")
    download = commands.add_parser("stage")
    download.add_argument("--artifact", required=True)
    download.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "stage":
        print(stage(args.artifact, args.root))
    else:
        provenance = {k: os.environ[v] for k, v in
                      (("thread_id", "WANDB_ARIA_THREAD_ID"), ("turn_id", "WANDB_ARIA_TURN_ID")) if v in os.environ}
        run = init_run("nanochat-assets", "publish-assets", {"_wb_agent": provenance})
        try:
            print(publish(run, args.name, args.type, args.file))
        finally:
            run.finish()


if __name__ == "__main__":
    main()