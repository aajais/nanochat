"""Trusted, warm GPU generation worker. No public listener or runtime credentials.

The controller transfers a state_dict, tokenizer and JSON jobs through the
authenticated Sandboxes file API. --submit is a bounded local file-IPC client.
"""

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path("/workspace/nanochat-rollout")
ASSETS = Path("/workspace/nanochat-assets")


def submit(request_id, timeout):
    os.replace(ROOT / "request.tmp", ROOT / "request.json")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = json.loads((ROOT / "response.json").read_text())
        except FileNotFoundError:
            response = {}
        if response.get("request_id") == request_id:
            return
        time.sleep(0.1)
    raise TimeoutError("Rollout request timed out; trainer must stop the worker")


def generate_jobs(engine, jobs):
    results = []
    for job in jobs:
        sequences, masks = [], []
        started = time.perf_counter()
        for seed in job["seeds"]:
            count = min(job["batch_size"], job["num_samples"] - len(sequences))
            if count <= 0:
                break
            seq, mask = engine.generate_batch(
                job["tokens"], num_samples=count, max_tokens=job["max_tokens"],
                temperature=job["temperature"], top_k=job["top_k"], seed=seed)
            sequences.extend(seq)
            masks.extend(mask)
        results.append({"job_id": job["job_id"], "sequences": sequences, "masks": masks,
                        "generation_seconds": time.perf_counter() - started})
    return results


def serve():
    import torch
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.flash_attention import USE_FA3
    import nanochat.engine as engine_module
    from nanochat.calculator import safe_calculator

    # Never evaluate model-generated Python in a CKS GPU container.
    engine_module.use_calculator = safe_calculator
    if not torch.cuda.is_available():
        raise RuntimeError("GPU rollout worker requires a visible CUDA GPU")
    torch.cuda.set_device(0)
    ROOT.mkdir(parents=True, exist_ok=True)
    model = engine = tokenizer = None
    version = seen = None
    model_config = None
    while True:
        try:
            request = json.loads((ROOT / "request.json").read_text())
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if request["request_id"] == seen:
            time.sleep(0.05)
            continue
        seen = request["request_id"]
        response = {"request_id": seen, "version": request["version"],
                    "worker": {"gpu": torch.cuda.get_device_name(0),
                               "torch_version": torch.__version__, "dtype": str(COMPUTE_DTYPE),
                               "flash_attention_3": USE_FA3}}
        try:
            if model_config is not None and request["model_config"] != model_config:
                raise ValueError("Model architecture changed inside a warm rollout worker")
            if model is None:
                model_config = request["model_config"]
                config = GPTConfig(**model_config)
                with torch.device("meta"):
                    model = GPT(config)
                model.to_empty(device="cuda")
                model.init_weights()
                tokenizer = RustBPETokenizer.from_directory(ASSETS / "tokenizer")
                if tokenizer.get_vocab_size() != config.vocab_size:
                    raise ValueError("Tokenizer and model vocabularies differ")
                engine = engine_module.Engine(model, tokenizer)
            if version != request["version"]:
                state = torch.load(ROOT / "policy.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                del state
                model.eval()
                version = request["version"]
            with torch.inference_mode():
                response["results"] = generate_jobs(engine, request["jobs"])
        except Exception as exc:
            response["error"] = f"{type(exc).__name__}: {exc}"
        temporary = ROOT / "response.tmp"
        temporary.write_text(json.dumps(response))
        os.replace(temporary, ROOT / "response.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.submit:
        submit(args.submit, args.timeout)
    else:
        serve()