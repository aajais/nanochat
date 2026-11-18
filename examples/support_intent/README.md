Support-intent evaluation (research demo)
========================================

This opt-in harness evaluates Nanochat checkpoints or a CPU lexical baseline on
the versioned CLINC150-derived five-way proxy dataset. It does not train a model,
change the training loop, enqueue Launch jobs, or promote a checkpoint.

From the repository root, in a separate evaluation environment:

```bash
python -m pip install -r examples/support_intent/requirements.txt
PYTHONPATH=. python examples/support_intent/run_eval.py --backend lexical_nb --split dev
PYTHONPATH=. python examples/support_intent/run_eval.py --backend nanochat --split dev \
  --checkpoint wandb/nanochat-demo/REPLACE_WITH_CHECKPOINT_COLLECTION:v0
```

The checkpoint argument is an explicit example: replace the collection with the
actual pinned model artifact. Authenticate to W&B before running. This demo
intentionally records evaluations in `wandb/nanochat-demo`; review that destination
before using it elsewhere. Artifact access requires appropriate project access.
The Nanochat backend requires one CUDA GPU and paired checkpoint metadata.

Defaults pin dataset `wandb/nanochat-demo/clinc150-support-fiveway:v0` and tokenizer
`wandb/nanochat-demo/pr1-smoke-tokenizer:v0`. Checkpoint/tokenizer checksums, context
length and state-dict compatibility are checked; prompts are never truncated.
Public test splits require the explicit `--allow-public-test` flag and must not
be used for model selection. `label_likelihood` is a separate closed-set
diagnostic, not evidence of greedy-generation format compliance or calibration.

Weave records a published Model, Dataset, Evaluation, prediction calls and scorer
calls around real inference. Expected labels are scorer-only, not model inputs.
The Model stores checkpoint identity, not weights. This public-data evaluation
logs request text, prompts and model outputs: do not substitute private data
without reviewing logging and retention. If tracing initialization fails,
inference can continue but the result remains unverified for promotion. Successful
SDK initialization/flush alone is not backend verification; independently verify
stored root/row/model/scorer calls and artifact metrics before using a result.
This PR does not install an automated promotion gate or connect it to training.

Metrics include accuracy, pooled macro-F1, per-class precision/recall, invalid
output rate, unknown recall, supported-example false abstention, and serial model
latency. Serial latency is not a serving benchmark. This narrow proxy and public
data are not production qualification; contamination and broader coverage remain
limitations. See DATASET_CARD.md for the dataset provenance and audit.

Local tests (no model download, training, or GPU required):

```bash
PYTHONPATH=examples/support_intent:. python -m unittest discover \
  -s examples/support_intent -p 'test_*.py' -v
```

The pure adapters and scoring tests use controlled/fake inference. A real CPU
evaluation was independently checked against backend trace delivery; this is not
an end-to-end GPU validation of the structured Nanochat evaluation path.