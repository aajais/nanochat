# Conversation SFT source review

## Stacked review order

1. Resumable SFT/runtime: assistant-only offline input preparation, atomic
   complete checkpoints, optimizer/loader/RNG restore, deadline handling,
   native distributed optimizer verification, and supporting captured training
   instrumentation changes.
2. Conversation evaluation: deterministic decoding/scoring, checkpoint identity,
   frozen split authorization and acceptance ledger, synthetic tests, public
   development fixtures/protocol. Final acceptance contents remain private.
3. Consolidated workflow: bounded allocation and measured token plan, distributed
   readiness, development selection/acceptance/package stages, and historical
   preparation/submission/metadata scripts under `research/nanochat_sft`.

The source fork starts at commit
`58d716e6255a5373f917ea9ca5e933d030db1779`.
The workflow is specific to the 1.384B checkpoint, exact tokenizer, 2,048-token
context, BF16 and eight NVIDIA L40s. It intentionally refuses hardware fallback.
It does not replace the base model or claim conversational quality from ARC.

## Validation

From the fully stacked checkout, with the pinned environment available:

```sh
export PYTHONPATH=.
export NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1
python test_workflow.py
python test_eight_rank_contract.py
python -m unittest discover -s tests -p 'test_sft_preparation.py'
python -m unittest discover -s tests -p 'test_conversation_eval_*.py'
```

These are CPU software checks, not successful eight-GPU readiness, throughput,
model-quality, or acceptance evidence. Private-fixture/tokenizer tests skip
explicitly in a public checkout; authenticated immutable artifacts are required
to run those tests. Never publish the held-out fixtures to make skips disappear.
The old fixture-authoring banks are intentionally not included.

## Export versus running job

This is a reviewable source export, not a replacement for the already-published
Launch job. The public workflow manifest rehashes the source files actually
included in this export; its hash differs from the running artifact. Approval
templates are not approvals. Future execution requires fresh publication,
explicit authorization, capacity, actual readiness and the original hard budget
rules. These PRs neither resubmit nor mutate the current job.

Missing data downloads, uncertain licensing, incomplete shard restores and
worker failures must remain visible failures. Dataset notices and the release
hold must be reviewed before any redistribution; these PRs confer no rights in
the omitted model or data assets.