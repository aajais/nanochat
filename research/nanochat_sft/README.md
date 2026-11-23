# Nanochat SFT research and operations history

These scripts capture the CPU preparation, remediation, publication, readiness,
submission and metadata work behind the November 2026 conversation-SFT effort.
They are historical, project-specific tools, not an auto-executing pipeline.

**Do not batch-run this directory.** Some scripts publish artifacts, update run
metadata or submit GPU jobs. Review each script, resolve its immutable inputs,
and obtain fresh approval before any mutation or compute spend. Older HOLD
decisions and submission paths are retained for audit, not current instructions.

Scripts originally ran under `/workspace/nanochat_sft` with sibling `prepared`,
source snapshots, local data caches and downloaded artifacts. Most still expect
that working layout; this directory preserves their source, not those payloads.
Use the root `workflow.py` and its explicit requirements for the consolidated
runtime. Never interpret the presence of a script as proof it ran successfully.

Omitted: datasets, tokenizer/checkpoint files, generated reports/logs, live
approval records, final acceptance fixtures, and the authoring banks that reveal
those fixtures. Retrieve authorized immutable inputs through W&B rather than
publishing protected acceptance content here.

The active job's published workflow hash is
`11ddd235677a159f08c9d4997d2fcad23f6ef237dc52c30cad9bb56525b08799`.
This GitHub source export is **not** that immutable artifact: tests have portable
paths, private inputs are omitted, and operations scripts are relocated. It must
be rebuilt, repinned and authorized before any future submission. These PRs do
not change or restart the running job.