# CLINC150 support-category evaluation derivative

## Source, attribution, and license

Adapted from **An Evaluation Dataset for Intent Classification and Out-of-Scope
Prediction**, Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke,
Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A.
Laurenzano, Lingjia Tang, and Jason Mars, EMNLP-IJCNLP2019.

- Repository: https://github.com/clinc/oos-eval
- Paper: https://aclanthology.org/D19-1131/
- Source revision: `828f8093932c8fe6ca7936c3d2e52903b1c523de`
- Source file: `data/data_full.json`; exact SHA256 is in `manifest.json`.
- License: **Creative Commons Attribution3.0 Unported (CC BY3.0)**;
  https://creativecommons.org/licenses/by/3.0/ . The original license and README
  are included. Texts are unchanged. Modifications: selected intents, mapped
  coarse labels, deterministic IDs, split naming, duplicate exclusion and audit.

The original English utterances were crowdsourced from paraphrases/scenarios;
they are not actual customer conversations or newly model-generated examples.

## Task contract and coverage limits

Return exactly one of the following strings (leading/trailing whitespace is
ignored; explanations, quotes, alternate casing, multiple labels are invalid):

| Label | Original source intents |
|---|---|
| billing | bill_balance, bill_due, pay_bill |
| account_access | account_blocked, change_user_name |
| technical_issue | reset_settings, sync_device |
| cancellation | cancel_reservation |
| unknown | weather, tell_joke, recipe, translate, directions, restaurant_suggestion, play_music, calendar, alarm, shopping_list, nutrition_info, book_flight |

The challenge splits use the original `oos` utterances, all mapped to unknown.
Unlisted source intents are excluded, **not automatically relabeled unknown**.
Original source intents and row indices remain attached to each example.

This is a narrow **proxy task**, not a comprehensive support taxonomy. The
technical class covers device setup/reset rather than broad troubleshooting;
cancellation covers reservations, not subscriptions/order cancellation;
account_access includes username changes. Real support should replace/extend
these definitions before claiming business relevance. Unknown means outside
this explicitly supported taxonomy, not "the model seems uncertain".

## Splits and leakage controls

- `train`: original train rows only; for fitting the lexical baseline and any
  separately authorized future specialization. It is not pretraining data added
  to the running BPB campaign.
- `dev`: original validation rows; the default evaluation split.
- `test`: original test rows; public held-out evaluation, explicit opt-in.
- `challenge_dev` / `challenge_test`: original oos_val/oos_test respectively.
  These are **OOD-only**, not a newly authored adversarial/privacy/safety suite.
- Normalize Unicode/case/punctuation/spacing for exact duplicate checks. Remove
  conflicting coarse-label duplicates from all folds. Preserve test rows first,
  then remove matching lower-priority rows without moving examples across folds.
- Exclude lower-priority rows with char3..5-gram TF-IDF cosine similarity >=0.95
  to a protected higher-priority fold. This is a conservative lexical screen,
  **not proof of semantic or seed-family isolation**. Details are in `audit.json`.
- There are no source customer/conversation IDs; cannot claim a customer-group
  split. Public pretraining contamination is unknown.
- Test data must not be used for prompt/threshold/model selection. Because this
  benchmark is public and exposure is possible, it is **not a private locked
  production release test**. Build an independently labeled real-traffic test
  before production qualification.

Actual post-audit row and class counts, removed IDs/reasons, and checksums are in
`manifest.json` and `audit.json`. Class imbalance is preserved and documented;
macro-F1 and per-class metrics are more informative than raw accuracy alone.

## Scoring

Report five-class macro-F1 only on folds covering all five classes, per-class
precision/recall/F1/support, full confusion matrix including invalid outputs,
accuracy, unknown recall, supported false-abstention rate, and invalid rate.
OOD-only splits report unknown recall, not misleading five-class macro-F1.
These aggregate metrics have no uncertainty bounds in v0; release decisions
need paired uncertainty analysis and task-specific critical-slice gates.

Two Nanochat decision methods are distinct protocols:
- `greedy_label`: unforced deterministic generation, strict output parsing;
  measures both task accuracy and instruction/output-contract compliance.
- `label_likelihood`: rank the mean NLL of fixed label continuations, including
  a space prefix and newline suffix. This is a length-normalized closed-set
  diagnostic, **not calibrated confidence or a format-validity test**. Always
  choosing a valid label does not demonstrate instruction following.

Serial model-call latency is diagnostic only; it excludes service/network load
and is not a concurrency/p95 serving qualification. No production thresholds
are automatically approved or declared passed by these jobs.