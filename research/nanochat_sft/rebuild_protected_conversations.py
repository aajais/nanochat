"""Rebuild offline cache against frozen ARC/conversation prompts; no training."""
import json,sys,time
from pathlib import Path

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'prepared'))
from scripts.prepare_conversation_sft import prepare
from nanochat.sft_runtime import sha256

folder=root/'conversation_evals_v1'
hashes=json.loads((folder/'SHA256.json').read_text())
for name,digest in hashes.items():assert sha256(folder/name)==digest
protected=[root/'arc_inputs/benchmark-inputs.json']+[folder/(s+'.json') for s in ('development','acceptance_primary','acceptance_correction')]
output=root/'conversation_cache_protected_v1'
started=time.monotonic()
manifest=prepare([root/'conversation_candidates/selected.jsonl'],output,root/'tokenizer',protected,
 progress=lambda r:print(json.dumps({'elapsed_s':round(time.monotonic()-started),**r}),flush=True))
old=json.loads((root/'conversation_cache/manifest.json').read_text())
for key in ('source_cache','selection','selection_seed','scope'):manifest[key]=old[key]
manifest['evaluation_fixture_status']='frozen-not-evaluated; shared task templates; human content review pending'
manifest['release_gate']={'training_approved':False,'completed':['ARC100 and all three conversational suite prompt exclusions'],
 'pending':['independent_content_and_license_review','human_evaluation_fixture_review',
            'protected_fingerprint_comparison_to_untouched_pretraining_holdout_without_opening_payloads']}
manifest['rebuild_comparison']={s:{'previous_sha256':old['files'][s+'.jsonl'],
 'current_sha256':manifest['files'][s+'.jsonl'],
 'byte_identical':old['files'][s+'.jsonl']==manifest['files'][s+'.jsonl']} for s in ('train','dev')}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
(output/'README.md').write_text('''# Protected provisional conversation cache — NOT TRAINING APPROVED

Rebuilt from the same pinned, cached sources against ARC100 and frozen conversational
development/primary/correction evaluation prompts. See manifest for exclusions and hashes.
No final acceptance generations have been produced. Existing untouched holdout payloads
were not opened. Matching uses exact normalization plus heuristic SimHash, not semantic proof.
The conversational fixtures share task templates and require human content review.
Independent source content/license review and protected pretraining-holdout fingerprint
comparison remain pending. The trainer MUST reject training_approved=false.
''')
print(json.dumps({'statistics':manifest['statistics'],'filter_counts':manifest['filter_counts'],
 'rebuild_comparison':manifest['rebuild_comparison'],'training_approved':False},indent=2),flush=True)