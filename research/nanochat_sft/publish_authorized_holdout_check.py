"""Publish only authorized hashes and read-only comparison evidence; no GPU work."""
import hashlib,json,os
from pathlib import Path
import wandb
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'authorized_holdout_check_v1'
SCOPE='wandb/nanochat-demo'
RECEIPT=ROOT/'published_authorized_holdout_check.json'
if RECEIPT.exists():raise FileExistsError('Inspect existing publication receipt; do not duplicate audit run')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
binding=json.loads((OUT/'candidate-binding-evidence.json').read_text())
comparison=json.loads((OUT/'comparison.json').read_text())
crosscheck=json.loads((OUT/'independent-crosscheck.json').read_text())
index=json.loads((OUT/'fingerprints.json').read_text())
assert binding['retained_conversations_verified']==comparison['candidates_checked']==17018
assert comparison['lexical_flags']==[] and crosscheck['agrees_with_exported_blake2b_index_result'] is True
assert comparison['index_sha256']==sha(OUT/'fingerprints.json')
assert comparison['candidate_sha256']==sha(OUT/'retained-candidate-verified-texts.jsonl')
assert binding['training_approved'] is False
api=wandb.Api()
candidate=api.artifact(binding['candidate_ref'])
assert candidate.digest=='3f356c1f3cb87a99d5c95dce19dc1d66'
assert candidate.metadata['training_approved'] is False
README='''# Authorized holdout fingerprint comparison — no training approval

Outcome: all 17,018 retained SFT conversations (16,184 train / 834 dev)
have zero normalized exact-document or shared contiguous 13-word matches with
all 2,000 documents in nanochat-untouched-holdout-20261117-v1:v0. This is the
existing pretraining/BPB holdout, NOT a final conversational acceptance test.
The exported index contains 2,000 document hashes and 963,615 unique 13-gram hashes.
Each candidate message and the joined conversation were compared. An independent
implementation from the original corpus using SHA256 instead of BLAKE2b also
found zero matches. Candidate text was re-rendered with the pinned tokenizer and
exactly matched each retained cache row's token IDs and assistant loss masks.

User authorized this automated export/comparison. Protected Parquet was read
programmatically in the sandbox; no holdout text was shown, sent to a model,
or added to these artifacts. Fingerprints are not a formal privacy guarantee.
No independent custodian approval or human review is claimed.

Integrity: Parquet bytes match the immutable artifact manifest's base64 MD5.
The original producer's holdout_sha256 is an ORDERED CORPUS hash, not a Parquet
file hash: SHA256 over each document's UTF-8 bytes preceded by its 8-byte
big-endian byte length. This was independently recomputed and matched. The
initial assertion that confused file and corpus hashes stopped before comparison;
it did not signal corpus mutation. Original producer source identity is recorded.

Limitations: normalization is NFKC + casefold + word extraction. This does not
prove semantic deduplication, detect all paraphrases or guarantee absence of
shorter embedded overlaps. The historical readiness artifact predates this
authorized export; its 'payload unopened/export pending' statements are now
superseded only by this evidence. Existing dataset bytes/manifest remain unchanged.

Status: this specific automated lexical comparison is complete. Independent
human dataset/fixture content review and source attribution/license acceptance
remain pending. training_approved=false; launch_ready=false. No GPU retry,
training update, model call or acceptance generation was performed.

Next action: review FILTERED_DATASET_REVIEW.md in nanochat-sft-readiness-review:v0
and the frozen v3 conversational fixtures. Record the human decision separately;
automated export is not approval of that packet or authorization to launch.
'''
(OUT/'README.md').write_text(README)
provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',name='authorized-holdout-fingerprint-comparison-no-gpu',
    job_type='sft-holdout-contamination-audit',dir=str(ROOT),save_code=False,
    config={'_wb_agent':provenance,'scope':'user-authorized automated holdout fingerprint export/comparison; not independent human review',
            'candidate_ref':binding['candidate_ref'],'training_approved':False,'launch_ready':False,
            'gpu_jobs_launched':0,'model_calls':0,'acceptance_generations':0})
receipt={'run_id':run.id,'run_url':run.url,'artifacts':{},'gpu_jobs_launched':0,'training_approved':False,'launch_ready':False}
RECEIPT.write_text(json.dumps(receipt,indent=2))
for ref in [binding['candidate_ref'],index['artifact_ref'],SCOPE+'/pr1-smoke-tokenizer:v0',
            SCOPE+'/nanochat-sft-readiness-review:v0',SCOPE+'/code-nanochat-l40-untouched-holdout:v0']:
    run.use_artifact(ref)

def publish(name,kind,files,metadata):
    assert all('protected-local-only' not in p.parts and p.suffix!='.parquet' for p,_ in files)
    hashes={target:sha(path) for path,target in files}
    checksum=OUT/(name+'-SHA256.json');checksum.write_text(json.dumps(hashes,indent=2))
    files=[*files,(checksum,'package-SHA256.json')]
    art=wandb.Artifact(name,type=kind,metadata={'training_approved':False,'launch_ready':False,
           'raw_holdout_text_uploaded':False,'human_review':'pending',**metadata})
    for path,target in files:art.add_file(str(path),name=target)
    logged=run.log_artifact(art,aliases=['automated-lexical-check-only']).wait()
    ref=SCOPE+'/'+logged.name
    receipt['artifacts'][name]={'ref':ref,'digest':logged.digest,'server_bytes_verified':False}
    RECEIPT.write_text(json.dumps(receipt,indent=2))
    server=api.artifact(ref)
    assert server.digest==logged.digest
    assert set(server.manifest.entries)=={target for _,target in files}
    assert server.metadata['training_approved'] is False and server.metadata['raw_holdout_text_uploaded'] is False
    local=Path(server.download(root=str(OUT/'server-verification'/name/server.version)))
    assert json.loads((local/'package-SHA256.json').read_text())==hashes
    for target,value in hashes.items():assert sha(local/target)==value
    result=receipt['artifacts'][name]
    result.update(server_bytes_verified=True,files=len(files),url=run.url.split('/runs/')[0]+f'/artifacts/{kind}/{name}/{server.version}')
    RECEIPT.write_text(json.dumps(receipt,indent=2));print(json.dumps({'published':result}),flush=True)
    return ref

fingerprint_ref=publish('nanochat-untouched-holdout-fingerprints','dataset-manifest',
    [(OUT/'fingerprints.json','fingerprints.json'),(OUT/'README.md','README.md'),
     (OUT/'origin-verification.json','origin-verification.json')],
    {'holdout_artifact_ref':index['artifact_ref'],'holdout_artifact_digest':index['artifact_digest'],
     'algorithm':index['algorithm'],'documents':index['documents'],
     'document_hashes':len(index['document_hashes']),'unique_13gram_hashes':len(index['shingle_hashes']),
     'independent_custodian_approval':False,'authorized_automated_export':True})
files=[(OUT/name,name) for name in ['README.md','comparison.json','candidate-binding-evidence.json',
    'candidate-line-map.json','origin-verification.json','independent-crosscheck.json']]
for path in [ROOT/'verify_holdout_origin.py',ROOT/'prepare_authorized_holdout_comparison.py',
             ROOT/'crosscheck_holdout_comparison.py',ROOT/'publish_authorized_holdout_check.py',
             ROOT/'readiness_review_v1/holdout_fingerprints.py',OUT/'producer-source/holdout_utils.py']:
    files.append((path,'audit-source/'+path.name))
publish('nanochat-sft-holdout-comparison','evaluation',files,
        {'candidate_ref':binding['candidate_ref'],'candidate_digest':candidate.digest,
         'fingerprint_artifact_ref':fingerprint_ref,'comparison_status':comparison['status'],
         'candidate_conversations':17018,'holdout_documents':2000,'lexical_flags':0,
         'independent_implementation_agreed':True,'semantic_deduplication_proven':False})
run.summary.update({'candidate_conversations':17018,'holdout_documents':2000,'lexical_flags':0,
                    'training_approved':False,'gpu_jobs_launched':0,'launch_ready':False})
run.finish()
server_run=api.run(SCOPE+'/'+receipt['run_id'])
assert server_run.state=='finished'
assert server_run.config['training_approved'] is False and server_run.config['gpu_jobs_launched']==0
receipt['run_state_verified']=server_run.state
receipt['candidate_training_approval_unchanged']=api.artifact(binding['candidate_ref']).metadata['training_approved'] is False
RECEIPT.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)