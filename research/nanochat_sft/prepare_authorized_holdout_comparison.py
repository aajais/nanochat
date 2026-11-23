"""Bind raw candidate messages to exact published v2 cache IDs/masks; no model calls."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prepared'))
from scripts.prepare_conversation_sft import normalize,text_of
from nanochat.tokenizer import RustBPETokenizer
from nanochat.sft_runtime import sha256

OUT=ROOT/'authorized_holdout_check_v1'
cache=ROOT/'conversation_cache_quality_v1'
manifest=json.loads((cache/'manifest.json').read_text())
records={};splits={}
for split in ('train','dev'):
    path=cache/(split+'.jsonl')
    assert sha256(path)==manifest['files'][path.name]
    for line in path.open():
        row=json.loads(line)
        assert row['id'] not in records
        records[row['id']]=row;splits[row['id']]=split
assert len(records)==17018
tok=RustBPETokenizer.from_directory(ROOT/'tokenizer')
assert sha256(ROOT/'tokenizer/tokenizer.pkl')==manifest['tokenizer_sha256']
target=OUT/'retained-candidate-verified-texts.jsonl'
seen=set();mapping=[];start=time.monotonic()
with target.open('x') as sink:
    for line in (ROOT/'conversation_candidates/selected.jsonl').open():
        raw=json.loads(line)
        key=hashlib.sha256(normalize(text_of(raw['messages'],True)).encode()).hexdigest()
        if key not in records or key in seen:continue
        ids,mask=tok.render_conversation({'messages':raw['messages']},max_tokens=10_000_000)
        if ids!=records[key]['ids'] or mask!=records[key]['mask']:continue
        seen.add(key)
        sink.write(json.dumps({'id':key,'split':splits[key],'messages':raw['messages']})+'\n')
        mapping.append({'candidate_line':len(mapping)+1,'id':key,'split':splits[key],'source':raw['source'],'source_row':raw['source_row']})
        if len(seen)%4000==0:print(json.dumps({'retained_rows_bound':len(seen),'elapsed_seconds':round(time.monotonic()-start)}),flush=True)
assert seen==set(records)
index=json.loads((OUT/'fingerprints.json').read_text())
holdout_manifest=json.loads((ROOT/'readiness_review_v1/holdout-manifest.json').read_text())
import pyarrow.parquet as pq
# The captured original producer hashes ordered, length-prefixed UTF-8 documents,
# NOT serialized Parquet bytes. The Parquet bytes separately match the immutable
# W&B manifest digest (verify_holdout_origin.py); verify both identities.
corpus=hashlib.sha256()
for batch in pq.ParquetFile(OUT/'protected-local-only/holdout.parquet').iter_batches(columns=['text'],batch_size=128):
    for text in batch.column(0).to_pylist():
        encoded=text.encode();corpus.update(len(encoded).to_bytes(8,'big'));corpus.update(encoded)
assert corpus.hexdigest()==holdout_manifest['holdout_sha256']
origin=json.loads((OUT/'origin-verification.json').read_text())
assert origin['parquet_bytes_match_immutable_artifact_manifest_md5'] is True
evidence={'candidate_ref':'wandb/nanochat-demo/conversation-sft-cache-candidate:v2','candidate_cache_files':manifest['files'],
          'tokenizer_sha256':manifest['tokenizer_sha256'],'retained_conversations_verified':len(seen),
          'candidate_text_sha256':sha256(target),'training_approved':False,
          'holdout_documents':index['documents'],'holdout_corpus_sha256_verified_against_original_manifest':True,
          'holdout_corpus_hash_method':'ordered documents; each UTF-8 payload prefixed by 8-byte big-endian byte length',
          'holdout_parquet_bytes_verified_against_immutable_artifact_manifest':True,
          'producer_corpus_hash_source_sha256':sha256(OUT/'producer-source/holdout_utils.py'),
          'holdout_document_fingerprints':len(index['document_hashes']),'holdout_unique_13gram_fingerprints':len(index['shingle_hashes']),
          'holdout_payload_opened_for_authorized_non_generative_fingerprinting':True,
          'raw_holdout_text_displayed_or_uploaded':False,'model_calls':0,'gpu_jobs_launched':0,'independent_human_approval':False}
(OUT/'candidate-line-map.json').write_text(json.dumps(mapping,indent=2))
(OUT/'candidate-binding-evidence.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(evidence,indent=2),flush=True)