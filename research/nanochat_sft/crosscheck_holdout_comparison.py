"""Independent SHA256 13-gram check directly from corpus, not exported index.
Protected text is processed in memory, never displayed or written in reports.
"""
import hashlib,json,re,unicodedata
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'authorized_holdout_check_v1'
def words(text):return re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold())
def shingles(tokens):
    return {hashlib.sha256(' '.join(tokens[i:i+13]).encode('utf-8')).digest() for i in range(len(tokens)-12)}

exact_documents=set();index=set();documents=0
for batch in pq.ParquetFile(OUT/'protected-local-only/holdout.parquet').iter_batches(columns=['text'],batch_size=128):
    for text in batch.column(0).to_pylist():
        tokens=words(text);exact_documents.add(' '.join(tokens));index.update(shingles(tokens));documents+=1
assert documents==2000 and len(exact_documents)==2000
rows=0;exact_matches=[];shared_matches=[]
for line in (OUT/'retained-candidate-verified-texts.jsonl').open():
    row=json.loads(line);rows+=1
    texts=[m['content'] for m in row['messages']]
    texts.append('\n'.join(texts))
    tokens=[words(text) for text in texts]
    if any(' '.join(t) in exact_documents for t in tokens):exact_matches.append(row['id'])
    if any(shingles(t)&index for t in tokens):shared_matches.append(row['id'])
prior=json.loads((OUT/'comparison.json').read_text())
assert rows==prior['candidates_checked']==17018
assert not prior['lexical_flags'] and not exact_matches and not shared_matches
result={'independent_method':'direct normalized document strings and SHA256 13-grams from original Parquet; no exported fingerprint index used',
        'holdout_documents':documents,'unique_holdout_13grams':len(index),'candidate_conversations':rows,
        'exact_document_match_conversations':len(exact_matches),'shared_13gram_match_conversations':len(shared_matches),
        'agrees_with_exported_blake2b_index_result':True,'raw_holdout_text_displayed_or_uploaded':False,
        'independent_human_review':False,'semantic_deduplication_proven':False,'gpu_jobs_launched':0}
(OUT/'independent-crosscheck.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)