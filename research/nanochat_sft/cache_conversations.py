"""Cache pinned public source shards. Never invokes a model or training service."""
import hashlib,json,os,time
from pathlib import Path
import requests
import pyarrow.parquet as pq

root=Path(__file__).resolve().parent
out=root/'conversation_sources'
out.mkdir(exist_ok=True)
repo='HuggingFaceTB/smoltalk'
# Immutable revision observed from the Hub API, never resolve main during download.
revision='5feaf2fd3ffca7c237fc38d1861bc30365d48ffa'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 session=requests.Session()
 tree=session.get(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/data?recursive=true&limit=100',timeout=30)
 tree.raise_for_status()
 entries={x['path']:x for x in tree.json()}
 components=['smol-magpie-ultra','smol-constraints','everyday-conversations','smol-summarize','metamathqa-50k']
 report={'repo':repo,'revision':revision,'files':{},'status':'source-cache-only-not-training-approved'}
 for component in components:
  name=f'data/{component}/train-00000-of-0000{6 if component=="smol-magpie-ultra" else 1}.parquet'
  entry=entries[name]
  target=out/(component+'.parquet')
  expected_sha=entry.get('lfs',{}).get('oid')
  cached=target.exists() and target.stat().st_size==entry['size']
  if cached and expected_sha:cached=sha(target)==expected_sha
  if not cached:
   print(json.dumps({'phase':'download','component':component,'bytes':entry['size']}),flush=True)
   partial=target.with_suffix('.partial')
   with session.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}',stream=True,timeout=(20,60)) as response:
    response.raise_for_status()
    with open(partial,'wb') as f:
     for chunk in response.iter_content(1<<20):
      if chunk:f.write(chunk)
     f.flush();os.fsync(f.fileno())
   if partial.stat().st_size!=entry['size']:raise ValueError('Download size mismatch')
   if expected_sha and sha(partial)!=expected_sha:raise ValueError('Download SHA mismatch')
   os.replace(partial,target)
  parquet=pq.ParquetFile(target)
  info={'path':name,'bytes':target.stat().st_size,'sha256':sha(target),
        'hub_lfs_sha256':expected_sha,'rows':parquet.metadata.num_rows,
        'columns':parquet.schema_arrow.names,'license':'mit (original MetaMathQA)' if component=='metamathqa-50k' else 'apache-2.0'}
  report['files'][component]=info
  (out/'sources.json').write_text(json.dumps(report,indent=2))
  print(json.dumps({'component':component,'status':'cached-and-verified',**info}),flush=True)
 card=session.get(f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md',timeout=30)
 card.raise_for_status();(out/'SMOLTALK_CARD.md').write_text(card.text)
 print(json.dumps({'status':'source-cache-complete','files':len(report['files']),'total_bytes':sum(x['bytes'] for x in report['files'].values())}),flush=True)

if __name__=='__main__':main()