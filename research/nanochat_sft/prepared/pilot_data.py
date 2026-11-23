"""Pinned-revision, single-pass HTTP-range corpus loader for one bounded pilot."""
import hashlib,io,json
from pathlib import Path

REVISION='915333b4f8b8684f39aeaafea600fea6f43fb703'
SOURCE='https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/'+REVISION+'/'
EXPECTED_DEV='22c71aa2d7fa08aeef48bbb4c6ace2b818a52384ffc67cedd7140366ae0d4754'

class RangeReader(io.RawIOBase):
 def __init__(self,url):
  import requests
  self.session=requests.Session();self.pos=0;self.url=url
  with self.session.get(url,headers={'Range':'bytes=0-0'},timeout=(15,40),stream=True) as r:
   r.raise_for_status()
   if r.status_code!=206 or '/' not in r.headers.get('Content-Range',''):raise RuntimeError('Bounded ranges unavailable')
   self.size=int(r.headers['Content-Range'].split('/')[-1]);self.url=r.url
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if not 0<=self.pos<=self.size:raise ValueError('Invalid range seek')
  return self.pos
 def read(self,size=-1):
  size=self.size-self.pos if size<0 else min(size,self.size-self.pos)
  if not size:return b''
  if size>64*1024*1024:raise RuntimeError('HTTP range exceeds64MiB')
  with self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+size-1}'},timeout=(15,40),stream=True) as r:
   r.raise_for_status()
   if r.status_code!=206:raise RuntimeError('Server ignored bounded range')
   expected=f'bytes {self.pos}-{self.pos+size-1}/{self.size}'
   if r.headers.get('Content-Range')!=expected:raise RuntimeError('Unexpected Content-Range')
   result=r.raw.read(size+1)
   if len(result)!=size:raise RuntimeError('HTTP range length mismatch')
  self.pos+=size;return result
 def close(self):
  if hasattr(self,'session'):self.session.close()
  super().close()

def text_hash(text):return hashlib.sha256(text.encode()).digest()
def docs_hash(docs):
 h=hashlib.sha256()
 for text in docs:
  b=text.encode();h.update(len(b).to_bytes(8,'big'));h.update(b)
 return h.hexdigest()

class PilotData:
 def __init__(self,base,run):
  self.base=Path(base);self.run=run;self.protected=set();self.role='development'
  self.records=[];self.documents=0;self.excluded=0;self.shards=set()
  self.receipt=self.base/'streamed-row-groups.jsonl'
 def prepare_validation(self):
  import pyarrow as pa
  import pyarrow.parquet as pq
  docs=[]
  with RangeReader(SOURCE+'shard_06542.parquet') as reader:
   pf=pq.ParquetFile(pa.PythonFile(reader,mode='r'))
   for batch in pf.iter_batches(batch_size=1024,columns=['text']):
    for text in batch.column(0).to_pylist():
     if not isinstance(text,str) or not text.strip() or text_hash(text) in self.protected:continue
     self.protected.add(text_hash(text));docs.append(text)
     if len(docs)==4000:break
    if len(docs)==4000:break
  if len(docs)!=4000:raise RuntimeError('Insufficient disjoint validation documents')
  dev,heldout=docs[:2000],docs[2000:]
  if docs_hash(dev)!=EXPECTED_DEV:raise RuntimeError('Development validation differs from historical reference')
  self.paths={}
  for role,rows in [('development',dev),('pilot_holdout',heldout)]:
   path=self.base/(role+'.parquet');pq.write_table(pa.table({'text':rows}),path,row_group_size=1000);self.paths[role]=path
  self.manifest={'revision':REVISION,'source':SOURCE,'training_shard_range':[0,6541],
   'training_access':'progressive HTTP ranges, each shard and row group at most once; no full corpus download',
   'development':{'shard':6542,'documents':2000,'sha256':docs_hash(dev),'role':'historically tuned public validation prefix'},
   'pilot_holdout':{'shard':6542,'documents':2000,'sha256':docs_hash(heldout),'role':'next disjoint public validation documents; excluded from training; evaluated once at pilot end'},
   'validation_exact_overlap':0,'protected_document_hashes':len(self.protected)}
  (self.base/'pilot-data-manifest.json').write_text(json.dumps(self.manifest,indent=2))
  self.run.config.update({'data_manifest':self.manifest})
  print('VALIDATION_READY='+json.dumps(self.manifest),flush=True)
 def batches(self,split,resume_state_dict,tokenizer_batch_size):
  import pyarrow as pa
  import pyarrow.parquet as pq
  if resume_state_dict is not None:raise RuntimeError('This pilot starts from scratch; automatic resume not authorized')
  if split=='val':
   pf=pq.ParquetFile(self.paths[self.role]);epoch=1
   while True:
    for rg in range(pf.num_row_groups):
     rows=pf.read_row_group(rg,columns=['text']).column('text').to_pylist()
     for start in range(0,len(rows),tokenizer_batch_size):yield rows[start:start+tokenizer_batch_size],(6542,rg,epoch)
    epoch+=1
  elif split=='train':
   with self.receipt.open('a') as receipt:
    for shard in range(6542):
     url=SOURCE+f'shard_{shard:05d}.parquet'
     with RangeReader(url) as reader:
      pf=pq.ParquetFile(pa.PythonFile(reader,mode='r'))
      for rg in range(pf.num_row_groups):
       raw=pf.read_row_group(rg,columns=['text']).column('text').to_pylist()
       rows=[x for x in raw if isinstance(x,str) and x.strip() and text_hash(x) not in self.protected]
       excluded=sum(isinstance(x,str) and text_hash(x) in self.protected for x in raw)
       self.documents+=len(rows);self.excluded+=excluded;self.shards.add(shard)
       record={'shard':shard,'row_group':rg,'documents_loaded':len(rows),'sha256_logical_documents':docs_hash(rows),'excluded_validation_duplicates':excluded}
       receipt.write(json.dumps(record)+'\n');receipt.flush();self.records.append(record)
       if rg==0:print('STREAM_SHARD_STARTED='+str(shard),flush=True)
       for start in range(0,len(rows),tokenizer_batch_size):yield rows[start:start+tokenizer_batch_size],(shard,rg,1)
   raise RuntimeError('Full pinned training corpus exhausted; refusing repeat')
  else:raise ValueError('Unknown split')
 def final_manifest(self):
  result={**self.manifest,'training_shards_accessed':sorted(self.shards),'training_row_groups_loaded':len(self.records),
   'training_documents_loaded':self.documents,'excluded_validation_duplicates':self.excluded,
   'train_validation_exact_overlap_after_filtering':0,'note':'Loaded documents include bounded dataloader prefetch; scheduled optimizer tokens reported separately.'}
  (self.base/'final-data-manifest.json').write_text(json.dumps(result,indent=2));return result