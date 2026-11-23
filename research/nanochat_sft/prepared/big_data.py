"""Pinned public corpus: distinct row groups per rank, bounded HTTP reads."""
import hashlib,json
from pathlib import Path
from pilot_data import RangeReader,SOURCE,REVISION,text_hash,docs_hash,EXPECTED_DEV
from big_config import assigned_row_groups

class DistributedData:
 def __init__(self,base,rank,world):
  self.base=Path(base);self.rank=rank;self.world=world
  self.records=0;self.documents=0;self.excluded=0;self.shards=set()
  manifest=json.loads((self.base/'pilot-data-manifest.json').read_text())
  assert manifest['revision']==REVISION and manifest['development']['sha256']==EXPECTED_DEV
  import pyarrow.parquet as pq
  self.dev=pq.read_table(self.base/'development.parquet',columns=['text']).column('text').to_pylist()
  protected=self.dev+pq.read_table(self.base/'pilot_holdout.parquet',columns=['text']).column('text').to_pylist()
  self.protected={text_hash(x) for x in protected}
  assert len(self.protected)==4000
  self.receipt=self.base/f'streamed-rank{rank}.jsonl'

 def batches(self,split,resume_state_dict,tokenizer_batch_size):
  import pyarrow as pa
  import pyarrow.parquet as pq
  if resume_state_dict is not None:raise RuntimeError('Automatic resume is not authorized')
  if split=='val':
   rows=self.dev[self.rank::self.world]
   if len(rows)<128:raise RuntimeError('Insufficient rank-local validation documents')
   epoch=1
   while True:
    for start in range(0,len(rows),tokenizer_batch_size):
     yield rows[start:start+tokenizer_batch_size],(6542,self.rank,epoch)
    epoch+=1
  elif split=='train':
   with self.receipt.open('x') as receipt:
    for shard in range(6542):
     with RangeReader(SOURCE+f'shard_{shard:05d}.parquet') as reader:
      pf=pq.ParquetFile(pa.PythonFile(reader,mode='r'))
      if pf.num_row_groups<self.world:raise RuntimeError('Too few row groups for eight-rank single-pass sharding')
      for rg in assigned_row_groups(pf.num_row_groups,self.rank,self.world):
       raw=pf.read_row_group(rg,columns=['text']).column('text').to_pylist()
       rows=[x for x in raw if isinstance(x,str) and x.strip() and text_hash(x) not in self.protected]
       excluded=sum(isinstance(x,str) and text_hash(x) in self.protected for x in raw)
       self.documents+=len(rows);self.excluded+=excluded;self.records+=1;self.shards.add(shard)
       record={'rank':self.rank,'shard':shard,'row_group':rg,'documents_loaded':len(rows),
        'sha256_logical_documents':docs_hash(rows),'excluded_validation_duplicates':excluded}
       receipt.write(json.dumps(record)+'\n');receipt.flush()
       if rg==self.rank:print(f'RANK{self.rank}_STREAM_SHARD={shard}',flush=True)
       for start in range(0,len(rows),tokenizer_batch_size):
        yield rows[start:start+tokenizer_batch_size],(shard,rg,1)
   raise RuntimeError('Pinned single-pass corpus exhausted; refusing repeat')
  else:raise ValueError('Unknown split')

 def finish(self):
  result={'rank':self.rank,'world_size':self.world,'training_row_groups_loaded':self.records,
   'training_documents_loaded':self.documents,'training_shards':sorted(self.shards),
   'excluded_validation_duplicates':self.excluded,'revision':REVISION,
   'training_sharding':'row_group % world_size == rank; physical groups nonoverlapping; no claim of semantic or corpus-wide text deduplication',
   'note':'Loaded documents include dataloader prefetch; optimizer-token counts tracked separately.'}
  (self.base/f'data-rank{self.rank}.json').write_text(json.dumps(result,indent=2))
  return result