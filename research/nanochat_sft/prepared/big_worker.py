"""One rank of the single approved torchrun execution."""
import json,os,runpy,sys,time
from pathlib import Path
import torch
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
import wandb
from big_config import CONFIG,NAME,RUN_ID,validate_config
import big_runtime as runtime
from big_data import DistributedData
import nanochat.dataloader as dataloader
import nanochat.observability as obs
from nanochat.common import DummyWandb

rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
assert world==8 and torch.cuda.device_count()==8
assert all(torch.cuda.get_device_name(i)=='NVIDIA L40' for i in range(8))
torch.set_num_threads(1)
base=Path(os.environ['NANOCHAT_BASE_DIR'])
dataset=DistributedData(base,rank,world)
dataloader._document_batches=dataset.batches
run=None
original_finish=None
if rank==0:
 worker_config=dict(CONFIG)
 provenance=json.loads(os.environ.get('BIGTRAIN_PROVENANCE','{}'))
 if provenance:worker_config['_wb_agent']=provenance
 run=wandb.init(entity='wandb',project='nanochat-demo',id=RUN_ID,name=NAME,config=worker_config)
 assert run.id==RUN_ID
 validate_config(dict(run.config))
 print('RUN_URL='+run.url,flush=True)
 for name in (CONFIG['tokenizer_artifact'],CONFIG['source_code_artifact']):run.use_artifact(name)
 run.config.update({'data_manifest':json.loads((base/'pilot-data-manifest.json').read_text()),
  'runtime_gpu':'NVIDIA L40','gpu_count':8,'distributed_data_sharding':'row_group_mod_world_size',
  'base_code_sha256':json.loads(Path('capture-manifest.json').read_text())},allow_val_change=True)
 run.define_metric('step')
 run.define_metric('*',step_metric='step')
 for key in ('train/loss','val/bpb'):run.define_metric(key,summary='min')
 runtime.RUN=run;runtime.BASE=base
 original_finish=run.finish
 # Keep the same run live through the terminal receipt after base_train returns.
 run.finish=lambda *a,**kw:None

def init_run(project,name,config,enabled=True):
 if not enabled:return DummyWandb()
 assert rank==0 and run is not None
 return run
obs.init_run=init_run
original_model_config=obs.log_model_config
def log_model_config(model,args,**resolved):
 assert sum(p.numel() for p in model.parameters())==CONFIG['parameter_count']
 from nanochat.common import COMPUTE_DTYPE
 assert COMPUTE_DTYPE==torch.bfloat16 and args.fp8 is False
 original_model_config(model,args,**resolved)
obs.log_model_config=log_model_config

keys=['depth','aspect_ratio','head_dim','max_seq_len','window_pattern','device_batch_size',
 'total_batch_size','num_iterations','matrix_lr','embedding_lr','unembedding_lr','scalar_lr',
 'weight_decay','warmup_steps','warmdown_ratio','final_lr_frac','eval_tokens','eval_every','save_every']
sys.argv=['scripts.base_train','--run',NAME,'--device-type','cuda',
 '--core-metric-every','-1','--sample-every','-1','--model-tag',RUN_ID]
for key in keys:sys.argv+=['--'+key.replace('_','-'),str(CONFIG[key])]
try:
 runpy.run_module('scripts.base_train',run_name='__main__')
 data=dataset.finish()
 if rank==0:
  assert runtime.LAST_STEP is not None and runtime.LAST_STEP>0 and runtime.LAST_CHECKPOINT
  step=runtime.LAST_STEP
  elapsed=time.monotonic()-float(os.environ['BIGTRAIN_ENTRY_STARTED'])
  full=step==CONFIG['num_iterations']
  final_meta=json.loads((base/'base_checkpoints'/RUN_ID/f'meta_{step:06d}.json').read_text())
  run.summary.update({'research/completed':full,'research/execution_finished':True,
   'research/final_val_bpb':final_meta['val_bpb'],
   'research/completed_steps':step,'research/training_tokens':step*CONFIG['total_batch_size'],
   'research/stop_reason':'token_horizon' if full else 'approved_wall_cap',
   'research/final_checkpoint':runtime.LAST_CHECKPOINT,'research/elapsed_entry_seconds':elapsed,
   'research/maximum_gpu_hours':192,'research/upper_bound_allocated_gpu_hours':elapsed*8/3600,
   'research/prior_holdout_evaluated':False,'research/automatic_successor':False})
  artifact=wandb.Artifact(RUN_ID+'-distributed-data',type='dataset',metadata={
   'world_size':8,'revision':CONFIG['data_revision'],'includes_corpus_text':False})
  artifact.add_file(str(base/'pilot-data-manifest.json'))
  for r in range(8):
   # Data receipt for another rank may arrive just after distributed cleanup.
   until=time.monotonic()+60
   while not (base/f'data-rank{r}.json').exists() and time.monotonic()<until:time.sleep(0.5)
   artifact.add_file(str(base/f'data-rank{r}.json'))
   artifact.add_file(str(base/f'streamed-rank{r}.jsonl'))
  run.log_artifact(artifact).wait(timeout=120)
  print('FULL_TRAINING_RESULT='+json.dumps({'steps':step,'tokens':step*16384,
   'stop_reason':'token_horizon' if full else 'approved_wall_cap','checkpoint':runtime.LAST_CHECKPOINT}),flush=True)
  original_finish()
except BaseException as exc:
 if rank==0 and run is not None:
  run.summary.update({'research/completed':False,'research/execution_finished':False,
   'research/failure':type(exc).__name__+': '+str(exc),'research/automatic_successor':False})
  original_finish(exit_code=1)
 raise