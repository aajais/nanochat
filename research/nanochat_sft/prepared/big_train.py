"""Bound one eight-L40 training attempt, including prep, compile and upload."""
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
from big_config import CONFIG,RUN_ID

started=time.monotonic()
parser=argparse.ArgumentParser()
parser.add_argument('--submitted-at',required=True,type=float)
submitted=parser.parse_args().submitted_at
elapsed_before_entry=max(0,time.time()-submitted)
hard_seconds=CONFIG['entry_hard_limit_seconds']-elapsed_before_entry
if hard_seconds<=0:raise RuntimeError('Approved wall-clock window expired before entry; no training or retry')
child=None
def terminate(signum,frame):
 if child is not None and child.poll() is None:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
 raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate)
signal.signal(signal.SIGINT,terminate)
def hard_stop():
 print('APPROVED_24H_BUDGET: force-stopping attempt; no retry',flush=True)
 if child is not None:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
 os._exit(124)

timer=threading.Timer(hard_seconds,hard_stop);timer.daemon=True;timer.start()
root=Path.cwd().resolve()
base=root/('bigtrain-assets-'+RUN_ID)
base.mkdir(exist_ok=False)
os.environ.update(WANDB_ENTITY='wandb',WANDB_PROJECT='nanochat-demo',WANDB_RUN_ID=RUN_ID,
 NANOCHAT_DTYPE='bfloat16',NANOCHAT_WEAVE='0',NANOCHAT_LOG_CODE='0',NANOCHAT_LOG_CHECKPOINTS='0',
 NANOCHAT_LOG_OPTIMIZER='0',NANOCHAT_PILOT_HOLDOUT='0',NANOCHAT_BASE_DIR=str(base),
 HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR=str(root/'inductor-cache'),
 TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NANOCHAT_EXECUTION_BACKEND='launch-8l40',
 BIGTRAIN_ENTRY_STARTED=str(started),
 BIGTRAIN_TRAIN_DEADLINE=str(started+max(0,CONFIG['training_stop_after_entry_seconds']-elapsed_before_entry)),USER='nanochat')
try:
 import torch,wandb
 if not torch.cuda.is_available() or torch.cuda.device_count()!=8:raise RuntimeError('Exactly eight CUDA GPUs required')
 names=[torch.cuda.get_device_name(i) for i in range(8)]
 if any(x!='NVIDIA L40' for x in names):raise RuntimeError('Approved L40 hardware mismatch: '+repr(names))
 if shutil.disk_usage(root).free<120*1024**3:raise RuntimeError('Require 120GiB free ephemeral storage for immutable checkpoints and upload staging')
 print('ALLOCATED_HARDWARE='+json.dumps({'gpus':names,'world_size':8,'wall_cap':86400}),flush=True)
 captured=json.loads((root/'capture-manifest.json').read_text())
 for name,digest in captured.items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Captured source changed: '+name)
 api=wandb.Api(timeout=60)
 token=api.artifact(CONFIG['tokenizer_artifact'])
 tok=base/'tokenizer';token.download(root=str(tok));token.verify(root=str(tok))
 from pilot_data import PilotData,docs_hash
 from big_data import RangeReader,SOURCE
 fake=SimpleNamespace(config={})
 data=PilotData(base,fake);data.prepare_validation()
 manifest=json.loads((base/'pilot-data-manifest.json').read_text())
 assert manifest['pilot_holdout']['sha256']=='798aeb231af45740b85799a89e2b835cf4ac08c02f6d3a34911a8701e48816af'
 manifest['pilot_holdout']['role']='Previously used pilot holdout; protected from training, NOT evaluated in this larger-model run'
 manifest['training_access']='Distinct row groups by modulo-eight rank; each physical row group loaded once, progressive bounded HTTP ranges'
 (base/'pilot-data-manifest.json').write_text(json.dumps(manifest,indent=2))
 # Fail before CUDA model allocation if this pinned corpus cannot shard as specified.
 import pyarrow as pa,pyarrow.parquet as pq
 with RangeReader(SOURCE+'shard_00000.parquet') as reader:
  groups=pq.ParquetFile(pa.PythonFile(reader,mode='r')).num_row_groups
 if groups<8:raise RuntimeError('Fewer than eight row groups in corpus shard')
 print('EIGHT_RANK_TRAINING_START='+json.dumps({'config':CONFIG,'first_shard_row_groups':groups,
  'setup_seconds':time.monotonic()-started}),flush=True)
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
  '--nproc-per-node=8','--max-restarts=0','big_worker.py']
 child=subprocess.Popen(command,env=os.environ.copy(),start_new_session=True)
 code=child.wait(timeout=max(1,hard_seconds-(time.monotonic()-started)))
 if code:raise RuntimeError('Distributed training failed with code '+str(code)+'; no retry')
finally:
 timer.cancel()
 if child is not None and child.poll() is None:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass