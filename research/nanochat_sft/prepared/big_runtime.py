"""Collective stop/nonfinite guards and bounded eight-rank checkpoint publication."""
import json,math,os,time
from pathlib import Path
import torch
import torch.distributed as dist
RUN=None
BASE=None
LAST_STEP=None
LAST_CHECKPOINT=None

def mark_training_started(loss,step):
 if dist.is_initialized() and dist.get_rank()!=0:return
 if not math.isfinite(float(loss)):raise FloatingPointError('Nonfinite first training loss')
 marker=Path(os.environ['BIGTRAIN_STARTED_MARKER'])
 if not marker.exists():
  payload={'optimizer_steps':int(step)+1,'loss':float(loss),'run_id':RUN.id}
  pending=marker.with_suffix('.pending')
  pending.write_text(json.dumps(payload));pending.replace(marker)
  RUN.summary.update({'research/training_started':True,'research/first_optimizer_loss':float(loss),'research/first_optimizer_step':int(step)+1})
  print('FIRST_OPTIMIZER_STEP='+json.dumps(payload),flush=True)

def collective_any(flag,device):
 value=torch.tensor(int(bool(flag)),device=device,dtype=torch.int32)
 if dist.is_initialized():dist.all_reduce(value,op=dist.ReduceOp.MAX)
 return bool(value.item())

def budget_stop(device,deadline=None):
 deadline=float(os.environ['BIGTRAIN_TRAIN_DEADLINE']) if deadline is None else deadline
 rank=dist.get_rank() if dist.is_initialized() else 0
 return collective_any(rank==0 and time.monotonic()>=deadline,device)

def require_finite_loss(loss):
 if collective_any(not bool(torch.isfinite(loss.detach()).all().item()),loss.device):
  raise FloatingPointError('Nonfinite loss on at least one rank; no retry')

def require_finite_gradients(model):
 grads=[p.grad.detach() for p in model.parameters() if p.grad is not None]
 norms=torch.stack([torch.linalg.vector_norm(g,dtype=torch.float32) for g in grads])
 bad=not bool(torch.isfinite(norms).all().item())
 if collective_any(bad,model.get_device()):raise FloatingPointError('Nonfinite gradient on at least one rank; no retry')
 return float(torch.linalg.vector_norm(norms).item())

def require_finite_validation(score,device):
 if collective_any(not math.isfinite(float(score)),device):raise FloatingPointError('Nonfinite validation; no retry')

def checkpoint(checkpoint_dir,step,model_data,optimizer_data,meta_data,rank=0):
 global LAST_STEP,LAST_CHECKPOINT
 from nanochat.checkpoint_manager import save_checkpoint as native_save
 folder=Path(checkpoint_dir)
 folder.mkdir(parents=True,exist_ok=True)
 native_save(checkpoint_dir,step,model_data,optimizer_data,meta_data,rank=rank)
 (folder/f'rank_{rank}_step_{step:06d}.json').write_text(json.dumps(meta_data,indent=2))
 dist.barrier()
 if rank==0:
  import wandb
  files=[f'model_{step:06d}.pt',f'meta_{step:06d}.json']
  files += [f'rank_{r}_step_{step:06d}.json' for r in range(8)]
  if optimizer_data is not None:files += [f'optim_{step:06d}_rank{r}.pt' for r in range(8)]
  if not all((folder/f).is_file() and (folder/f).stat().st_size>0 for f in files):
   raise RuntimeError('Incomplete distributed checkpoint; no retry')
  artifact=wandb.Artifact(RUN.id+'-checkpoint',type='model',metadata={
   'step':step,'world_size':8,'parameter_count':1384122122,
   'includes_optimizer':optimizer_data is not None,'optimizer_shards':8 if optimizer_data is not None else 0,
   'validation_bpb':meta_data['val_bpb'],'exact_resume_guaranteed':False})
  for file in files:artifact.add_file(str(folder/file))
  result=RUN.log_artifact(artifact,aliases=['latest',f'step-{step}'])
  result.wait(timeout=600)
  LAST_CHECKPOINT='wandb/nanochat-demo/'+result.name
  RUN.summary.update({'research/latest_checkpoint':LAST_CHECKPOINT,
   'research/checkpoint_step':step,'research/checkpoint_has_optimizer':optimizer_data is not None})
  print('CHECKPOINT_PUBLISHED='+LAST_CHECKPOINT,flush=True)
 LAST_STEP=step
 dist.barrier()