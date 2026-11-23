"""Metadata-only backfill from logged child config and verified pinned source."""
import ast,json,sys
from pathlib import Path
import wandb
ROOT=Path('/workspace/nanochat_sft');P=ROOT/'final_workflow_v2'
sys.path.insert(0,str(P))
from nanochat.workflow_policy import verify_package
package,identity=verify_package(P)
defaults={}
tree=ast.parse((P/'scripts/chat_sft_resumable.py').read_text())
for node in ast.walk(tree):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args:
  if isinstance(node.args[0],ast.Constant) and str(node.args[0].value).startswith('--'):
   for kw in node.keywords:
    if kw.arg=='default':
     try:defaults[node.args[0].value[2:].replace('-','_')]=ast.literal_eval(kw.value)
     except ValueError:pass
api=wandb.Api();parent=api.run('wandb/nanochat-demo/h1fkgo63');child=api.run('wandb/nanochat-demo/26bb573f')
assert parent.config['workflow_sha256']==identity
c=child.config
assert c['stage']=='readiness' and c['dataset_artifact']==package['assets']['candidate']['ref']
keys=['context','microbatch','global_input_tokens','grad_accum_steps','embedding_lr','unembedding_lr','matrix_lr','scalar_lr','lr_scale','load_optimizer']
updates={k:c[k] for k in keys}
for k in ['context','microbatch','global_input_tokens','embedding_lr','unembedding_lr','matrix_lr','scalar_lr','lr_scale','load_optimizer']:
 assert c[k]==defaults[k],k
rates={k.split('optim/base_lr_group_')[1]:v for k,v in child.summary.items() if k.startswith('optim/base_lr_group_')}
if not rates:raise ValueError('Observed optimizer base learning rates unavailable')
updates.update(world_size=8,optimizer='MuonAdamW',weight_decay=0.0,precision='bfloat16',fp8=False,
 loss='assistant_only',fresh_sft_optimizer=True,model=c['model'],base_step=c['base_step'],
 base_checkpoint=package['assets']['base']['ref'],dataset_artifact=c['dataset_artifact'],
 optimizer_group_base_lrs=rates,
 main_sft={
  'warmup_ratio':0.03,'warmdown_ratio':0.5,'weight_decay':0.0,
  'eval_every_updates':defaults['eval_every'],'eval_batches':defaults['eval_batches'],
  'checkpoint_every_seconds':defaults['save_every_seconds'],
  'save_reserve_seconds':defaults['save_reserve_seconds'],
  'update_budget_policy':'Frozen from eight-rank readiness throughput before main handoff; 100000 is a safety ceiling, not the selected budget',
  'max_updates_safety_ceiling':100000},
 readiness_sft={k:c[k] for k in ['fixed_horizon','checkpoint_steps','eval_every','save_every_seconds','calibration_steps','save_reserve_seconds']},
 hyperparameter_metadata={
  'source_child_run':'wandb/nanochat-demo/26bb573f','workflow_sha256':identity,
  'method':'Shared values from actual logged readiness config; main-only defaults verified against pinned source; optimizer base LRs observed in child summary.',
  'metadata_only':True,'running_training_settings_changed':False,
  'base_lrs_exclude_time_varying_schedule_multiplier':True})
old=dict(parent.config)
for k,v in updates.items():
 if k in old and old[k]!=v:raise ValueError('Refusing conflicting config overwrite: '+k)
before=ROOT/'hyperparameter-backfill-before.json'
if not before.exists():before.write_text(json.dumps(old,indent=2)+'\n')
parent.config.update(updates);parent.update()
verify_api=wandb.Api();verify_api.flush()
confirmed=verify_api.run('wandb/nanochat-demo/h1fkgo63')
for k,v in updates.items():
 if confirmed.config.get(k)!=v:raise ValueError('Server readback mismatch: '+k)
for k,v in old.items():
 if confirmed.config.get(k)!=v:raise ValueError('Existing config changed: '+k)
(ROOT/'hyperparameter-backfill-receipt.json').write_text(json.dumps({'run_id':confirmed.id,'state':confirmed.state,'updated':updates,'server_verified':True},indent=2)+'\n')
print(json.dumps({'run_id':confirmed.id,'state':confirmed.state,'updated_keys':list(updates),'observed_optimizer_group_base_lrs':rates,'server_verified':True}))