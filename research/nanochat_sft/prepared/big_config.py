"""Single user-approved 1.384B run. No successors, retries, or resume."""
import math,os
PROJECT='wandb/nanochat-demo'
QUEUE='nanochat-demo-k8s'
JOB='nanochat-1384m-8l40-startup3'
RUN_ID=os.environ.get('BIGTRAIN_RUN_ID','b1384b0')
NAME='nanochat-1.384B-8L40-'+RUN_ID
SCALE=math.sqrt(8192/16384)
CONFIG={
 'research_mode':'full_pretraining_1384m','research_protocol':'nanochat-1384m-8l40-20261120-v1',
 'depth':24,'aspect_ratio':64,'head_dim':128,'max_seq_len':2048,'window_pattern':'L',
 'device_batch_size':1,'total_batch_size':16384,'num_iterations':534529,
 'matrix_lr':0.04*SCALE,'embedding_lr':0.3*SCALE,'unembedding_lr':0.008*SCALE,
 'scalar_lr':0.5*SCALE,'weight_decay':0.56,'warmup_steps':100,'warmdown_ratio':0.25,
 'final_lr_frac':0.05,'target_param_data_ratio':12,'seed':42,'fp8':False,
 'eval_tokens':262144,'eval_every':10000,'save_every':66816,
 'parameter_count':1384122122,'scheduled_training_tokens':8757723136,
 'maximum_training_tokens':8757727488,'world_size':8,'gpu_type':'NVIDIA L40',
 'job_wall_limit_seconds':86400,'maximum_gpu_hours':192,'maximum_attempts':3,
 'startup_timeout_seconds':900,'retry_scope':'transient_startup_only_before_first_optimizer_step',
 'training_stop_after_entry_seconds':84900,'entry_hard_limit_seconds':86100,
 'automatic_retries':True,'automatic_successors':False,'automatic_resume':False,
 'training_from_scratch':True,'checkpoint_upload':True,
 'selected_config_run':'wandb/nanochat-demo/f3949dcc',
 'selected_nominal_recipe':{'matrix_lr':0.04,'embedding_lr':0.3,'unembedding_lr':0.008,
  'scalar_lr':0.5,'weight_decay':0.56,'total_batch_size':8192},
 'lr_transfer':'Nominal learning rates divided by sqrt(2) to offset the repository sqrt(batch) correction when moving batch8192 to16384. Width/muP and depth-dependent weight decay remain repository-native; not proven optimal at target scale.',
 'token_horizon_basis':'12*(transformer_matrices+lm_head), repository default; excludes lookup embeddings',
 'source_commit':'58d716e6255a5373f917ea9ca5e933d030db1779',
 'source_code_artifact':'wandb/nanochat-demo/code-nanochat-merged-rebaseline:v0',
 'tokenizer_artifact':'wandb/nanochat-demo/pr1-smoke-tokenizer:v0',
 'training_job_artifact':PROJECT+'/'+JOB+':v0',
 'data_revision':'915333b4f8b8684f39aeaafea600fea6f43fb703',
 'validation_role':'Development-only, context2048 and rank-strided documents; NOT directly matched to context256 BPB; prior public holdout protected but not evaluated.',
 'resume_note':'Sharded optimizer and per-rank loader cursors captured at terminal checkpoint; no bit-exact RNG/loader resume guarantee. No automatic resume authorized.',
 'approved_at':'2026-11-20T21:26:04Z',
}

def assigned_row_groups(count,rank,world_size):
 if not 0<=rank<world_size:raise ValueError('Invalid rank')
 return range(rank,count,world_size)

def validate_config(config):
 for key,value in CONFIG.items():
  actual=config.get(key)
  valid=(isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual) and math.isclose(actual,value,rel_tol=1e-14,abs_tol=0)) if isinstance(value,float) else actual==value
  if not valid:raise ValueError('Unexpected configuration: '+key)
 if config['total_batch_size']%(8*config['device_batch_size']*config['max_seq_len']):
  raise ValueError('Invalid distributed batch')