"""Reproducible public support-intent evaluation; never trains Nanochat."""
import argparse,hashlib,json,os,re,threading,time
from pathlib import Path
from eval_core import LABELS,PROMPT,LexicalNB,load_rows,parse_output,score

DEFAULT_DATASET='wandb/nanochat-demo/clinc150-support-fiveway:v0'
DEFAULT_TOKENIZER='wandb/nanochat-demo/pr1-smoke-tokenizer:v0'

def require_pinned(ref):
 if not isinstance(ref,str) or not re.fullmatch(r'[^/]+/[^/]+/[^:]+:v\d+',ref):
  raise ValueError('Use an explicit entity/project/artifact:vN reference, not latest')

def generate_label(model,tokenizer,text,device,max_new_tokens,context):
 import torch
 ids=[tokenizer.get_bos_token_id()]+tokenizer.encode(PROMPT.format(text=text))
 if len(ids)+max_new_tokens>context:raise ValueError('Prompt exceeds trained context; truncation is prohibited')
 generated=[]
 stop={tokenizer.get_bos_token_id(),tokenizer.encode_special('<|assistant_end|>')}
 with torch.inference_mode():
  for _ in range(max_new_tokens):
   logits=model(torch.tensor([ids+generated],dtype=torch.long,device=device))
   token=int(logits[0,-1].argmax())
   if token in stop:break
   generated.append(token)
   if '\n' in tokenizer.decode(generated):break
 return tokenizer.decode(generated),len(ids),len(generated)

def rank_labels(model,tokenizer,text,device,context):
 import torch
 import torch.nn.functional as F
 prefix=[tokenizer.get_bos_token_id()]+tokenizer.encode(PROMPT.format(text=text))
 means=[]
 with torch.inference_mode():
  for label in LABELS:
   # Boundary tokenization is explicit and frozen: prefix tokens then continuation tokens.
   suffix=tokenizer.encode(' '+label+'\n')
   seq=prefix+suffix
   if len(seq)>context:raise ValueError('Candidate exceeds trained context; truncation prohibited')
   x=torch.tensor([seq[:-1]],dtype=torch.long,device=device)
   logits=model(x)[0,len(prefix)-1:len(seq)-1]
   y=torch.tensor(suffix,dtype=torch.long,device=device)
   means.append(float(F.cross_entropy(logits,y,reduction='mean')))
 return LABELS[min(range(len(LABELS)),key=means.__getitem__)],means

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--backend',choices=['lexical_nb','nanochat'],required=True)
 parser.add_argument('--dataset',default=DEFAULT_DATASET)
 parser.add_argument('--split',choices=['dev','test','challenge_dev','challenge_test'],default='dev')
 parser.add_argument('--checkpoint',default=None)
 parser.add_argument('--method',choices=['greedy_label','label_likelihood'],default='greedy_label')
 parser.add_argument('--allow-public-test',action='store_true')
 args=parser.parse_args()
 import wandb
 run=wandb.init(entity='wandb',project='nanochat-demo',job_type='support-intent-evaluation',
  name='support-intent-'+args.backend,config={'backend':args.backend,'dataset_artifact':args.dataset,
   'split':args.split,'checkpoint_artifact':args.checkpoint,'method':args.method,
   'tokenizer_artifact':DEFAULT_TOKENIZER,'allow_public_test':args.allow_public_test,
   'wall_limit_seconds':900,'max_new_tokens':12,'max_examples':0,'training_performed':False})
 timer=None
 try:
  cfg=dict(run.config)
  backend=cfg['backend'];split=cfg['split'];method=cfg['method']
  if backend not in ('lexical_nb','nanochat') or method not in ('greedy_label','label_likelihood'):
   raise ValueError('Unsupported backend or method')
  if split in ('test','challenge_test') and cfg['allow_public_test'] is not True:
   raise ValueError('Public test evaluation requires explicit allow_public_test=true')
  limit=int(cfg['wall_limit_seconds'])
  if not 1<=limit<=900:raise ValueError('Wall limit must be1..900seconds')
  def stop():
   print('EVALUATION WALL LIMIT EXCEEDED',flush=True);os._exit(124)
  timer=threading.Timer(limit,stop);timer.daemon=True;timer.start()
  data_ref=cfg['dataset_artifact'];require_pinned(data_ref)
  artifact=run.use_artifact(data_ref)
  folder=Path.cwd()/('eval-assets-'+run.id);folder.mkdir(exist_ok=False)
  data_dir=folder/'dataset';artifact.download(root=str(data_dir));artifact.verify(root=str(data_dir))
  rows=load_rows(data_dir,split)
  count=int(cfg['max_examples'])
  if count<0:raise ValueError('max_examples must be >=0')
  if count:rows=rows[:count]
  predictions=[];latencies=[];records=[]
  if backend=='lexical_nb':
   model=LexicalNB().fit(load_rows(data_dir,'train'))
   run.config.update({'decision_rule':'multinomial_NB_uniform_class_prior_alpha1_unigram_bigram',
     'output_constrained_by_construction':True,'test_set_used_for_fit':False})
   def predict_one(text,example_id):
    start=time.perf_counter();pred=model.predict(text)
    return {'prediction':pred,'raw_output':pred,'latency_seconds':time.perf_counter()-start,
            'input_text':text,'method':'lexical_nb','inference_mode':'live'}
   identity={'checkpoint_artifact':'lexical_nb-fit-on-'+data_ref,
    'model_weight_sha256':hashlib.sha256((data_dir/'train.jsonl').read_bytes()).hexdigest(),
    'tokenizer_artifact':'lexical-unigram-bigram-v1','backend':backend,'method':'lexical_nb'}
  else:
   checkpoint=cfg['checkpoint_artifact'];require_pinned(checkpoint)
   require_pinned(cfg['tokenizer_artifact'])
   new_tokens=int(cfg['max_new_tokens'])
   if not 1<=new_tokens<=32:raise ValueError('max_new_tokens must be1..32')
   os.environ.update(HF_HUB_OFFLINE='1',NANOCHAT_DTYPE='bfloat16',NANOCHAT_WEAVE='0',NANOCHAT_LOG_CHECKPOINTS='0')
   import torch
   if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise ValueError('Nanochat job requires exactly one CUDA GPU')
   if not torch.cuda.is_bf16_supported():raise ValueError('BF16-capable GPU required')
   from nanochat.tokenizer import RustBPETokenizer
   from nanochat.gpt import GPT,GPTConfig
   # Read safe JSON metadata; only load weights from an explicitly chosen project artifact.
   if not checkpoint.startswith('wandb/nanochat-demo/'):
    raise ValueError('Checkpoint must belong to the current research project')
   ck=run.use_artifact(checkpoint);ckdir=folder/'checkpoint';ck.download(root=str(ckdir));ck.verify(root=str(ckdir))
   models=list(ckdir.rglob('model_*.pt'))
   if len(models)!=1:raise ValueError('Checkpoint artifact must contain exactly one model')
   model_path=models[0];step=int(model_path.stem.split('_')[-1])
   meta_path=model_path.with_name(f'meta_{step:06d}.json')
   meta=json.loads(meta_path.read_text());mc=meta['model_config']
   if int(meta.get('step',step))!=step:raise ValueError('Checkpoint metadata step mismatch')
   context=int(mc['sequence_len'])
   tok_art=run.use_artifact(cfg['tokenizer_artifact']);tokdir=folder/'tokenizer'
   tok_art.download(root=str(tokdir));tok_art.verify(root=str(tokdir))
   tok=RustBPETokenizer.from_directory(str(tokdir))
   if tok.get_vocab_size()!=mc['vocab_size']:raise ValueError('Checkpoint/tokenizer mismatch')
   device=torch.device('cuda:0')
   with torch.device('meta'):model=GPT(GPTConfig(**mc))
   model.to_empty(device=device);model.init_weights()
   weights=torch.load(model_path,map_location=device,weights_only=True)
   weights={k.removeprefix('_orig_mod.'):v for k,v in weights.items()}
   model.load_state_dict(weights,strict=True,assign=True);del weights
   model.eval()
   run.config.update({'runtime_gpu':torch.cuda.get_device_name(0),'runtime_torch':torch.__version__,
    'checkpoint_step':step,'trained_context':context,'prompt':PROMPT,
    'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
    'model_weight_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
    'output_constrained_by_construction':method=='label_likelihood',
    'decision_rule':method,'latency_scope':'serial CUDA-synchronized model calls, not serving latency'})
   # Fail before any prediction if any prompt would overflow. Never truncate.
   for row in rows:
    size=1+len(tok.encode(PROMPT.format(text=row['text'])))
    extra=new_tokens if method=='greedy_label' else max(len(tok.encode(' '+l+'\n')) for l in LABELS)
    if size+extra>context:raise ValueError('Dataset prompt overflows checkpoint context')
   # Warm-up uses a development input only, irrespective of evaluated split.
   warm=load_rows(data_dir,'dev')[0]['text']
   if method=='greedy_label':generate_label(model,tok,warm,device,new_tokens,context)
   else:rank_labels(model,tok,warm,device,context)
   torch.cuda.synchronize()
   def predict_one(text,example_id):
    torch.cuda.synchronize();start=time.perf_counter()
    if method=='greedy_label':
     raw,prompt_tokens,output_tokens=generate_label(model,tok,text,device,new_tokens,context)
     pred=parse_output(raw);details={'raw_output':raw,'prompt_tokens':prompt_tokens,'generated_tokens':output_tokens}
    else:
     pred,nll=rank_labels(model,tok,text,device,context);details={'candidate_mean_nll':dict(zip(LABELS,nll))}
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return {'prediction':pred,**details,'latency_seconds':elapsed,
     'prompt':PROMPT.format(text=text),'method':method,'inference_mode':'live'}
   identity={'checkpoint_artifact':checkpoint,'model_weight_sha256':run.config.model_weight_sha256,
    'tokenizer_artifact':cfg['tokenizer_artifact'],'backend':backend,'method':method,
    'max_new_tokens':new_tokens}
  # The central tracing adapter owns Model/Dataset/Scorer/Evaluation publication.
  # NANOCHAT_WEAVE=0 above suppresses the older ad-hoc integration only;
  # it does not disable this explicitly initialized structured evaluation.
  from structured_eval import execute
  outputs,receipt=execute(rows,predict_one,identity,dataset_artifact=data_ref,split=split)
  for row,output in zip(rows,outputs):
   predictions.append(output['prediction']);latencies.append(output['latency_seconds'])
   records.append({'id':row['id'],'expected':row['label'],**output})
  run.summary['eval/weave_call_id']=receipt.get('evaluation_call_id')
  run.summary['eval/weave_trace_url']=receipt.get('trace_url')
  run.summary['eval/weave_backend_verified']=False
  run.summary['eval/weave_trace_error']=receipt.get('trace_error')
  run.config.update({'structured_evaluation':True,'evaluation_schema':'support-structured-v1',
   'evaluation_role':'qualification' if not count and split in ('dev','challenge_dev') else 'smoke_or_final',
   'source_training_run':cfg.get('source_training_run'),
   'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
   'dataset_split_sha256':hashlib.sha256((data_dir/(split+'.jsonl')).read_bytes()).hexdigest()})
  result=score(rows,predictions,latencies)
  out=folder/'results';out.mkdir()
  (out/'weave-receipt.json').write_text(json.dumps(receipt,indent=2))
  (out/'metrics.json').write_text(json.dumps(result,indent=2))
  (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
  output=wandb.Artifact('support-intent-results-'+run.id,type='evaluation',metadata={
    'dataset':data_ref,'split':split,'backend':backend,'method':method if backend=='nanochat' else 'lexical_nb',
    'production_release_qualified':False})
  output.add_dir(str(out));run.log_artifact(output).wait()
  metrics={f'eval/{k}':v for k,v in result.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}
  for label,values in result['per_class'].items():
   metrics.update({f'eval/per_class/{label}/{k}':v for k,v in values.items()})
  if backend=='nanochat' and method=='greedy_label':metrics['eval/output_contract_validity']=1-result['invalid_rate']
  metrics['eval/completed']=True;metrics['eval/full_split']=not bool(count);run.log(metrics)
  print('RUN_URL='+run.url,flush=True);print('RESULT='+json.dumps(result),flush=True)
  run.finish()
 except BaseException:
  run.finish(exit_code=1);raise
 finally:
  if timer:timer.cancel()

if __name__=='__main__':main()