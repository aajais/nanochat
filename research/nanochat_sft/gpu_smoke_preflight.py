"""CPU-only code preflight; not an experiment or model-quality evaluation."""
import json
import os
from pathlib import Path
import sys
import tempfile
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TORCHDYNAMO_DISABLE']='1'
sys.path.insert(0,'/workspace/nanochat_sft/gpu_smoke_job_v1')
import torch
from gpu_smoke_worker import digest
from nanochat.gpt import GPT,GPTConfig
from nanochat.sft_runtime import PackedLoader,save_sft,restore_sft,Deadline
torch.set_num_threads(2)
def components():
    model=GPT(GPTConfig(sequence_len=32,vocab_size=128,n_layer=2,n_head=2,n_kv_head=2,n_embd=64,window_pattern='L'))
    model.init_weights()
    optimizer=model.setup_optimizer(weight_decay=0.)
    for g in optimizer.param_groups:
        g['lr']*=.1;g['initial_lr']=g['lr'];g['weight_decay']=0.
    loader=PackedLoader([dict(ids=[1,2,3,4,5],mask=[0,0,0,1,1],category='general')],32,1,1,{'general':1.})
    return model,optimizer,loader
def update(m,o,l):
    o.zero_grad(set_to_none=True)
    x,y=l.next('cpu');loss=m(x,y);loss.backward();o.step();o.zero_grad(set_to_none=True)
def snapshot(m,o,l):return digest([m.state_dict(),o.state_dict(),l.state_dict()])
m,o,l=components();update(m,o,l)
with tempfile.TemporaryDirectory(dir='/workspace/nanochat_sft') as temp:
    before=snapshot(m,o,l)
    path=save_sft(temp,1,m,o,l,{'step':1,'horizon':3,'warmup_steps':1},{'preflight':True})
    assert snapshot(m,o,l)==before
    update(m,o,l);expected=snapshot(m,o,l)
    m2,o2,l2=components()
    restored=restore_sft(path,m2,o2,l2,{'preflight':True},'cpu')
    assert restored['loop']=={'step':1,'horizon':3,'warmup_steps':1}
    assert snapshot(m2,o2,l2)==before
    update(m2,o2,l2)
    assert snapshot(m2,o2,l2)==expected
assert Deadline(0,0).should_stop('cpu')
print(json.dumps({'cpu_code_preflight':'passed','native_optimizer_resume':True,'gpu_not_tested':True}),flush=True)