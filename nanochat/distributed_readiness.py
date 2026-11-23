"""Eight-L40 topology, native optimizer layout, measured token-plan contract."""
import math
import torch

def verify_hardware(world=None):
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if names!=['NVIDIA L40']*8 or world not in (None,8):
        raise ValueError('Eight NVIDIA L40s on one node/world size eight required; no fallback')
    return dict(gpu_count=8,gpu_names=names)

def optimizer_partition(optimizer,rank,world):
    groups=[]
    for group in optimizer.param_groups:
        params=group['params'];kind=group['kind'];entries=[]
        if kind=='adamw':
            for p in params:
                state=optimizer.state[p]
                shape=list(p.shape) if p.numel()<1024 or world==1 else [p.shape[0]//world,*p.shape[1:]]
                if p.numel()>=1024 and p.shape[0]%world:raise ValueError('Indivisible AdamW shard')
                for name in ('exp_avg','exp_avg_sq'):
                    if name not in state or list(state[name].shape)!=shape:raise ValueError('AdamW shard mismatch')
                entries.append(dict(parameter_shape=list(p.shape),local_moment_shape=shape))
        elif kind=='muon':
            p=params[0];chunk=math.ceil(len(params)/world);state=optimizer.state[p]
            second=[chunk,p.shape[-2],1] if p.shape[-2]>=p.shape[-1] else [chunk,1,p.shape[-1]]
            if list(state['momentum_buffer'].shape)!=[chunk,*p.shape] or list(state['second_momentum_buffer'].shape)!=second:
                raise ValueError('Muon shard mismatch')
            entries.append(dict(parameter_shape=list(p.shape),group_parameters=len(params),chunk=chunk,
                owned_start=rank*chunk,owned_count=min(chunk,max(0,len(params)-rank*chunk))))
        else:raise ValueError('Unknown optimizer kind')
        groups.append(dict(kind=kind,entries=entries))
    return dict(rank=rank,world_size=world,optimizer_class=type(optimizer).__name__,groups=groups,validated=True)

def token_plan(proof,now,main_end,save_seconds=600,overhead_fraction=.25):
    if proof.get('world_size')!=8 or proof.get('same_world_size_verified') is not True:
        raise ValueError('Eight-rank proof required for token planning')
    stable=proof['first_result']['measurements'][-6:]
    if len(stable)<6 or any(m['world_size']!=8 or m['global_input_tokens']!=16384 or not math.isfinite(m['seconds']) or m['seconds']<=0 for m in stable):
        raise ValueError('Six stable production updates required')
    seconds=max(m['seconds'] for m in stable)
    assistant=sum(m['assistant_tokens'] for m in stable)/len(stable)
    available=max(0,main_end-now-save_seconds)
    updates=math.floor(available*(1-overhead_fraction)/seconds)
    if updates<4:raise ValueError('Insufficient measured training budget')
    return dict(world_size=8,global_input_tokens_per_update=16384,microbatch_per_rank=1,gradient_accumulation=1,
        measured_updates=6,measured_seconds_per_update=seconds,input_tokens_per_second=16384/seconds,
        supervised_tokens_per_second=assistant/seconds,available_main_seconds=available,
        checkpoint_and_validation_overhead_fraction=overhead_fraction,save_upload_reserved_seconds=save_seconds,
        expected_main_updates=updates,expected_input_tokens=updates*16384,
        expected_supervised_assistant_tokens=math.floor(updates*assistant),frozen_before_main=True,quality_guarantee=False)