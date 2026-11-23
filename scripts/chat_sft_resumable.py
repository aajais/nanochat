"""Candidate SFT entry point. Requires verified local assets; never fetches datasets.

Uses the pinned nanochat model and its native distributed MuonAdamW optimizer.
This is intentionally separate from the historical chat_sft.py entry point.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.common import compute_init, compute_cleanup, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.sft_runtime import (PackedLoader, Deadline, sha256, collective_flag,
                                 save_sft, restore_sft, lr_multiplier, barrier,
                                 capture_rng, restore_rng, atomic_json)
from nanochat.workflow_policy import approved_worker
from nanochat.distributed_readiness import verify_hardware


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--dataset-artifact', required=True, help='Exact dataset artifact version, never :latest')
    p.add_argument('--tokenizer-dir', required=True)
    p.add_argument('--base-dir', required=True, help='Local weights + meta directory')
    p.add_argument('--base-step', type=int, default=133632)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume', default=None, help='Complete SFT step directory, same world size only')
    p.add_argument('--load-optimizer', type=int, choices=[0], default=0)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--save-reserve-seconds', type=float, default=600)
    p.add_argument('--overhead-reserve-fraction', type=float, default=0.25)
    p.add_argument('--save-every-seconds', type=float, default=900)
    p.add_argument('--calibration-steps', type=int, default=12)
    p.add_argument('--max-updates', type=int, required=True)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--global-input-tokens', type=int, default=16384)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--run-name', default='nanochat-conversation-sft')
    p.add_argument('--embedding-lr', type=float, default=0.21213203435596426)
    p.add_argument('--unembedding-lr', type=float, default=0.005656854249492381)
    p.add_argument('--matrix-lr', type=float, default=0.028284271247461905)
    p.add_argument('--scalar-lr', type=float, default=0.3535533905932738)
    p.add_argument('--lr-scale', type=float, default=0.1)
    p.add_argument('--device-type', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--approval-file')
    p.add_argument('--allocation-started-unix',type=float)
    p.add_argument('--stage',choices=['readiness','main'],default='main')
    p.add_argument('--readiness-evidence')
    p.add_argument('--cpu-fixture',action='store_true',help='Tiny synthetic native-model CPU plumbing only, never GPU')
    p.add_argument('--fixed-horizon',type=int,default=0,help='Readiness-only deterministic schedule')
    p.add_argument('--stop-after-step',type=int,default=0)
    p.add_argument('--pause-after-step',type=int,default=0)
    p.add_argument('--inject-failure-step',type=int,default=0)
    p.add_argument('--inject-failure-rank',type=int,default=0)
    p.add_argument('--checkpoint-steps',type=int,nargs='*',default=[])
    p.add_argument('--training-plan-file')
    return p


def main():
    args = parser().parse_args()
    if args.max_updates <= args.calibration_steps + 2:
        raise ValueError('Update cap too short for calibration and schedule')
    if not 0 <= args.overhead_reserve_fraction < 1 or args.save_reserve_seconds <= 0:
        raise ValueError('Invalid deadline reserves')
    if args.deadline_unix <= time.time() + args.save_reserve_seconds:
        raise ValueError('Deadline leaves no training time')
    data_dir, tok_dir = Path(args.data_dir), Path(args.tokenizer_dir)
    manifest = json.loads((data_dir/'manifest.json').read_text())
    approved_worker(args,manifest)  # Separate explicit decisions, never flip immutable v3's approval flag.
    if args.device_type == 'cuda' and COMPUTE_DTYPE != torch.bfloat16:
        raise ValueError('This protocol requires BF16; set NANOCHAT_DTYPE=bfloat16')
    # The custom optimizer owns gradient reduction: do NOT wrap the model in DDP.
    ddp, rank, _, world, device = compute_init(args.device_type)
    if not args.cpu_fixture:verify_hardware(world)
    if rank==0:Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    barrier()
    if ddp and args.device_type != 'cuda':
        raise ValueError('CPU distributed training is not an approved runner')
    divisor = world * args.microbatch * args.context
    if args.global_input_tokens % divisor or args.global_input_tokens < divisor:
        raise ValueError('GPU count must divide global sequence batch; choose a divisor of 8')
    accumulation = args.global_input_tokens // divisor
    deadline = Deadline(args.deadline_unix, args.save_reserve_seconds)
    if manifest['context'] != args.context:
        raise ValueError('Dataset context differs')
    for name, digest in manifest['files'].items():
        if sha256(data_dir/name) != digest:
            raise ValueError('Dataset hash failure: ' + name)
    tok_hash = sha256(tok_dir/'tokenizer.pkl')
    if manifest['tokenizer_sha256'] != tok_hash:
        raise ValueError('Dataset/tokenizer mismatch')
    if args.cpu_fixture:
        class FixtureTokenizer:
            def get_vocab_size(self):return 256
            def get_bos_token_id(self):return 0
        tokenizer=FixtureTokenizer()
    else:
        tokenizer = RustBPETokenizer.from_directory(tok_dir)
    base_meta_path = Path(args.base_dir)/f'meta_{args.base_step:06d}.json'
    base_path = Path(args.base_dir)/f'model_{args.base_step:06d}.pt'
    meta = json.loads(base_meta_path.read_text())
    cfg = GPTConfig(**meta['model_config'])
    if cfg.sequence_len != args.context or cfg.vocab_size != tokenizer.get_vocab_size():
        raise ValueError('Model, context, and tokenizer incompatible')
    # Use explicit strict loading: no architecture fallback and no key patching.
    with torch.device('meta'):
        raw_model = GPT(cfg)
    raw_model.to_empty(device=device)
    raw_model.init_weights()
    state = torch.load(base_path, map_location=device, weights_only=True)
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    raw_model.load_state_dict(state, strict=True, assign=True)
    del state
    optimizer = raw_model.setup_optimizer(embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr, matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr, weight_decay=0.0)
    # setup_optimizer hard-codes AdamW decay and smear-gate LR. Override all
    # groups explicitly; scale the effective rates, including fixed-rate groups.
    for group in optimizer.param_groups:
        group['weight_decay'] = 0.0
        group['lr'] *= args.lr_scale
        group['initial_lr'] = group['lr']
    train_rows, val_rows = read_rows(data_dir/'train.jsonl'), read_rows(data_dir/'dev.jsonl')
    loader = PackedLoader(train_rows, args.context, args.microbatch,
                          tokenizer.get_bos_token_id(), manifest['effective_shares'], seed=42+rank)
    source = Path(__file__).resolve().parents[1]
    code_hashes = {str(p.relative_to(source)): sha256(p) for parent in ('nanochat','scripts','tasks')
                   for p in sorted((source/parent).glob('*.py'))}
    invariant_config = {k:v for k,v in vars(args).items() if k not in
                        {'resume','output_dir','run_name','base_dir','data_dir','tokenizer_dir',
                         'stop_after_step','pause_after_step','inject_failure_step','inject_failure_rank','readiness_evidence'}}
    identity = {'config': invariant_config, 'code': code_hashes,
                'dataset_manifest': sha256(data_dir/'manifest.json'),
                'tokenizer_sha256': tok_hash, 'base_metadata_sha256': sha256(base_meta_path),
                'base_weights_sha256': sha256(base_path)}
    loop = {'step': 0, 'horizon': None, 'warmup_steps': None, 'input_tokens': 0,
            'assistant_tokens': 0, 'best_dev_loss': None, 'dev_worsening_streak': 0,
            'calibration_seconds': [], 'stop_reason': None}
    if args.fixed_horizon:
        loop.update(horizon=args.fixed_horizon,warmup_steps=1)
    if args.training_plan_file:
        plan=json.loads(Path(args.training_plan_file).read_text())
        loop.update(horizon=plan['expected_main_updates'],warmup_steps=max(1,math.ceil(.03*plan['expected_main_updates'])))
    resume_rng = None
    if args.resume:
        restored = restore_sft(args.resume, raw_model, optimizer, loader, identity, device)
        resume_rng = capture_rng()
        loop = restored['loop']
        if loop['step'] != restored['step']:
            raise ValueError('Step mismatch')
    model = raw_model if args.no_compile else torch.compile(raw_model, dynamic=False)
    run = None
    try:
        if rank == 0:
            provenance = {key: os.environ[name] for key,name in
                          [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')]
                          if name in os.environ}
            run = wandb.init(entity='wandb', project='nanochat-demo', name=args.run_name,
                             config={**vars(args), '_wb_agent': provenance, 'identity':identity,
                                     'model':asdict(cfg), 'grad_accum_steps':accumulation})
            for ref, kind in ([] if args.cpu_fixture else [('b1384b1-checkpoint:v2','model'),
                              ('pr1-smoke-tokenizer:v0','tokenizer')]):
                run.use_artifact('wandb/nanochat-demo/'+ref)
            if ':latest' in args.dataset_artifact or ':' not in args.dataset_artifact:
                raise ValueError('Exact dataset artifact version required')
            if not args.cpu_fixture:
                run.use_artifact(args.dataset_artifact)
            run.log({f'optim/base_lr_group_{i}': g['initial_lr'] for i,g in enumerate(optimizer.param_groups)})
        barrier()
        if resume_rng is not None:
            # Logging/provenance setup must not perturb the restored random stream.
            restore_rng(resume_rng)
        last_save = time.time()
        last_saved_step = -1
        receipts=[]
        measurements=[]
        worker_pids=[os.getpid()]
        if dist.is_initialized():
            worker_pids=[None]*world;dist.all_gather_object(worker_pids,os.getpid())

        def checkpoint():
            nonlocal last_save, last_saved_step
            if loop['step'] == last_saved_step:
                return
            checkpoint_rng=capture_rng()
            folder = save_sft(args.output_dir, loop['step'], raw_model, optimizer, loader, loop, identity)
            if rank == 0:
                art = wandb.Artifact(f'{run.id}-sft-resume', type='model',
                    metadata={'step':loop['step'], 'world_size':world, 'includes_optimizer':True,
                              'format':'nanochat-sft-complete-v1','same_world_size_only':True})
                art.add_dir(str(folder))
                if args.cpu_fixture:
                    receipt={'step':loop['step'],'local_only_cpu_fixture':True,'server_verified':False}
                else:
                    published=run.log_artifact(art, aliases=[f'step-{loop["step"]}']).wait()
                    ref='wandb/nanochat-demo/'+published.name
                    server=wandb.Api().artifact(ref)
                    complete=json.loads((folder/'COMPLETE.json').read_text())
                    expected=set(complete)|{'COMPLETE.json'}
                    if set(server.manifest.entries)!=expected or server.digest!=published.digest:
                        raise ValueError('Server checkpoint manifest mismatch')
                    # Actual server-byte verification is mandatory before handoff.
                    verified=Path(server.download(root=str(Path(args.output_dir)/'verified'/server.version)))
                    if sha256(verified/'COMPLETE.json')!=sha256(folder/'COMPLETE.json'):
                        raise ValueError('Server COMPLETE differs')
                    for name,entry in complete.items():
                        if sha256(verified/name)!=entry['sha256']:
                            raise ValueError('Server checkpoint bytes differ: '+name)
                    receipt={'step':loop['step'],'ref':ref,'digest':server.digest,'server_verified':True}
                receipts.append(receipt)
                atomic_json(Path(args.output_dir)/'uploads.json',receipts)
            barrier()
            restore_rng(checkpoint_rng)  # Service/upload bookkeeping must not perturb continuation.
            last_save, last_saved_step = time.time(), loop['step']

        while True:
            step = loop['step']
            horizon = loop['horizon'] or args.max_updates
            if deadline.should_stop(device) or step >= horizon or (args.stop_after_step and step>=args.stop_after_step):
                loop['stop_reason'] = ('signal' if deadline.signalled else 'deadline') if step < horizon and not (args.stop_after_step and step>=args.stop_after_step) else ('readiness_stop' if args.stop_after_step and step>=args.stop_after_step else 'schedule_complete')
                checkpoint()
                break
            start = time.perf_counter()
            microbatches = [loader.next(device) for _ in range(accumulation)]
            local_count = sum(int((y != -1).sum()) for _,y in microbatches)
            counts = torch.tensor(local_count, device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(counts)
            if counts.item() == 0:
                raise ValueError('No supervised tokens')
            if loop['horizon'] is None:
                # Low, fixed calibration rate; joins the measured-horizon warmup continuously.
                multiplier = 0.01
            elif step < loop['warmup_steps']:
                offset=0 if args.fixed_horizon or args.training_plan_file else args.calibration_steps
                multiplier = 0.01 + 0.99 * ((step-offset+1) /
                                  max(1,loop['warmup_steps']-offset))
            else:
                multiplier = lr_multiplier(step,horizon,loop['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * multiplier
                if group['kind'] == 'muon':
                    group['momentum'] = 0.85 + 0.10*min(step/300,1)
            raw_model.zero_grad(set_to_none=True)
            sum_loss = torch.zeros((), device=device)
            for x,y in microbatches:
                loss = model(x,y,loss_reduction='sum')
                if collective_flag(not bool(torch.isfinite(loss)),device):
                    raise FloatingPointError('Nonfinite loss; restore only a prior complete checkpoint')
                (loss * world / counts).backward()
                sum_loss += loss.detach()
            bad_grad = any(not bool(torch.isfinite(p.grad).all()) for p in raw_model.parameters() if p.grad is not None)
            if collective_flag(bad_grad,device):
                raise FloatingPointError('Nonfinite gradients; no automatic retry')
            optimizer.step()
            raw_model.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            duration = time.perf_counter()-start
            measured=torch.tensor(duration,device=device,dtype=torch.float64)
            if dist.is_initialized():dist.all_reduce(measured,op=dist.ReduceOp.MAX)
            duration=float(measured.item())
            if dist.is_initialized():
                dist.all_reduce(sum_loss)
            loop['step'] += 1
            if args.inject_failure_step and loop['step']==args.inject_failure_step and rank==args.inject_failure_rank:
                raise RuntimeError('Intentional readiness failure propagation probe')
            loop['input_tokens'] += args.global_input_tokens
            loop['assistant_tokens'] += int(counts.item())
            measurements.append(dict(step=loop['step'],seconds=duration,global_input_tokens=args.global_input_tokens,
                assistant_tokens=int(counts.item()),world_size=world))
            if loop['horizon'] is None:
                loop['calibration_seconds'].append(duration)
                if loop['step'] == args.calibration_steps:
                    # Exclude initial compilation/warmup updates; use slowest stable rank.
                    stable = loop['calibration_seconds'][max(0,args.calibration_steps//2):]
                    seconds = torch.tensor(sum(stable)/len(stable), device=device)
                    if dist.is_initialized():
                        dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
                    remaining = max(0,args.deadline_unix-time.time()-args.save_reserve_seconds)
                    count = int(remaining*(1-args.overhead_reserve_fraction)/seconds.item())
                    horizon = min(args.max_updates,loop['step']+count)
                    frozen = torch.tensor(horizon,device=device,dtype=torch.int64)
                    if dist.is_initialized():
                        dist.broadcast(frozen,src=0)
                    horizon = int(frozen.item())
                    if horizon <= loop['step']+2:
                        raise RuntimeError('Insufficient time after calibration')
                    loop['horizon'] = horizon
                    loop['warmup_steps'] = min(horizon-1,max(loop['step']+1,math.ceil(.03*horizon)))
            metrics = {'sft/step':loop['step'],'train/loss':float(sum_loss/counts),
                       'tokens/input_slots':loop['input_tokens'],'tokens/assistant':loop['assistant_tokens'],
                       'train/input_tokens_per_second':args.global_input_tokens/duration,
                       'train/lr_multiplier':multiplier,'schedule/horizon':loop['horizon'] or 0,
                       **{f'optim/lr_group_{i}':g['lr'] for i,g in enumerate(optimizer.param_groups)}}
            # Fixed, offline development samples; never final acceptance examples.
            if args.eval_every > 0 and loop['step'] % args.eval_every == 0 and not deadline.should_stop(device):
                val = PackedLoader(val_rows,args.context,args.microbatch,tokenizer.get_bos_token_id(),
                                   manifest['effective_shares'],seed=1729+rank)
                sums = torch.zeros(2,device=device,dtype=torch.float64)
                raw_model.eval()
                with torch.no_grad():
                    for _ in range(args.eval_batches):
                        x,y=val.next(device)
                        sums[0] += raw_model(x,y,loss_reduction='sum')
                        sums[1] += (y!=-1).sum()
                if dist.is_initialized():
                    dist.all_reduce(sums)
                raw_model.train()
                dev_loss=float(sums[0]/sums[1])
                if not math.isfinite(dev_loss):
                    raise FloatingPointError('Nonfinite development loss')
                metrics['dev/assistant_nll']=dev_loss
                best=loop['best_dev_loss']
                loop['dev_worsening_streak'] = loop['dev_worsening_streak']+1 if best is not None and dev_loss>best else 0
                loop['best_dev_loss'] = min(best,dev_loss) if best is not None else dev_loss
                # Loss-only stopping is NOT equivalent to the conversational stop rule.
                metrics['review/conversation_check_required']=int(loop['dev_worsening_streak']>=2)
            # Report realized supervised-token proportions summed across all ranks.
            cats=sorted(manifest['effective_shares'])
            category_counts=torch.tensor([loader.category_tokens.get(c,0) for c in cats],device=device,dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(category_counts)
            metrics.update({f'tokens/category_{c}':float(category_counts[i]) for i,c in enumerate(cats)})
            if rank==0:
                run.log(metrics)
                print(json.dumps(metrics),flush=True)
                atomic_json(Path(args.output_dir)/'progress.json',{'step':loop['step'],'finite':True,'worker_pids':worker_pids,'world_size':world})
            save_due = collective_flag(time.time()-last_save>=args.save_every_seconds or loop['step'] in args.checkpoint_steps,device)
            if save_due:
                checkpoint()
            if args.pause_after_step and loop['step']>=args.pause_after_step:
                while not deadline.should_stop(device):time.sleep(.05)
        if rank==0:
            run.summary['stop_reason']=loop['stop_reason']
            atomic_json(Path(args.output_dir)/'result.json',dict(passed=True,loop=loop,identity=identity,
                world_size=world,device=str(device),compute_dtype=str(COMPUTE_DTYPE),
                cpu_fixture=args.cpu_fixture,uploads=receipts,measurements=measurements))
            run.finish()
    except BaseException as exc:
        atomic_json(Path(args.output_dir)/f'failure.rank-{rank:04d}.json',{'passed':False,'rank':rank,'error':type(exc).__name__+': '+str(exc),'step':loop['step']})
        if rank==0 and run is not None:
            atomic_json(Path(args.output_dir)/'failure.json',{'passed':False,'error':type(exc).__name__+': '+str(exc),'step':loop['step']})
            run.finish(exit_code=1)
        raise  # torchrun must retain the worker's nonzero exit status
    finally:
        compute_cleanup()


if __name__ == '__main__':
    main()