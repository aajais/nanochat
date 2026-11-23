"""One-GPU native-model plumbing check, not a quality evaluation or SFT release."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ['NANOCHAT_DTYPE'] = 'bfloat16'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.sft_runtime import PackedLoader, save_sft, restore_sft
from nanochat.tokenizer import RustBPETokenizer


def digest(value):
    h = hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); walk(x[k])
        elif isinstance(x, (tuple,list)):
            for v in x: walk(v)
        else:
            h.update(repr(x).encode())
    walk(value)
    return h.hexdigest()


def main():
    phase = sys.argv[1]
    root = Path('smoke_work')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported()
    torch.set_num_threads(8)
    torch.manual_seed(451); random.seed(451); np.random.seed(451)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # A deterministic single-GPU replay is the restore gate for this smoke.
    torch.use_deterministic_algorithms(True)
    metadata = json.loads((root/'base/meta_133632.json').read_text())
    config = GPTConfig(**metadata['model_config'])
    assert config.sequence_len == 2048 and config.vocab_size == 32768
    with torch.device('meta'):
        model = GPT(config)
    state = torch.load(root/'base/model_133632.pt', map_location='cpu', weights_only=False, mmap=True)
    assert len(state) == 175
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.cos, model.sin = model._precompute_rotary_embeddings(
        model.rotary_seq_len, config.n_embd//config.n_head, device='cpu')
    model = model.to('cuda').train()
    assert sum(p.numel() for p in model.parameters()) == 1384122122
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    tokenizer = RustBPETokenizer.from_directory(str(root/'tokenizer'))
    assert tokenizer.get_vocab_size() == config.vocab_size
    records = []
    for i in range(16):
        # Entirely synthetic, deliberately outside development/acceptance tasks.
        prefix = [tokenizer.get_bos_token_id(), tokenizer.encode_special('<|user_start|>')]
        prefix += tokenizer.encode(f'Infrastructure plumbing probe {i}: return the marker GPU_PROBE_{i:02d}.')
        prefix += [tokenizer.encode_special('<|user_end|>'), tokenizer.encode_special('<|assistant_start|>')]
        answer = tokenizer.encode(f'GPU_PROBE_{i:02d}') + [tokenizer.encode_special('<|assistant_end|>')]
        records.append({'ids':prefix+answer, 'mask':[0]*len(prefix)+[1]*len(answer), 'category':'general'})
    loader = PackedLoader(records,2048,1,tokenizer.get_bos_token_id(),{'general':1.0},seed=451)
    optimizer = model.setup_optimizer(unembedding_lr=.004, embedding_lr=.2, matrix_lr=.02,
                                      scalar_lr=.5, weight_decay=0.)
    for group in optimizer.param_groups:
        group['lr'] *= .1
        group['initial_lr'] = group['lr']
        group['weight_decay'] = 0.
    identity = json.loads((root/'identity.json').read_text())
    events = []
    def snapshot():
        torch.cuda.synchronize()
        return {'model':digest(model.state_dict()),'optimizer':digest(optimizer.state_dict()),
                'loader':digest(loader.state_dict()),'cpu_rng':digest(torch.get_rng_state()),
                'cuda_rng':digest(torch.cuda.get_rng_state_all()),
                'python_rng':digest(random.getstate()),'numpy_rng':digest(np.random.get_state())}
    def update(step):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        # Explicit changing schedule tests restoration of scheduler progress.
        factor = (1. if step == 1 else .5)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr']*factor
        assistant_tokens = 0
        total_loss = 0.
        for _ in range(8):
            x,y = loader.next('cuda')
            assistant_tokens += int((y!=-1).sum())
            loss = model(x,y)
            assert bool(torch.isfinite(loss)), 'Nonfinite loss'
            total_loss += float(loss.detach())/8
            (loss/8).backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite gradient'
        optimizer.step()
        torch.cuda.synchronize()
        event = {'phase':phase,'step':step,'finite_loss':True,'loss_for_debug_only':total_loss,
                 'input_tokens':16384,'assistant_tokens':assistant_tokens,
                 'seconds':time.monotonic()-started,
                 'optimizer_group_rates':[g['lr'] for g in optimizer.param_groups]}
        events.append(event)
        (root/f'{phase}_progress.json').write_text(json.dumps(events,indent=2))
        print(json.dumps(event),flush=True)
    if phase == 'save':
        initial = digest(next(model.parameters()))
        update(1)
        assert digest(next(model.parameters())) != initial, 'Weights did not change'
        # Gradients are not checkpoint state and must be cleared on both paths.
        optimizer.zero_grad(set_to_none=True)
        before = snapshot()
        checkpoint = save_sft(root/'checkpoints',1,model,optimizer,loader,
                              {'step':1,'horizon':3,'warmup_steps':1},identity)
        (root/'checkpoint_path.txt').write_text(str(checkpoint))
        assert snapshot() == before, 'Save changed replay state'
        (root/'saved_snapshot.json').write_text(json.dumps(before,indent=2))
        update(2)
        optimizer.zero_grad(set_to_none=True)
        expected = snapshot()
        (root/'expected_snapshot.json').write_text(json.dumps(expected,indent=2))
    elif phase == 'restore':
        checkpoint = (root/'checkpoint_path.txt').read_text()
        loop = restore_sft(checkpoint,model,optimizer,loader,identity,'cuda')['loop']
        assert loop == {'step':1,'horizon':3,'warmup_steps':1}, loop
        assert snapshot() == json.loads((root/'saved_snapshot.json').read_text()), 'Restored state differs before update'
        update(loop['step']+1)
        optimizer.zero_grad(set_to_none=True)
        actual = snapshot()
        expected = json.loads((root/'expected_snapshot.json').read_text())
        comparisons = {key:actual[key]==expected[key] for key in expected}
        assert all(comparisons.values()), comparisons
        result = {'passed':True,'fresh_process_restore':True,'exact_replay':comparisons,
                  'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(0),
                  'max_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'microbatch':1,'context':2048,'gradient_accumulation':8,
                  'input_tokens_per_update':16384,'no_quality_claim':True,
                  'single_gpu_only':True,'attention':'pytorch_sdpa','fp8':False,
                  'torch_compile':False}
        (root/'worker_result.json').write_text(json.dumps(result,indent=2))
    else:
        raise ValueError(phase)


if __name__ == '__main__':
    main()