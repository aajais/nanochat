"""Offline packing and fail-closed, same-world-size SFT checkpoint protocol.

Checkpoints assume one shared filesystem mounted at the same path on all ranks.
Only COMPLETE.json makes a checkpoint restorable. No checkpoint overwrites.
"""
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist


class PhaseTimer:
    """Flushed phase events; no function arguments, credentials or responses."""
    def __init__(self, path, rank=0):
        self.path, self.rank, self.records = Path(path), rank, []

    def emit(self, record):
        value = dict(record, rank=self.rank, timestamp=time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False)+'\n')
            stream.flush()
        if self.rank == 0:
            print(json.dumps({'phase_timing': value}), flush=True)
        return value

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        self.emit(dict(phase=name, event='start'))
        try:
            yield
        except BaseException as exc:
            self.records.append(self.emit(dict(phase=name, event='error',
                seconds=time.perf_counter()-start, error_type=type(exc).__name__)))
            raise
        else:
            self.records.append(self.emit(dict(phase=name, event='end', seconds=time.perf_counter()-start)))

    def call(self, phase_name, function, *args, **kwargs):
        with self.phase(phase_name):
            return function(*args, **kwargs)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def collective_flag(flag, device):
    t = torch.tensor(int(flag), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


class PackedLoader:
    """Deterministic weighted draws with replacement, complete-conversation packing.

    `weights[category] / category_assistant_tokens` per example makes expected
    assistant-token shares proportional to requested category shares, before
    packing effects. Realized shares must still be logged.
    """
    def __init__(self, rows, sequence_len, batch_size, bos, weights, seed=42, buffer_size=32):
        self.rows = rows
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.bos = bos
        self.buffer_size = buffer_size
        if not rows or any(not 2 <= len(r['ids']) <= sequence_len + 1 for r in rows):
            raise ValueError('Empty dataset or overlength/empty conversation')
        for r in rows:
            if len(r['ids']) != len(r['mask']) or not any(r['mask'][1:]):
                raise ValueError('Invalid mask or zero supervised tokens')
        totals = {}
        for r in rows:
            c = r['category']
            totals[c] = totals.get(c, 0) + sum(r['mask'][1:])
        self.weights = [weights.get(r['category'], 0) / totals[r['category']] for r in rows]
        if sum(self.weights) <= 0:
            raise ValueError('No positive sampling weights')
        self.cumulative_weights = list(itertools.accumulate(self.weights))
        self.rng = random.Random(seed)
        self.buffer = []
        self.draws = 0
        self.category_tokens = {}

    def state_dict(self):
        return {'rng': self.rng.getstate(), 'buffer': list(self.buffer),
                'draws': self.draws, 'category_tokens': dict(self.category_tokens)}

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])
        self.buffer = list(state['buffer'])
        self.draws = state['draws']
        self.category_tokens = dict(state['category_tokens'])

    def next(self, device):
        rows, masks = [], []
        for _ in range(self.batch_size):
            ids, mask = [], []
            capacity = self.sequence_len + 1
            while len(ids) < capacity:
                while len(self.buffer) < self.buffer_size:
                    self.buffer.append(self.rng.choices(range(len(self.rows)), cum_weights=self.cumulative_weights, k=1)[0])
                    self.draws += 1
                fitting = [(len(self.rows[idx]['ids']), j) for j, idx in enumerate(self.buffer)
                           if len(self.rows[idx]['ids']) <= capacity - len(ids)]
                if not fitting:
                    break
                _, j = max(fitting)
                record = self.rows[self.buffer.pop(j)]
                ids.extend(record['ids'])
                mask.extend(record['mask'])
                cat = record['category']
                self.category_tokens[cat] = self.category_tokens.get(cat, 0) + sum(record['mask'])
            if not ids:
                raise RuntimeError('Packing made no progress')
            pad = capacity - len(ids)
            ids += [self.bos] * pad
            mask += [0] * pad
            rows.append(ids)
            masks.append(mask)
        batch = torch.tensor(rows, device=device, dtype=torch.long)
        target = batch[:, 1:].clone().contiguous()
        target[torch.tensor(masks, device=device)[:, 1:] == 0] = -1
        return batch[:, :-1].contiguous(), target


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def save_sft(root, step, model, optimizer, loader, loop, identity):
    from nanochat.distributed_readiness import optimizer_partition
    rank, world = rank_world()
    folder = Path(root) / f'step-{step:08d}'
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=False)
    barrier()
    if rank == 0:
        atomic_torch(folder / 'model.pt', model.state_dict())
        atomic_json(folder / 'loop.json', {'step': step, 'world_size': world,
                                        'loop': loop, 'identity': identity})
    atomic_torch(folder / f'rank-{rank:04d}.pt', {
        'optimizer': optimizer.state_dict(), 'loader': loader.state_dict(), 'rng': capture_rng(),
        'optimizer_partition':optimizer_partition(optimizer,rank,world) if step else None})
    barrier()
    if rank == 0:
        names = ['model.pt', 'loop.json'] + [f'rank-{r:04d}.pt' for r in range(world)]
        atomic_json(folder / 'COMPLETE.json', {name: {'sha256': sha256(folder/name),
                                                    'bytes': (folder/name).stat().st_size}
                                             for name in names})
    barrier()
    return folder


def validate_checkpoint(folder, identity, world):
    folder = Path(folder)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    expected = {'model.pt', 'loop.json'} | {f'rank-{r:04d}.pt' for r in range(world)}
    if set(complete) != expected:
        raise ValueError('Incomplete or wrong-world-size checkpoint')
    # Each rank validates every file; expensive, but fail-closed before optimizer load.
    for name, info in complete.items():
        path = folder / name
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError('Corrupt checkpoint file: ' + name)
    meta = json.loads((folder/'loop.json').read_text())
    if meta['world_size'] != world or meta['identity'] != identity:
        raise ValueError('Resume world size, code, tokenizer, data or configuration changed')
    return meta


def restore_sft(folder, model, optimizer, loader, identity, device):
    rank, world = rank_world()
    meta = validate_checkpoint(folder, identity, world)
    model.load_state_dict(torch.load(Path(folder)/'model.pt', map_location=device, weights_only=True), strict=True)
    # Only restore trusted checkpoints created by this workflow.
    state = torch.load(Path(folder)/f'rank-{rank:04d}.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    loader.load_state_dict(state['loader'])
    restore_rng(state['rng'])
    return meta


def lr_multiplier(step, horizon, warmup_steps, warmdown_ratio=0.5):
    if horizon <= 0 or warmup_steps < 1 or warmup_steps >= horizon:
        raise ValueError('Invalid schedule')
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    start = max(warmup_steps, int(horizon * (1 - warmdown_ratio)))
    if step < start:
        return 1.0
    return max(0.0, (horizon - 1 - step) / max(1, horizon - 1 - start))


class Deadline:
    def __init__(self, unix_deadline, save_reserve):
        self.unix_deadline, self.save_reserve = unix_deadline, save_reserve
        self.signalled = False
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)

    def _signal(self, *_):
        self.signalled = True

    def should_stop(self, device):
        return collective_flag(self.signalled or time.time() >= self.unix_deadline - self.save_reserve, device)