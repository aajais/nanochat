"""Offline, CPU-only binding of verified nanochat checkpoints to the eval runner.

No inference occurs while loading. A manifest's SHA256 must be supplied from an
independently trusted receipt; computing a hash of an arbitrary manifest is NOT
authentication. This API is preparation plumbing, not GPU/release approval.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

BASE_WEIGHTS = 'c76eb3a6e3c482647e7d0f0f7077689369d060593762803678e2d1d4fcbbdbf0'
BASE_METADATA = '7ef3096f4dbc72b4a7a7f879aed0a0f69abc810317046183349741f8b9decaec'
TOKENIZER = 'a228803c68627f650e640c819e71ea0b90f49a96118858ed781a00a5ba63b6bf'
MODEL_CONFIG = dict(sequence_len=2048, vocab_size=32768, n_layer=24,
                    n_head=12, n_kv_head=12, n_embd=1536, window_pattern='L')
SOURCE_NAMES = ('gpt.py', 'common.py', 'optim.py', 'flash_attention.py',
                'tokenizer.py', 'conversation_eval.py', 'conversation_eval_checkpoint.py')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_file(root, name, expected):
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Require SHA256 for ' + str(name))
    root = Path(root).resolve()
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('Relative file path required')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing file or path escapes root: ' + name)
    if sha256(path) != expected:
        raise ValueError('SHA256 mismatch: ' + name)
    return path


def verify_manifest(root, manifest_path, trusted_manifest_sha256, tokenizer_dir):
    manifest_path = Path(manifest_path)
    checked_file(manifest_path.parent, manifest_path.name, trusted_manifest_sha256)
    spec = json.loads(manifest_path.read_text())
    if spec.get('schema') != 'nanochat-eval-checkpoint-v1':
        raise ValueError('Unknown checkpoint manifest schema')
    if spec.get('model_config') != MODEL_CONFIG:
        raise ValueError('No model/architecture substitution is allowed')
    if spec.get('base_weights_sha256') != BASE_WEIGHTS:
        raise ValueError('Wrong base lineage')
    if spec.get('tokenizer_sha256') != TOKENIZER:
        raise ValueError('Wrong tokenizer identity')
    checked_file(tokenizer_dir, 'tokenizer.pkl', TOKENIZER)
    sources = spec.get('source_sha256', {})
    if set(sources) != set(SOURCE_NAMES):
        raise ValueError('Incomplete source identity')
    for name in SOURCE_NAMES:
        checked_file(Path(__file__).parent, name, sources[name])
    weights = checked_file(root, spec['weights_file'], spec['checkpoint_sha256'])
    if spec.get('kind') == 'base':
        if spec['checkpoint_sha256'] != BASE_WEIGHTS:
            raise ValueError('Not the pinned base weights')
        metadata = checked_file(root, spec['metadata_file'], BASE_METADATA)
        if json.loads(metadata.read_text())['model_config'] != MODEL_CONFIG:
            raise ValueError('Base metadata configuration mismatch')
    elif spec.get('kind') == 'sft':
        # All optimizer shards must exist and hash correctly, but are NEVER
        # deserialized for evaluation. A complete checkpoint, not weights alone.
        complete_path = checked_file(root, spec['complete_file'], spec['complete_sha256'])
        folder = complete_path.parent
        complete = json.loads(complete_path.read_text())
        if 'loop.json' not in complete or 'model.pt' not in complete:
            raise ValueError('Incomplete SFT checkpoint')
        loop_path = checked_file(folder, 'loop.json', complete['loop.json']['sha256'])
        loop = json.loads(loop_path.read_text())
        identity = loop.get('identity', {})
        if (identity.get('base_weights_sha256') != BASE_WEIGHTS or
                identity.get('tokenizer_sha256') != TOKENIZER or
                identity.get('base_metadata_sha256') != BASE_METADATA):
            raise ValueError('SFT training lineage mismatch')
        for name in SOURCE_NAMES[:5]:
            if identity.get('code', {}).get('nanochat/' + name) != sources[name]:
                raise ValueError('SFT training source mismatch: ' + name)
        world = loop['world_size']
        if type(world) is not int or world < 1:
            raise ValueError('Invalid SFT world size')
        names = {'model.pt', 'loop.json'} | {f'rank-{r:04d}.pt' for r in range(world)}
        if set(complete) != names:
            raise ValueError('Missing or extra SFT shards')
        for name, entry in complete.items():
            file = checked_file(folder, name, entry['sha256'])
            if file.stat().st_size != entry['bytes']:
                raise ValueError('SFT shard size mismatch')
        if weights != (folder/'model.pt').resolve() or complete['model.pt']['sha256'] != spec['checkpoint_sha256']:
            raise ValueError('SFT manifest points to different weights')
    else:
        raise ValueError('Unknown checkpoint kind')
    return spec, weights


def strict_model(state, config):
    """Native architecture, no key patching/default-initialized substitutions."""
    import torch
    from nanochat.gpt import GPT, GPTConfig
    with torch.device('meta'):
        model = GPT(GPTConfig(**config))
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError('Checkpoint state keys mismatch')
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError('Checkpoint tensor shape mismatch: ' + name)
        if value.device.type != 'cpu' or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('Invalid/nonfinite checkpoint tensor: ' + name)
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False)
    # Nonpersistent RoPE buffers constructed on meta are absent from state_dict.
    # Rebuild these, NEVER call init_weights() on checkpoint parameters.
    model.cos, model.sin = model._precompute_rotary_embeddings(
        model.rotary_seq_len, config['n_embd'] // config['n_head'], device='cpu')
    model.eval()
    return model


@dataclass
class VerifiedEvaluator:
    backend: object
    codec: object
    identity: dict

    def run_development(self, suite_dir, *, output_path, ledger_path):
        """Explicit call only; acceptance is deliberately not exposed here."""
        from nanochat.conversation_eval import run_frozen
        if self.backend.identity != self.identity:
            raise ValueError('Verified backend identity was changed')
        return run_frozen(suite_dir, 'development', self.codec, self.backend,
                          self.identity, output_path=output_path, ledger_path=ledger_path)


def load_verified(root, manifest_path, *, trusted_manifest_sha256, tokenizer_dir, device='cpu'):
    # No implicit GPU choice or remote downloads, including kernel downloads.
    if device != 'cpu':
        raise PermissionError('This integration is CPU-only; GPU requires a separate approved smoke test')
    spec, weights = verify_manifest(root, manifest_path, trusted_manifest_sha256, tokenizer_dir)
    import torch
    if torch.cuda.is_available():
        raise PermissionError('Hide GPUs with CUDA_VISIBLE_DEVICES="" before starting this CPU-only process')
    from nanochat.common import COMPUTE_DTYPE
    if COMPUTE_DTYPE != torch.float32:
        raise ValueError('CPU integration requires NANOCHAT_DTYPE=float32 before import')
    from nanochat.conversation_eval import ChatCodec, NativeGreedyBackend
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    model = strict_model(state, spec['model_config'])
    codec = ChatCodec.from_directory(tokenizer_dir)
    if codec.tokenizer.get_vocab_size() != spec['model_config']['vocab_size']:
        raise ValueError('Tokenizer/model vocabulary mismatch')
    identity = {'kind': 'model', 'checkpoint_kind': spec['kind'],
                'checkpoint_sha256': spec['checkpoint_sha256'], 'tokenizer_sha256': TOKENIZER,
                'checkpoint_manifest_sha256': trusted_manifest_sha256,
                'source_sha256': spec['source_sha256'], 'device': 'cpu',
                'compute_dtype': 'float32', 'kernel_path': 'native_full_prefix_sdpa',
                'base_artifact': 'wandb/nanochat-demo/b1384b1-checkpoint:v2'}
    backend = NativeGreedyBackend(model)
    backend.identity = dict(identity)
    return VerifiedEvaluator(backend, codec, identity)