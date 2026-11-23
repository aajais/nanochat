import json,hashlib,sys
from pathlib import Path
import wandb
import torch

root=Path('/workspace/nanochat_sft')
sys.path.insert(0,str(root/'prepared'))
from nanochat.gpt import GPT,GPTConfig
from nanochat.tokenizer import RustBPETokenizer

api=wandb.Api()
artifact=api.artifact('wandb/nanochat-demo/b1384b1-checkpoint:v2')
print('Downloading only the fixed base weights for CPU structural verification (no training).',flush=True)
path=artifact.get_path('model_133632.pt').download(root=str(root/'base_metadata'))
meta=json.loads((root/'base_metadata/meta_133632.json').read_text())
state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
with torch.device('meta'):
 model=GPT(GPTConfig(**meta['model_config']))
expected=model.state_dict()
missing=sorted(set(expected)-set(state));extra=sorted(set(state)-set(expected))
mismatch={k:{'expected':list(expected[k].shape),'actual':list(state[k].shape)}
          for k in set(state)&set(expected) if expected[k].shape!=state[k].shape}
if missing or extra or mismatch:raise ValueError({'missing':missing,'extra':extra,'shape_mismatch':mismatch})
model.load_state_dict(state,strict=True,assign=True)
tokenizer=RustBPETokenizer.from_directory(root/'tokenizer')
assert tokenizer.get_vocab_size()==meta['model_config']['vocab_size']
from nanochat.sft_runtime import sha256
result={'artifact':'wandb/nanochat-demo/b1384b1-checkpoint:v2','artifact_digest':artifact.digest,
        'weights_sha256':sha256(path),'tensor_keys':len(state),
        'parameters':sum(p.numel() for p in model.parameters()),
        'model_config':meta['model_config'],'tokenizer_vocab':tokenizer.get_vocab_size(),
        'tokenizer_sha256':sha256(root/'tokenizer/tokenizer.pkl'),
        'strict_state_assignment':'pass','finite_tensor_scan':'not_run',
        'forward_pass':'not_run','gpu_test':'not_run'}
(root/'base_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)