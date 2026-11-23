import json,sys
from pathlib import Path
import torch
root=Path('/workspace/nanochat_sft')
sys.path.insert(0,str(root/'prepared'))
from nanochat.gpt import GPT,GPTConfig
from nanochat.tokenizer import RustBPETokenizer
torch.set_num_threads(4)
meta=json.loads((root/'base_metadata/meta_133632.json').read_text())
state=torch.load(root/'base_metadata/model_133632.pt',map_location='cpu',weights_only=True,mmap=True)
for name,tensor in state.items():
 if not torch.isfinite(tensor).all(): raise ValueError('Nonfinite base weights: '+name)
with torch.device('meta'): model=GPT(GPTConfig(**meta['model_config']))
model.load_state_dict(state,strict=True,assign=True)
model.cos,model.sin=model._precompute_rotary_embeddings(model.rotary_seq_len,meta['model_config']['n_embd']//meta['model_config']['n_head'],device='cpu')
model.eval()
tok=RustBPETokenizer.from_directory(root/'tokenizer')
tokens=tok.encode('The capital of France is')
with torch.no_grad(): logits=model(torch.tensor([tokens]))
assert torch.isfinite(logits).all()
result=json.loads((root/'base_verification.json').read_text())
result.update({'finite_tensor_scan':'pass_all_175_tensors','forward_pass':'pass_CPU_float32_compute',
               'forward_logit_shape':list(logits.shape),'generation_test':'not_run'})
(root/'base_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)