"""Verified native interactive chat. No tools, downloads or hosted fallback."""
import argparse
import json
from pathlib import Path
import torch
from nanochat.conversation_eval import ChatCodec,NativeGreedyBackend
from nanochat.conversation_eval_checkpoint import strict_model,MODEL_CONFIG
from nanochat.sft_runtime import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    manifest=json.loads((a.bundle/'MODEL_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if sha256(a.bundle/name)!=digest:raise ValueError('Package drift: '+name)
    print('INTERNAL RESEARCH CANDIDATE: human review and release approval are pending.')
    model=strict_model(torch.load(a.bundle/'model/model.pt',map_location='cpu',weights_only=True,mmap=True),MODEL_CONFIG).to(a.device).eval()
    codec=ChatCodec.from_directory(a.bundle/'tokenizer');backend=NativeGreedyBackend(model);history=[]
    while True:
        try:user=input('User> ')
        except (EOFError,KeyboardInterrupt):break
        if not user.strip():continue
        prompt,kept=codec.prompt(history,user)
        ids=[];stopped=False
        for token in backend.stream(prompt,max_new_tokens=256,temperature=0):
            ids.append(int(token))
            if token==codec.end:stopped=True;break
            if token in codec.controls:raise RuntimeError('Model invented control/other speaker; response rejected')
            if len(ids)>=256:break
        text=codec.tokenizer.decode(ids[:-1] if stopped else ids)
        print('Assistant> '+text)
        if not stopped:
            print('[Response hit bound without end-of-turn; not added to history.]');history=kept
        else:history=kept+[{'turn':len(kept),'user':user,'assistant_ids':ids}]

if __name__=='__main__':main()