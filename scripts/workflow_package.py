"""Package research output; never equate missing human review with release."""
import argparse
import json
from pathlib import Path
import wandb
from nanochat.sft_runtime import atomic_json,sha256
from nanochat.workflow_policy import verify_package

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();work=a.work
    evaluation=json.loads((work/'evaluation/result.json').read_text())
    approval=json.loads((work/'approval.json').read_text())
    selection=evaluation['selected'];root=Path(__file__).resolve().parents[1]
    package,package_sha=verify_package(root)
    output=work/'packaging';output.mkdir(exist_ok=False)
    notice=approval['decisions']['license_notice_disposition'].get('notice_bundle')
    if not isinstance(notice,dict) or not notice:raise PermissionError('Accepted authoritative notice bundle missing')
    files={}
    def add(source,target):
        if target in files:raise ValueError('Duplicate package target')
        files[target]=Path(source)
    weights=Path(selection['weights']);add(weights,'model/model.pt')
    if selection['kind']=='sft':
        for path in weights.parent.iterdir():
            if path.is_file() and path.name!='model.pt':add(path,'model/'+path.name)
    else:add(work/'assets/base/meta_133632.json','model/base-meta.json')
    for name in ('tokenizer.pkl','token_bytes.pt'):add(work/'assets/tokenizer'/name,'tokenizer/'+name)
    for name in package['source_sha256']:add(root/name,'source/'+name)
    add(root/'WORKFLOW.json','source/WORKFLOW.json')
    for name,text in notice.items():
        target=Path(name)
        if target.is_absolute() or '..' in target.parts or not isinstance(text,str) or not text.strip():raise ValueError('Malformed approved notice file')
        local=output/'notices'/target;local.parent.mkdir(parents=True,exist_ok=True);local.write_text(text)
        add(local,'notices/'+target.as_posix())
    card=output/'MODEL_CARD.md'
    card.write_text('# Nanochat internal-demo research candidate — NOT RELEASE APPROVED\n\n'
        'Partial pretraining; base 1,384,122,122 parameters, context 2,048. Internal use only.\n'
        f"Selected kind: {selection['kind']}; weights SHA256: {selection['checkpoint_sha256']}.\n"
        'Selection used the explicitly approved mechanical development rule, not human usefulness.\n'
        f"Automated acceptance state: {evaluation['gate_outcome']}. Human review remains pending.\n"
        'ARC100 is a small historical retention check, not broad benchmark evidence.\n'
        'Inspect evaluation evidence before any demonstration. Do not substitute or relabel a fallback.\n'
        'Run: PYTHONPATH=source python -m scripts.packaged_chat --bundle .\n'
        'Runtime enforces complete-turn history eviction, 2,048 tokens, 256-token bound and actual EOT.\n')
    add(card,'MODEL_CARD.md');add(work/'approval.json','approval.json')
    for path in (work/'evaluation').iterdir():
        if path.is_file():add(path,'evaluation/'+path.name)
    manifest=dict(release_approved=False,internal_demo_only=True,selection=selection,workflow_sha256=package_sha,
        files={target:sha256(source) for target,source in files.items()})
    atomic_json(output/'MODEL_MANIFEST.json',manifest);add(output/'MODEL_MANIFEST.json','MODEL_MANIFEST.json')
    run=wandb.init(entity='wandb',project='nanochat-demo',job_type='sft-research-package',config={'release_approved':False,'workflow_sha256':package_sha})
    try:
        for pin in package['assets'].values():run.use_artifact(pin['ref'])
        main=json.loads((work/'main/result.json').read_text())
        for receipt in main['uploads']:run.use_artifact(receipt['ref'])
        artifact=wandb.Artifact(run.id+'-nanochat-research-result',type='model',metadata={'release_approved':False,'human_review_pending':True,'selected_weights_sha256':selection['checkpoint_sha256']})
        for target,source in files.items():artifact.add_file(str(source),name=target)
        logged=run.log_artifact(artifact).wait();ref='wandb/nanochat-demo/'+logged.name
        server=wandb.Api().artifact(ref)
        if server.digest!=logged.digest or set(server.manifest.entries)!=set(files):raise ValueError('Package server metadata mismatch')
        downloaded=Path(server.download(root=str(output/'server-verified')))
        for target,source in files.items():
            if sha256(downloaded/target)!=sha256(source):raise ValueError('Package server byte mismatch')
        atomic_json(output/'result.json',dict(completed=True,ref=ref,digest=server.digest,server_bytes_verified=True,release_approved=False))
        run.log({'packaging/server_verified':1});run.finish()
    except BaseException:
        run.finish(exit_code=1);raise

if __name__=='__main__':main()