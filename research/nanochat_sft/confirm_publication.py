import json
from pathlib import Path
import wandb
root=Path('/workspace/nanochat_sft')
api=wandb.Api()
run=api.run('wandb/nanochat-demo/eytpir2u')
artifact=api.artifact('wandb/nanochat-demo/conversation-sft-preparation:candidate-not-launch-ready')
assert run.state=='finished'
assert artifact.metadata['launch_ready'] is False
expected=json.loads((root/'bundle_sha256.json').read_text())
assert set(artifact.manifest.entries)==set(expected)|{'evidence/bundle_sha256.json'}
assert artifact.digest=='6799440af8cb33a9d4d3225115ccc83e'
result={'run_url':run.url,'artifact':f'wandb/nanochat-demo/conversation-sft-preparation:{artifact.version}',
        'artifact_url':f'https://forge.coreweave.com/wandb/wandb/nanochat-demo/artifacts/code/conversation-sft-preparation/{artifact.version}',
        'digest':artifact.digest,'file_count':len(artifact.manifest.entries),
        'verified_from_server':True,'launch_ready':False,'gpu_jobs_launched':0}
(root/'published_preparation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))