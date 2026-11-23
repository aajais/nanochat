"""Resolve the public image manifest without writing or printing registry tokens."""
import hashlib
import json
from pathlib import Path
import requests

repo = 'wandb/autoresearch-base'
session = requests.Session()
auth = session.get('https://auth.docker.io/token',
                   params={'service':'registry.docker.io','scope':f'repository:{repo}:pull'}, timeout=25)
auth.raise_for_status()
headers = {'Authorization':'Bearer '+auth.json()['token'],
           'Accept':', '.join(['application/vnd.oci.image.index.v1+json',
                               'application/vnd.docker.distribution.manifest.list.v2+json',
                               'application/vnd.oci.image.manifest.v1+json',
                               'application/vnd.docker.distribution.manifest.v2+json'])}
def fetch(ref):
    response = session.get(f'https://registry-1.docker.io/v2/{repo}/manifests/{ref}', headers=headers, timeout=25)
    response.raise_for_status()
    digest = 'sha256:'+hashlib.sha256(response.content).hexdigest()
    assert response.headers['Docker-Content-Digest'] == digest
    return response.json(), digest
manifest, index_digest = fetch('latest')
if 'manifests' in manifest:
    matches = [m for m in manifest['manifests'] if m.get('platform',{}).get('architecture')=='amd64'
               and m.get('platform',{}).get('os')=='linux']
    assert len(matches)==1
    manifest, digest = fetch(matches[0]['digest'])
else:
    digest = index_digest
receipt = dict(image=f'{repo}@{digest}', image_manifest_sha256=digest,
               source_tag='latest', resolved_index_digest=index_digest,
               platform='linux/amd64', manifest=manifest)
Path('/workspace/nanochat_sft/pinned_training_image.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='manifest'}), flush=True)