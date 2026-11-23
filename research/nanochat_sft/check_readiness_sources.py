"""Read upstream dataset license evidence and protected holdout manifest only."""
import hashlib,json
from pathlib import Path
import requests,wandb

root=Path(__file__).resolve().parent
out=root/'readiness_review_v1'
out.mkdir(exist_ok=True)
result={}
for repo in ('HuggingFaceTB/everyday-conversations-llama3.1-2k','meta-math/MetaMathQA'):
    key=repo.replace('/','--')
    try:
        response=requests.get('https://huggingface.co/api/datasets/'+repo,timeout=20)
        response.raise_for_status()
        data=response.json()
        revision=data['sha']
        (out/(key+'-metadata.json')).write_text(json.dumps(data,indent=2))
        url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md'
        card=requests.get(url,timeout=20)
        card.raise_for_status()
        (out/(key+'-README.md')).write_bytes(card.content)
        license_files=[x['rfilename'] for x in data.get('siblings',[]) if Path(x['rfilename']).name.lower() in ('license','license.md','license.txt','notice','notice.txt')]
        extra={}
        for name in license_files[:4]:
            r=requests.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}',timeout=20)
            r.raise_for_status()
            (out/(key+'-'+Path(name).name)).write_bytes(r.content)
            extra[name]=hashlib.sha256(r.content).hexdigest()
        result[repo]={'revision':revision,'license':data.get('cardData',{}).get('license'),
                      'readme_url':url,'readme_sha256':hashlib.sha256(card.content).hexdigest(),
                      'license_files':extra,'status':'upstream_evidence_retrieved_not_legal_approval'}
    except Exception as error:
        result[repo]={'status':'unresolved','error':str(error)[:400]}
artifact=wandb.Api().artifact('wandb/nanochat-demo/nanochat-untouched-holdout-20261117-v1:v0')
manifest=Path(artifact.get_path('holdout-manifest.json').download(root=str(out)))
result['protected_holdout']={'ref':'wandb/nanochat-demo/'+artifact.name,'digest':artifact.digest,
                            'manifest':json.loads(manifest.read_text()),
                            'payloads_opened':False,'per_example_fingerprint_files':[
                                e.path for e in artifact.manifest.entries.values()
                                if 'fingerprint' in e.path.lower() or 'index' in e.path.lower()]}
(out/'source-evidence.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)