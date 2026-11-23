"""Trace protected artifact identity through original captured producer source."""
import hashlib,json,base64
from pathlib import Path
import wandb
ROOT=Path(__file__).resolve().parent/'authorized_holdout_check_v1'
api=wandb.Api()
job=json.loads((ROOT/'producer-job/wandb-job.json').read_text())
source_id=job['source']['artifact'].removeprefix('wandb-artifact://_id/')
query='''query GetArtifact($id: ID!) { artifact(id: $id) {
artifactSequence { name project { name entityName } } aliases { alias }
} }'''
data=api._service_api.execute_graphql(query,{'id':source_id})['artifact']
sequence=data['artifactSequence']
versions=[a['alias'] for a in data['aliases'] if a['alias'].startswith('v') and a['alias'][1:].isdigit()]
assert len(versions)==1
source_ref=f"{sequence['project']['entityName']}/{sequence['project']['name']}/{sequence['name']}:{versions[0]}"
source=api.artifact(source_ref)
files=list(source.manifest.entries)
assert 'holdout_entry.py' in files
path=source.get_path('holdout_entry.py').download(root=str(ROOT/'producer-source'))
protected=api.artifact('wandb/nanochat-demo/nanochat-untouched-holdout-20261117-v1:v0')
file=ROOT/'protected-local-only/holdout.parquet'
actual=base64.b64encode(hashlib.md5(file.read_bytes()).digest()).decode()
assert protected.digest=='cd86aa08a615f820943f8e402f988573'
assert protected.manifest.entries['holdout.parquet'].digest==actual
evidence={'producer_source_ref':source_ref,'producer_source_digest':source.digest,'producer_entrypoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'parquet_bytes_match_immutable_artifact_manifest_md5':True,'files':files}
(ROOT/'origin-verification.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(evidence,indent=2))