"""Read back the submitted immutable job and validate its manifest contract."""
import ast
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,'/workspace/skills/wbagent/scripts')
import launch_helpers as launch
import wandb

ROOT=Path('/workspace/nanochat_sft')
corrected='--corrected' in sys.argv
receipt=json.loads((ROOT/('gpu_smoke_corrected_job.json' if corrected else 'gpu_smoke_submission.json')).read_text())
artifact=wandb.Api().artifact(receipt['job_ref'])
assert artifact.digest==receipt['job_artifact_digest']
destination=ROOT/'checks/gpu_smoke_job_server'
job=json.loads(Path(artifact.get_path('wandb-job.json').download(root=str(destination))).read_text())
source_id=job['source']['artifact'].removeprefix('wandb-artifact://_id/')
query='''query GetArtifact($id: ID!) { artifact(id: $id) {
artifactSequence { name project { name entityName } } aliases { alias }
} }'''
data=wandb.Api()._service_api.execute_graphql(query,{'id':source_id})['artifact']
sequence=data['artifactSequence']
versions=[a['alias'] for a in data['aliases'] if a['alias'].startswith('v') and a['alias'][1:].isdigit()]
assert len(versions)==1, versions
source_ref=f"{sequence['project']['entityName']}/{sequence['project']['name']}/{sequence['name']}:{versions[0]}"
source=wandb.Api().artifact(source_ref)
names=['SMOKE_PACKAGE.json','gpu_smoke_supervisor.py','gpu_smoke_worker.py']
files={name:Path(source.get_path(name).download(root=str(destination))).read_text() for name in names}
package=json.loads(files['SMOKE_PACKAGE.json'])
tree=ast.parse(files['gpu_smoke_supervisor.py'])
keys=sorted({n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript)
             and isinstance(n.value,ast.Name) and n.value.id=='package'
             and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)})
missing=sorted(set(keys)-set(package))
result={'job_ref':receipt['job_ref'],'job_artifact_digest_verified':True,
        'source_ref':source_ref,'source_digest':source.digest,
        'package_manifest_keys':sorted(package),'supervisor_manifest_keys':keys,
        'missing_keys':missing,'manifest_contract_passed':not missing,
        'server_base_image':job['source'].get('base_image'),
        'source_bytes_verified':all(hashlib.sha256(files[name].encode()).hexdigest()==package['source_files_sha256'][name]
                                    for name in ['gpu_smoke_supervisor.py','gpu_smoke_worker.py'])}
try:
    if corrected: raise RuntimeError('Corrected job published only; no queue item or GPU resubmission')
    item=launch._fetch_launch_queue_item('wandb','nanochat-demo-k8s',receipt['queue_item_id'])
    spec=item['runSpec']
    if isinstance(spec,str): spec=json.loads(spec)
    kube=spec['resource_args']['kubernetes']['spec']
    result['server_job_deadline_seconds']=kube['activeDeadlineSeconds']
    result['server_backoff_limit']=kube['backoffLimit']
    result['server_gpu_limit']=kube['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu']
    result['queue_state']=item['state']
    result['run_id']=item.get('associatedRunId')
except Exception as error:
    result['queue_spec_verification_error']=str(error)
missing_files=[]
bad_hashes=[]
for name,expected in package['source_files_sha256'].items():
    try:
        local=source.get_path(name).download(root=str(destination))
        if hashlib.sha256(Path(local).read_bytes()).hexdigest()!=expected: bad_hashes.append(name)
    except KeyError:
        missing_files.append(name)
result.update(missing_source_files=missing_files,bad_source_hashes=bad_hashes)
(ROOT/('gpu_smoke_corrected_job_server_verification.json' if corrected else 'gpu_smoke_server_verification.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)