"""Read final server identities/status; no training, no object mutation."""
import json
from pathlib import Path
import wandb
ROOT=Path(__file__).resolve().parent
path=ROOT/'published_readiness_resolution.json'
receipt=json.loads(path.read_text());api=wandb.Api()
for name,record in receipt['artifacts'].items():
    server=api.artifact(record['ref'])
    assert server.digest==record['digest']
    assert server.metadata['launch_ready'] is False
    assert len(server.manifest.entries)==record['files']
    if name=='conversation-sft-cache-candidate':
        local=server.get_path('manifest.json').download(root=str(ROOT/'readiness_final_server_metadata'))
        manifest=json.loads(Path(local).read_text())
        assert manifest['release_gate']['training_approved'] is False
        assert manifest['statistics']['train']['examples']==16184
        assert manifest['statistics']['dev']['examples']==834
        assert manifest['conversation_evaluation_artifact']==receipt['exact_fixtures']
run=api.run('wandb/nanochat-demo/'+receipt['run_id'])
assert run.state=='finished' and run.config['launch_ready'] is False
receipt['run_state']=run.state
receipt['run_summary_verified']=bool(dict(run.summary))
receipt['summary_note']='Summary did not persist after SDK write and one explicit update; artifact evidence and config are authoritative. No further retries.'
receipt['publication_status']='all four artifact versions independently read back; all file bytes previously verified by publisher'
path.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)