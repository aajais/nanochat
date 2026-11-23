"""Bounded GPU smoke supervisor: fail closed; never launches main SFT."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

START = time.monotonic()
def alarm_handler(*_):
    raise TimeoutError('Smoke deadline reached; training is not approved by this result')
signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(870)  # Kubernetes independently caps this whole Job at 900 seconds.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NANOCHAT_DTYPE'] = 'bfloat16'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'

import wandb

ROOT = Path('smoke_work')
ROOT.mkdir(exist_ok=False)
run = None

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def budget(reserve=150):
    remaining = 850-(time.monotonic()-START)-reserve
    if remaining <= 0:
        raise TimeoutError('No smoke budget remains after upload/exit reserve')
    return remaining

def child(argv, seconds):
    subprocess.run(argv, check=True, timeout=min(seconds,budget()))

def main():
    global run
    package = json.loads(Path('SMOKE_PACKAGE.json').read_text())
    for path, expected in package['source_files_sha256'].items():
        assert sha256(path) == expected, f'Code hash mismatch: {path}'
    run = wandb.init(entity='wandb',project='nanochat-demo',job_type='gpu-readiness-smoke',
                     name='native-base-one-gpu-save-restore-smoke',config={
                         'purpose':'plumbing_only_not_model_quality','gpu_count':1,
                         'hard_job_deadline_seconds':900,'automatic_retry':False,
                         'main_training_approved_by_this_job':False,
                         'container_image':package['image'],'fixture':'synthetic-probes-v1',
                         'context':2048,'microbatch':1,'gradient_accumulation':8,
                         'fp8':False,'dtype':'bfloat16','torch_compile':False})
    run.log({'smoke/started':1,'smoke/elapsed_seconds':time.monotonic()-START})
    # Failure status is explicitly propagated; no worker exception is ignored.
    failed = subprocess.run([sys.executable,'-c','raise SystemExit(17)'],timeout=10)
    assert failed.returncode == 17
    # Test deadline signalling in an isolated CPU child, without consuming model RNG.
    deadline_test = '''
import json,time
from nanochat.sft_runtime import Deadline
d=Deadline(time.time()+1,0)
assert not d.should_stop('cpu')
time.sleep(1.1)
assert d.should_stop('cpu')
d=Deadline(time.time()+100,0)
d._signal(None,None)
assert d.should_stop('cpu')
print('deadline and graceful signal checks passed',flush=True)
'''
    child([sys.executable,'-c',deadline_test],30)
    expected_artifacts = [
        ('wandb/nanochat-demo/b1384b1-checkpoint:v2','c07a4b33a3081fab517e5265cc8a79ed','base',
         {'model_133632.pt':'c76eb3a6e3c482647e7d0f0f7077689369d060593762803678e2d1d4fcbbdbf0',
          'meta_133632.json':'7ef3096f4dbc72b4a7a7f879aed0a0f69abc810317046183349741f8b9decaec'}),
        ('wandb/nanochat-demo/pr1-smoke-tokenizer:v0',None,'tokenizer',
         {'tokenizer.pkl':'a228803c68627f650e640c819e71ea0b90f49a96118858ed781a00a5ba63b6bf'})]
    identities = []
    for ref, expected_digest, destination, files in expected_artifacts:
        budget()
        artifact = run.use_artifact(ref)
        if expected_digest is not None: assert artifact.digest == expected_digest
        for name, expected_hash in files.items():
            path = artifact.get_path(name).download(root=str(ROOT/destination))
            assert sha256(path) == expected_hash, f'Artifact hash mismatch: {name}'
        identities.append({'ref':ref,'digest':artifact.digest,'file_sha256':files})
    (ROOT/'identity.json').write_text(json.dumps({'artifacts':identities,'code':package,
                                               'fixture':'synthetic-probes-v1','world_size':1},indent=2))
    run.log({'smoke/assets_verified':1,'smoke/elapsed_seconds':time.monotonic()-START})
    child([sys.executable,'gpu_smoke_worker.py','save'],360)
    run.log({'smoke/save_phase_completed':1,'smoke/elapsed_seconds':time.monotonic()-START})
    child([sys.executable,'gpu_smoke_worker.py','restore'],300)
    result = json.loads((ROOT/'worker_result.json').read_text())
    assert result['passed']
    checkpoint = Path((ROOT/'checkpoint_path.txt').read_text())
    # Upload the complete checkpoint, including native optimizer/RNG/loader shards.
    # A successful replay alone is not evidence that the upload completed.
    budget(reserve=15)
    output = wandb.Artifact(f'{run.id}-verified-native-smoke-checkpoint',type='sft-smoke-checkpoint',
                            metadata={'plumbing_only':True,'identity':identities,
                                      'exact_replay':result['exact_replay'],
                                      'not_a_selected_chat_model':True})
    output.add_dir(str(checkpoint),name='checkpoint')
    for path in ROOT.glob('*.json'): output.add_file(str(path),name='evidence/'+path.name)
    output.add_file('SMOKE_PACKAGE.json',name='evidence/SMOKE_PACKAGE.json')
    run.log_artifact(output).wait()
    receipt = dict(result,checkpoint_artifact=f'wandb/nanochat-demo/{output.name}',
                   checkpoint_digest=output.digest,checkpoint_upload_completed=True,
                   failure_exit_test=True,deadline_unit_test=True,
                   elapsed_seconds=time.monotonic()-START,
                   main_sft_not_started=True,data_release_still_provisional=True)
    (ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2))
    run.save(str(ROOT/'receipt.json'),base_path=str(ROOT),policy='now')
    run.log({'smoke/passed':1,'smoke/exact_restore':1,'smoke/checkpoint_upload_completed':1,
             'smoke/elapsed_seconds':time.monotonic()-START})
    run.summary.update(receipt)
    run.finish(exit_code=0)
    print(json.dumps(receipt),flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        failure = {'passed':False,'error':str(error),'error_type':type(error).__name__,
                   'elapsed_seconds':time.monotonic()-START,'main_sft_not_started':True}
        (ROOT/'failure.json').write_text(json.dumps(failure,indent=2))
        if run is not None:
            try:
                run.log({'smoke/passed':0})
                run.save(str(ROOT/'failure.json'),base_path=str(ROOT),policy='now')
                run.finish(exit_code=1)
            except BaseException:
                pass
        raise SystemExit(1)