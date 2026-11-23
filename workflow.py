"""One authorized allocation: production readiness -> SFT -> evaluation -> package.

No retries, no shell indirection, no external datasets. Importing never launches.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

from nanochat.sft_runtime import atomic_json,sha256,validate_checkpoint
from nanochat.workflow_policy import Allocation,authorize,verify_package,TRAIN_SAVE

ROOT=Path(__file__).resolve().parent
CHILDREN=set()

def terminate_workers(process,folder):
    progress=Path(folder)/'progress.json'
    if progress.exists():
        pids=json.loads(progress.read_text()).get('worker_pids',[])
        if pids:
            for pid in pids:
                try:os.kill(int(pid),signal.SIGTERM)
                except ProcessLookupError:pass
            return
    os.killpg(process.pid,signal.SIGTERM)

def child(argv,end,log,*,signal_at_step=False,expect_failure=False):
    """Supervise actual children; never turn a nonzero exit into success."""
    env=dict(os.environ,WANDB_RUN_ID=uuid.uuid4().hex[:8],WANDB_RESUME='never',
             OMP_NUM_THREADS='1',TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
        CHILDREN.add(process.pid)
        sent=False
        try:
            while process.poll() is None:
                if time.time()>=end:raise TimeoutError('Stage hard deadline: '+log.name)
                if signal_at_step and not sent:
                    progress=log.parent/'progress.json'
                    if progress.exists() and json.loads(progress.read_text())['step']>=1:
                        # Single production worker or torchrun receives a real termination signal.
                        terminate_workers(process,log.parent);sent=True
                time.sleep(.05)
            if expect_failure:
                if process.returncode==0:raise AssertionError('Intentional worker failure was swallowed')
            elif process.returncode!=0:
                raise subprocess.CalledProcessError(process.returncode,argv)
            return dict(exit_code=process.returncode,signal_sent=sent)
        finally:
            if process.poll() is None:
                terminate_workers(process,log.parent)
                try:process.wait(timeout=max(.01,min(600,end-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            CHILDREN.discard(process.pid)

def nested_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    if hasattr(a,'shape'):
        import numpy as np
        return bool(np.array_equal(a,b))
    return a==b

def compare_replay(first,second):
    import torch
    first,second=Path(first),Path(second)
    x=json.loads((first/'loop.json').read_text());y=json.loads((second/'loop.json').read_text())
    if x!=y:raise AssertionError('Schedule/config/counters/loader horizon replay differs')
    for name in json.loads((first/'COMPLETE.json').read_text()):
        if name.endswith('.pt'):
            a=torch.load(first/name,map_location='cpu',weights_only=False,mmap=True)
            b=torch.load(second/name,map_location='cpu',weights_only=False,mmap=True)
            if not nested_equal(a,b):raise AssertionError('Replay mismatch: '+name)

def verify_result(folder,expected_reason=None,expected_world=None):
    folder=Path(folder)
    result=json.loads((folder/'result.json').read_text())
    if result['passed'] is not True:raise ValueError('Worker did not complete')
    if expected_world is not None and result['world_size']!=expected_world:raise ValueError('Readiness/main world size mismatch')
    if expected_reason and result['loop']['stop_reason']!=expected_reason:raise ValueError('Wrong shutdown cause')
    step=result['loop']['step']; checkpoint=folder/f'step-{step:08d}'
    validate_checkpoint(checkpoint,result['identity'],result['world_size'])
    if not result['uploads'] or any(r.get('server_verified') is not True for r in result['uploads']):
        raise ValueError('Actual server checkpoint verification missing')
    return result,checkpoint

def verify_partitions(checkpoint,world):
    import torch
    for rank in range(world):
        payload=torch.load(Path(checkpoint)/f'rank-{rank:04d}.pt',map_location='cpu',weights_only=False,mmap=True)
        layout=payload.get('optimizer_partition') or {}
        if layout.get('rank')!=rank or layout.get('world_size')!=world or layout.get('validated') is not True or layout.get('optimizer_class')!='MuonAdamW':
            raise ValueError('Missing native rank-specific optimizer-sharding proof')

def training_command(work,assets,approval,allocation,stage,*,end=None,gpus=1):
    prefix=[sys.executable,'-m','scripts.chat_sft_resumable'] if gpus==1 else [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={gpus}','-m','scripts.chat_sft_resumable']
    return prefix+['--data-dir',str(assets/'candidate'),'--dataset-artifact','wandb/nanochat-demo/conversation-sft-cache-candidate:v3',
        '--tokenizer-dir',str(assets/'tokenizer'),'--base-dir',str(assets/'base'),'--output-dir',str(work),
        '--deadline-unix',str(end or allocation.end_for(stage)),'--allocation-started-unix',str(allocation.started),
        '--approval-file',str(approval),'--stage',stage,'--save-reserve-seconds',str(150 if stage=='readiness' else TRAIN_SAVE),
        '--max-updates','100000','--no-compile','--load-optimizer','0']

def readiness(work,assets,approval,budget,package_sha,gpus):
    end=budget.end_for('readiness')-15
    base=training_command(work,assets,approval,budget,'readiness',end=end,gpus=gpus)
    base+=['--fixed-horizon','16','--save-every-seconds','999999','--checkpoint-steps','11','--eval-every','0']
    def cmd(folder,extra=()):
        x=list(base);x[x.index('--output-dir')+1]=str(folder)
        return x+list(extra)
    a,b,c,d,f=[work/n for n in ('continuous','restored','sigterm','deadline','failure')]
    child(cmd(a,['--stop-after-step','12']),end,a/'worker.log')
    ar,ap=verify_result(a,'readiness_stop',gpus)
    validate_checkpoint(a/'step-00000011',ar['identity'],gpus)
    verify_partitions(a/'step-00000011',gpus)
    child(cmd(b,['--resume',str(a/'step-00000011'),'--stop-after-step','12']),end,b/'worker.log')
    br,bp=verify_result(b,'readiness_stop',gpus);verify_partitions(bp,gpus)
    compare_replay(ap,bp)
    signalcmd=cmd(c,['--pause-after-step','1'])
    signalcmd[signalcmd.index('--save-every-seconds')+1]='999999'
    sig=child(signalcmd,end,c/'worker.log',signal_at_step=True)
    _,cp=verify_result(c,'signal',gpus);verify_partitions(cp,gpus)
    deadlinecmd=cmd(d,['--pause-after-step','1'])
    deadlinecmd[deadlinecmd.index('--save-every-seconds')+1]='999999'
    # Use the same production Deadline path. Leave upload time inside readiness.
    deadlinecmd[deadlinecmd.index('--deadline-unix')+1]=str(min(end,time.time()+180))
    deadlinecmd[deadlinecmd.index('--save-reserve-seconds')+1]='150'
    child(deadlinecmd,end,d/'worker.log');_,dp=verify_result(d,'deadline',gpus);verify_partitions(dp,gpus)
    failure=child(cmd(f,['--inject-failure-step','1','--inject-failure-rank',str(gpus-1)]),end,f/'worker.log',expect_failure=True)
    if not (f/f'failure.rank-{gpus-1:04d}.json').is_file():raise ValueError('Missing nonzero-rank failure evidence')
    if time.time()>=end:raise TimeoutError('Readiness did not complete within cap')
    report=dict(passed=True,workflow_sha256=package_sha,allocation_started_unix=budget.started,
        completed_unix=time.time(),production_entrypoint=True,native_optimizer=True,
        finite_bf16_update=ar['compute_dtype']=='torch.bfloat16',periodic_complete_checkpoint=True,
        fresh_process_exact_replay=True,real_sigterm=sig['signal_sent'],real_deadline=True,
        actual_server_uploads_verified=True,nonzero_worker_failure_propagated=failure['exit_code']!=0,
        first_result=ar,restored_result=br,world_size=gpus,same_world_size_verified=gpus==8,
        optimizer_sharding_verified=gpus==8,shutdown_world_size=gpus)
    if not report['finite_bf16_update'] or not report['real_sigterm']:raise ValueError('Incomplete native GPU proof')
    atomic_json(work/'readiness.json',report)
    return work/'readiness.json'

def stage_assets(run,package,destination):
    for key,pin in package['assets'].items():
        artifact=run.use_artifact(pin['ref'])
        if artifact.digest!=pin['digest']:raise ValueError('Artifact identity drift: '+key)
        target=destination/key
        if key=='base':
            target.mkdir(parents=True)
            for name in ('model_133632.pt','meta_133632.json'):artifact.get_path(name).download(root=str(target))
        else:
            artifact.download(root=str(target));artifact.verify(root=str(target))
        for name,digest in pin.get('sha256',{}).items():
            if sha256(target/name)!=digest:raise ValueError('Asset file mismatch: '+key+'/'+name)

def upload_evidence(run,work,name):
    import wandb
    artifact=wandb.Artifact(run.id+'-'+name,type='evaluation',metadata={'release_approved':False})
    for path in sorted(work.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.log','.md','.sqlite') and 'assets' not in path.parts:
            artifact.add_file(str(path),name=str(path.relative_to(work)))
    published=run.log_artifact(artifact).wait()
    server=wandb.Api().artifact('wandb/nanochat-demo/'+published.name)
    if server.digest!=published.digest:raise ValueError('Evidence publication not verified')
    return server

def prepare_training_plan(run,work,proof,budget,package_sha):
    from nanochat.distributed_readiness import token_plan
    plan=token_plan(json.loads(proof.read_text()),time.time(),budget.end_for('main')-10)
    plan.update(workflow_sha256=package_sha,allocation_started_unix=budget.started,
        readiness_sha256=sha256(proof),reported_before_handoff=True)
    atomic_json(work/'training-token-plan.json',plan)
    print(json.dumps({'before_main_training_token_plan':plan}),flush=True)
    run.log({'plan/expected_input_tokens':plan['expected_input_tokens'],
        'plan/expected_assistant_tokens':plan['expected_supervised_assistant_tokens'],
        'plan/measured_input_tokens_per_second':plan['input_tokens_per_second']})

def run_stages(run,work,approval,budget,package_sha,gpus):
    proof=readiness(work/'readiness',work/'assets',approval,budget,package_sha,gpus)
    prepare_training_plan(run,work,proof,budget,package_sha)
    upload_evidence(run,work,'verified-readiness-evidence')
    maincmd=training_command(work/'main',work/'assets',approval,budget,'main',gpus=gpus)
    maincmd+=['--readiness-evidence',str(proof),'--training-plan-file',str(work/'training-token-plan.json')]
    child(maincmd,budget.end_for('main')-10,work/'main'/'worker.log')
    verify_result(work/'main',expected_world=gpus)
    child([sys.executable,'-m','scripts.workflow_evaluate','--work',str(work),
           '--end',str(budget.end_for('evaluation')-10)],budget.end_for('evaluation'),work/'evaluation-worker.log')
    child([sys.executable,'-m','scripts.workflow_package','--work',str(work)],
          budget.end_for('packaging'),work/'packaging-worker.log')

def main():
    import wandb
    package,package_sha=verify_package(ROOT)
    # A run's existence is not launch permission. Submission must have checked
    # the same record before allocating a GPU; runtime validates it again.
    run=wandb.init(entity='wandb',project='nanochat-demo',job_type='consolidated-sft-workflow')
    work=ROOT/'workflow_work';work.mkdir(exist_ok=False)
    try:
        approval=work/'approval.json';atomic_json(approval,dict(run.config.get('approval',{})))
        authorize(approval,package_sha,package['assets'])
        budget=Allocation(run.config.get('allocation_started_unix',float('nan')))
        atomic_json(work/'allocation.json',budget.spec())
        # Last-resort bound survives Python exceptions, SDK stalls and cleanup.
        def hard_stop():
            for pid in list(CHILDREN):
                try:os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError:pass
            os._exit(124)
        watchdog=threading.Timer(max(0,budget.end-time.time()),hard_stop)
        watchdog.daemon=True;watchdog.start()
        def interrupted(*_):raise InterruptedError('Workflow interrupted; no handoff/retry')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        os.environ.update(NANOCHAT_DTYPE='bfloat16',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TORCHDYNAMO_DISABLE='1')
        import torch
        from nanochat.distributed_readiness import verify_hardware
        gpus=int(run.config.get('gpus',0))
        if gpus!=8:raise ValueError('Eight-GPU configuration required; one-GPU proposal superseded')
        verify_hardware()
        def expired(*_):raise TimeoutError('Shared allocation deadline; no retry')
        signal.signal(signal.SIGALRM,expired)
        signal.alarm(max(1,int(budget.end-time.time()-10)))
        stage_assets(run,package,work/'assets')
        run_stages(run,work,approval,budget,package_sha,gpus)
        run.log({'workflow/completed':1,'workflow/elapsed_seconds':time.time()-budget.started})
        upload_evidence(run,work,'complete-evidence')
        run.finish()
    except BaseException as exc:
        atomic_json(work/'workflow-failure.json',{'passed':False,'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),'main_handoff_on_failure':False})
        try:upload_evidence(run,work,'failure-evidence')
        except BaseException:pass  # Never turn upload failure into workflow success.
        run.finish(exit_code=1)
        raise
    finally:signal.alarm(0)

if __name__=='__main__':main()