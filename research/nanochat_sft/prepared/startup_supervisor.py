"""One allocation, three bounded transient startup attempts, no training restarts."""
import argparse,json,os,signal,subprocess,sys,time,threading
from pathlib import Path
from big_config import CONFIG

def retryable(text,timed_out):
    deterministic=('ValueError:','AssertionError','SyntaxError:','ModuleNotFoundError:',
                   'CUDA out of memory','Nonfinite','Captured source changed','hardware mismatch',
                   'Unexpected configuration','PermissionError','401 Unauthorized','403 Forbidden')
    if any(s in text for s in deterministic):return False
    return timed_out or any(s in text.lower() for s in (
        'connection reset','connection aborted','connectionerror','readtimeout',
        'timed out','temporarily unavailable','502 bad gateway','503 service unavailable',
        '504 gateway timeout','ncclremoteerror','ncclsystemerror'))

def stop(child):
    if child.poll() is None:
        child.send_signal(signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)

def supervise(command_factory,root,run,deadline,startup_seconds=900,max_attempts=3,poll=2,backoff=15):
    child=None
    try:
        for attempt in range(1,max_attempts+1):
            if time.time()>=deadline:raise RuntimeError('Aggregate wall budget exhausted')
            rid='b1384b'+str(attempt)
            marker=root/('training-started-'+str(attempt)+'.json')
            logfile=root/('startup-attempt-'+str(attempt)+'.log')
            if marker.exists() or logfile.exists():raise RuntimeError('Attempt receipt already exists; refuse replay')
            env=os.environ.copy()
            for key in ('WANDB_LAUNCH','WANDB_LAUNCH_CONFIG_PATH','WANDB_RUN_ID','WANDB_SWEEP_ID'):
                env.pop(key,None)
            env.update(BIGTRAIN_RUN_ID=rid,WANDB_RUN_ID=rid,
                       BIGTRAIN_STARTED_MARKER=str(marker.resolve()),
                       BIGTRAIN_PROVENANCE=json.dumps(dict(run.config).get('_wb_agent',{})))
            run.summary.update({'startup/attempt':attempt,'startup/child_run_id':rid,
                                'startup/maximum_attempts':max_attempts})
            run.log({'startup/attempt':attempt})
            began=time.time();timed_out=False;started=False
            with logfile.open('x') as out:
                child=subprocess.Popen(command_factory(attempt),stdout=out,stderr=subprocess.STDOUT,
                                       env=env,start_new_session=True)
                while child.poll() is None:
                    if marker.exists() and not started:
                        started=True
                        run.summary.update({'startup/training_started':True,
                                            'startup/first_step':json.loads(marker.read_text()),
                                            'startup/training_run_url':'https://forge.coreweave.com/wandb/wandb/nanochat-demo/runs/'+rid})
                        print('TRAINING_STARTED='+rid,flush=True)
                    if time.time()>=deadline:
                        stop(child);raise RuntimeError('Aggregate 24-hour wall budget exhausted')
                    if not started and time.time()-began>=startup_seconds:
                        timed_out=True;stop(child);break
                    time.sleep(poll)
            started=started or marker.exists()
            if started:
                run.summary.update({'startup/first_step':json.loads(marker.read_text()),
                                    'startup/training_run_url':'https://forge.coreweave.com/wandb/wandb/nanochat-demo/runs/'+rid})
            text=logfile.read_text(errors='replace')[-20000:]
            print(text[-5000:],flush=True)
            run.summary.update({'startup/last_exit_code':child.returncode,
                                'startup/last_timed_out':timed_out,
                                'startup/training_started':started})
            run.save(str(logfile),base_path=str(root),policy='now')
            if started:
                run.summary['startup/retries_disabled_after_training']=True
                return 0 if child.returncode==0 else 1
            if child.returncode==0:raise RuntimeError('Child finished without optimizer-step evidence')
            if attempt==max_attempts or not retryable(text,timed_out):
                raise RuntimeError('Startup stopped: retry limit or non-transient failure; inspect child logs')
            print('TRANSIENT_STARTUP_RETRY='+str(attempt+1),flush=True)
            time.sleep(min(backoff,max(0,deadline-time.time())))
    finally:
        if child is not None:stop(child)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submitted-at',type=float,required=True)
    args=parser.parse_args()
    import wandb
    run=wandb.init(entity='wandb',project='nanochat-demo',id='b1384b0',
                   name='nanochat-1.384B-startup-supervisor-3-attempts',config=CONFIG)
    root=Path.cwd().resolve()
    deadline=args.submitted_at+86400-30
    result=1
    try:
        result=supervise(lambda attempt:[sys.executable,'-u','big_train.py','--submitted-at',str(args.submitted_at)],
                         root,run,deadline)
    except BaseException as exc:
        run.summary['startup/failure']=str(exc);raise
    finally:
        run.summary['startup/aggregate_gpu_hour_upper_bound']=max(0,time.time()-args.submitted_at)*8/3600
        run.finish(exit_code=result)
    return result

if __name__=='__main__':sys.exit(main())