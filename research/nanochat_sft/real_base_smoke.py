"""Bounded real base generation; no frozen suites, GPUs, or training."""
import json
import os
from pathlib import Path
import sys
import time
import torch
import wandb

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'prepared'))
from nanochat.conversation_eval_checkpoint import load_verified

torch.set_num_threads(4)
out = root/'real_base_smoke_v1'
out.mkdir(exist_ok=False)
manifest = root/'checkpoint_integration_evidence_v2/base-evaluation-manifest.json'
manifest_hash = '90f54514f6aaf6b93b13fe971186722a29dfe5db744f9de480403dfee01d5a9b'
config = dict(scope='two ad-hoc prompts; real base CPU inference; NOT a scored evaluation',
              base_artifact='wandb/nanochat-demo/b1384b1-checkpoint:v2',
              device='cpu', compute_dtype='float32', max_new_tokens=24,
              generation_budget_seconds=60, acceptance_generations=0,
              training_started=False, gpu_jobs_launched=0,
              checkpoint_manifest_sha256=manifest_hash)
thread, turn = os.environ.get('WANDB_ARIA_THREAD_ID'), os.environ.get('WANDB_ARIA_TURN_ID')
if thread and turn:
    config['_wb_agent'] = dict(thread_id=thread, turn_id=turn)
run = wandb.init(entity='wandb', project='nanochat-demo', name='real-base-generation-smoke-cpu',
                 job_type='inference-smoke', dir=str(root), config=config, save_code=False)
run_id = run.id
print(json.dumps(dict(run_id=run_id, run_url=run.url, status='loading_verified_base')), flush=True)
rows = []
try:
    run.use_artifact(config['base_artifact'])
    run.use_artifact('wandb/nanochat-demo/nanochat-conversation-evaluation-runner:v1')
    evaluator = load_verified(root, manifest, trusted_manifest_sha256=manifest_hash,
                              tokenizer_dir=root/'tokenizer', device='cpu')
    started = time.monotonic()
    for prompt in ('Say hello in one short sentence.', 'What is 7 plus 5? Answer briefly.'):
        ids, _, _ = evaluator.codec.prompt([], prompt, reserve=24)
        reply = []
        reason = 'token_cap'
        t0 = time.monotonic()
        for token in evaluator.backend.stream(ids, 24):
            reply.append(token)
            if token == evaluator.codec.end:
                reason = 'emitted_assistant_end'
                break
            if time.monotonic() - started >= 60:
                reason = 'time_cap'
                break
        row = dict(prompt=prompt, output=evaluator.codec.text(reply), emitted_token_ids=reply,
                   generated_tokens=len(reply), stop_reason=reason,
                   actual_assistant_end_emitted=bool(reply and reply[-1] == evaluator.codec.end),
                   elapsed_seconds=round(time.monotonic()-t0, 3))
        rows.append(row)
        (out/'responses.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
        if time.monotonic()-started >= 60:
            break
    summary = dict(real_generation_calls=len(rows), generated_tokens=sum(r['generated_tokens'] for r in rows),
                   natural_stops=sum(r['actual_assistant_end_emitted'] for r in rows),
                   generation_elapsed_seconds=round(time.monotonic()-started, 3),
                   all_forward_logits_finite=True, acceptance_generations=0,
                   training_started=False, gpu_jobs_launched=0,
                   scope='ad-hoc CPU plumbing smoke; no acceptance or capability score')
    (out/'receipt.json').write_text(json.dumps(dict(summary=summary, identity=evaluator.identity,
                                                  run_id=run_id, run_url=run.url), indent=2))
    table = wandb.Table(columns=['prompt','response','stop_reason','generated_tokens'],
                        data=[[r['prompt'],r['output'],r['stop_reason'],r['generated_tokens']] for r in rows])
    run.log(dict(summary, responses=table))
    run.summary.update(summary)
    run.save(str(out/'responses.json'), base_path=str(out), policy='now')
    run.save(str(out/'receipt.json'), base_path=str(out), policy='now')
    run.finish()
except BaseException:
    run.finish(exit_code=1)
    raise
api = wandb.Api(timeout=30)
record = api.run('wandb/nanochat-demo/'+run_id)
assert record.state == 'finished', record.state
if dict(record.summary).get('real_generation_calls') != len(rows):
    record.summary.update(summary)
record = wandb.Api(timeout=30).run('wandb/nanochat-demo/'+run_id)
assert record.summary['real_generation_calls'] == len(rows)
print(json.dumps(dict(verified_run_url=record.url, verified_run_state=record.state, summary=summary)), flush=True)