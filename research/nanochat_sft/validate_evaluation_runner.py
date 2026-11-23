"""Reproducible CPU-only runner/regression evidence; does not load weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scratch = out/'scratch'
    scratch.mkdir()
    env = dict(os.environ, PYTHONPATH=str(root/'prepared'), OMP_NUM_THREADS='1',
               TORCHDYNAMO_DISABLE='1', TMPDIR=str(scratch))
    evidence = {'scope': 'Synthetic CPU plumbing and existing CPU preparation regressions only',
                'model_evaluations_run': 0, 'gpu_jobs_launched': 0,
                'frozen_acceptance_generation_calls': 0,
                'native_checkpoint_generation_tested': False,
                'human_fixture_review': 'pending', 'launch_ready': False,
                'training_approved': False, 'release_approved': False, 'tests': {}}
    for name in ('conversation_eval_runner', 'sft_preparation', 'conversation_eval_freeze'):
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_'+name+'.py', '-v']
        result = subprocess.run(command, cwd=root/'prepared', env=env, text=True, capture_output=True)
        log = result.stdout + result.stderr
        (out/(name+'.log')).write_text(log)
        found = re.search(r'Ran (\d+) tests? in', log)
        evidence['tests'][name] = {'returncode': result.returncode, 'run': int(found[1]) if found else None,
                                   'log_sha256': hashlib.sha256(log.encode()).hexdigest(),
                                   'command': command}
        (out/'test_evidence.json').write_text(json.dumps(evidence, indent=2))
        if result.returncode or not found or not log.rstrip().endswith('OK'):
            print(log, flush=True)
            raise RuntimeError('Targeted tests failed: '+name)
        print(json.dumps({'tests': name, 'passed': int(found[1])}), flush=True)
    smoke = out/'synthetic-smoke'
    command = [sys.executable, '-m', 'scripts.conversation_eval', 'synthetic-smoke',
               '--suite-dir', str(root/'conversation_evals_v1'), '--output-dir', str(smoke)]
    result = subprocess.run(command, cwd=root/'prepared', env=env, text=True, capture_output=True)
    (out/'smoke.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('Synthetic smoke failed: '+result.stderr)
    score = json.loads((smoke/'synthetic-score.json').read_text())
    assert score['release_decision'] == 'synthetic_not_release_evidence'
    assert score['gates']['human_usefulness']['status'] == 'pending'
    assert score['gates']['arc100']['status'] == 'pending'
    for key in ('instruction','multi_turn','grounded','repetition','boundaries'):
        assert score['gates'][key]['status'] == 'pass', (key,score['gates'][key])
    evidence['synthetic_smoke'] = json.loads(result.stdout)
    evidence['targeted_tests_passed'] = sum(t['run'] for t in evidence['tests'].values())
    evidence['repository_wide_tests_claimed'] = False
    evidence['known_historical_test_issue'] = 'TestAccumulationTelemetry AST misses require_finite_loss; not modified in this work'
    (out/'test_evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps({'targeted_tests_passed': evidence['targeted_tests_passed'],
                      'synthetic_smoke': evidence['synthetic_smoke']}), flush=True)


if __name__ == '__main__':
    main()