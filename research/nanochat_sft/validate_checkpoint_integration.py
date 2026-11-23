"""Reproducible CPU integration checks; NO base-model forward or generation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NANOCHAT_DTYPE='float32',
               PYTHONPATH=str(root/'prepared'), OMP_NUM_THREADS='2')
    results = {}
    for name in ('conversation_eval_checkpoint', 'conversation_eval_runner',
                 'sft_preparation', 'conversation_eval_freeze'):
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                   '-p', 'test_' + name + '.py', '-v']
        result = subprocess.run(command, cwd=root/'prepared', env=env,
                                text=True, capture_output=True)
        log = result.stdout + result.stderr
        (out/(name + '.log')).write_text(log)
        found = re.search(r'Ran (\d+) tests? in', log)
        results[name] = {'returncode': result.returncode,
                         'run': int(found[1]) if found else None,
                         'log_sha256': hashlib.sha256(log.encode()).hexdigest()}
        if result.returncode or not found or not log.rstrip().endswith('OK'):
            print(log, flush=True)
            raise RuntimeError('CPU test failed: ' + name)
        print(json.dumps({'tests': name, 'passed': int(found[1])}), flush=True)

    # CPU-only BEFORE any native model import, also when invoked outside shell.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NANOCHAT_DTYPE'] = 'float32'
    sys.path.insert(0, str(root/'prepared'))
    from nanochat import conversation_eval_checkpoint as checkpoint
    from nanochat.conversation_eval import ChatCodec, PINNED_HASHES, load_frozen
    import torch
    torch.set_num_threads(4)
    manifest = dict(schema='nanochat-eval-checkpoint-v1', kind='base',
                    model_config=checkpoint.MODEL_CONFIG,
                    base_weights_sha256=checkpoint.BASE_WEIGHTS,
                    checkpoint_sha256=checkpoint.BASE_WEIGHTS,
                    tokenizer_sha256=checkpoint.TOKENIZER,
                    weights_file='base_metadata/model_133632.pt',
                    metadata_file='base_metadata/meta_133632.json',
                    source_sha256={name: checkpoint.sha256(Path(checkpoint.__file__).parent/name)
                                   for name in checkpoint.SOURCE_NAMES})
    manifest_file = out/'base-evaluation-manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2))
    trust = checkpoint.sha256(manifest_file)
    print('Loading the pinned base on CPU, scanning tensors; no forward/generation.', flush=True)
    evaluator = checkpoint.load_verified(root, manifest_file,
                  trusted_manifest_sha256=trust, tokenizer_dir=root/'tokenizer')
    model = evaluator.backend.model
    assert len(model.state_dict()) == 175
    assert sum(p.numel() for p in model.parameters()) == 1384122122
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert evaluator.backend.identity == evaluator.identity
    assert model.cos.device.type == model.sin.device.type == 'cpu'
    codec = evaluator.codec
    assert codec.tokenizer.get_vocab_size() == 32768
    # Render one invented, non-eval prompt; do NOT invoke the model.
    prompt, history, removed = codec.prompt([], 'Integration plumbing only. Reply briefly.')
    assert len(prompt) + 256 <= 2048 and history == [] and removed == []
    load_receipt = dict(status='verified_CPU_load_only_no_forward', identity=evaluator.identity,
                        tensors=175, parameters=1384122122,
                        all_tensors_finite=True, rotary_buffers_rebuilt=True,
                        synthetic_prompt_rendered=True, base_model_forward_calls=0,
                        base_model_generation_calls=0, gpu_jobs_launched=0)
    (out/'base-load-receipt.json').write_text(json.dumps(load_receipt, indent=2))

    suite_dir = root/'conversation_evals_v1'
    suite, protocol = load_frozen(suite_dir, 'development')
    # Only byte hashes for sealed acceptance; NEVER parse acceptance payloads.
    fixture_hashes = {name: checkpoint.sha256(suite_dir/name) for name in PINNED_HASHES}
    assert fixture_hashes == PINNED_HASHES
    counts = dict(Counter(case['capability'] for case in suite['examples']))
    check_counts = dict(Counter(check['kind'] for case in suite['examples'] for check in case['turn_checks']))
    review = dict(scope='AI-assisted development content/protocol review; NOT human approval',
                  development_cases=len(suite['examples']),
                  development_responses=sum(len(case['messages']) for case in suite['examples']),
                  capability_counts=counts, turn_check_counts=check_counts,
                  frozen_hashes=fixture_hashes, fixture_files_changed=False,
                  acceptance_payloads_parsed=False, acceptance_generations=0,
                  blockers=[
                      dict(family='exclude_distractors', cases=4, severity='ambiguous_prompt',
                           detail='Amber, violet, silver and coral have non-color senses. Object-name exclusion is not unambiguous.',
                           proposal='Version explicitly typed distractors rather than silently changing expected answers.'),
                      dict(family='reference_resolution', cases=4, severity='underspecified_exact_format',
                           detail='Location phrase is not constrained to exact wording; e.g. the east shelf is semantically correct but fails east shelf.',
                           proposal='Version an explicit answer format or pre-register accepted variants.')],
                  limitations=[
                      '25 template families repeated over four content instances; splits are not independent task-family generalization.',
                      'explicit_correction adds a destination rather than replacing one; replace_value and changed_preference do test real corrections.',
                      'Sealed acceptance shares generator families; human review must assess corresponding risks without tuning on outputs.',
                      'This AI-assisted content review is not the human fixture review or human model-usefulness review.'],
                  release_status='blocked_pending_fixture_revision_and_human_review')
    (out/'fixture-review.json').write_text(json.dumps(review, indent=2))
    evidence = dict(scope='CPU checkpoint plumbing and fixture review, NOT conversational quality',
                    tests=results, targeted_tests_passed=sum(row['run'] for row in results.values()),
                    verified_base_load=load_receipt, fixture_review=review,
                    checkpoint_manifest_sha256=trust,
                    base_model_forward_calls=0, base_model_generation_calls=0,
                    model_evaluations_run=0, gpu_jobs_launched=0,
                    acceptance_generations=0, sft_checkpoints_tested='fabricated completeness manifests only',
                    gpu_integration_tested=False, human_fixture_review='pending',
                    training_approved=False, launch_ready=False, release_approved=False)
    (out/'test-evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps({'targeted_tests_passed': evidence['targeted_tests_passed'],
                      'base_load': load_receipt['status'], 'fixture_blockers': 2,
                      'model_evaluations_run': 0, 'gpu_jobs_launched': 0}), flush=True)


if __name__ == '__main__':
    main()