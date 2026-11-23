"""CPU-safe CLI. No command in this module loads model weights or runs a model."""
import argparse
import json
from pathlib import Path

from nanochat.conversation_eval import (ChatCodec, canonical_hash, load_frozen,
                                      read_records, run_suite, score_records)
from nanochat.conversation_eval_synthetic import CharacterTokenizer, ScriptedBackend, synthetic_suite


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def review_template(suite, records):
    return {'records_sha256': canonical_hash(records),
            'notice': 'Fill by human review of actual outputs. No LLM judge is implemented. Synthetic records are NOT model evidence.',
            'responses': {c['id'] + ':' + str(i): {
                'reviewer': '', 'rationale': '',
                **({'check_pass': None} if check['kind'] == 'human' else {}),
                'loop_free': None, 'boundary_ok': None,
                'criteria': check.get('criteria', []), 'check_kind': check['kind']}
                for c in suite['examples'] for i, check in enumerate(c['turn_checks'])},
            'human_conversations': {c['id']: {'reviewer': '', 'rationale': '', 'blocking_categories': []}
                                    for c in suite['examples'] if c['capability'] == 'human_review'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('synthetic-smoke', help='Fabricated fixtures/replay only; no frozen suite responses')
    smoke.add_argument('--suite-dir', type=Path, required=True, help='Reads protocol.json only')
    smoke.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    score = commands.add_parser('score', help='Score existing token evidence; never generates responses')
    score.add_argument('--suite-dir', type=Path, required=True)
    score.add_argument('--records', type=Path, required=True)
    score.add_argument('--tokenizer-dir', type=Path)
    score.add_argument('--reviews', type=Path)
    score.add_argument('--arc-evidence', type=Path)
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--review-template-output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'synthetic-smoke':
        from nanochat.conversation_eval import digest, PINNED_HASHES
        if digest(args.suite_dir / 'protocol.json') != PINNED_HASHES['protocol.json']:
            raise ValueError('Frozen protocol mismatch')
        protocol = json.loads((args.suite_dir / 'protocol.json').read_text())
        suite = synthetic_suite()
        codec = ChatCodec(CharacterTokenizer())
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_new(args.output_dir/'synthetic-suite.json', suite)
        with (args.output_dir/'protocol.json').open('x') as stream:
            stream.write((args.suite_dir/'protocol.json').read_text())
        identity = {'kind': 'synthetic', 'description': 'SCRIPTED TEST, NOT MODEL OUTPUT'}
        backend = ScriptedBackend(codec)
        records = run_suite(suite, protocol, codec, backend, identity,
                            output_path=args.output_dir/'synthetic-records.jsonl')
        result = score_records(suite, protocol, codec, records)
        write_new(args.output_dir/'synthetic-score.json', result)
        write_new(args.output_dir/'human-review-template.json', review_template(suite, records))
        print(json.dumps({'status': 'synthetic_plumbing_completed', 'cases': 100, 'scripted_turns': backend.calls,
                          'release_decision': result['release_decision'], 'gpu_jobs_launched': 0,
                          'model_evaluations_run': 0}))
    else:
        records = read_records(args.records)
        if records['mode'] == 'synthetic':
            if records['split'] != 'synthetic':
                raise ValueError('Synthetic evidence cannot score a frozen behavioral split')
            suite = json.loads((args.suite_dir/'synthetic-suite.json').read_text())
            # Protocol is immutable and stored separately from fabricated content.
            from nanochat.conversation_eval import digest, PINNED_HASHES
            protocol_path = args.suite_dir/'protocol.json'
            if digest(protocol_path) != PINNED_HASHES['protocol.json']:
                raise ValueError('Frozen protocol mismatch')
            protocol = json.loads(protocol_path.read_text())
            codec = ChatCodec(CharacterTokenizer())
        elif records['mode'] == 'model':
            if args.tokenizer_dir is None:
                raise ValueError('--tokenizer-dir required for actual model evidence')
            suite, protocol = load_frozen(args.suite_dir, records['split'])
            codec = ChatCodec.from_directory(args.tokenizer_dir)
        else:
            raise ValueError('Unknown evidence mode')
        reviews = json.loads(args.reviews.read_text()) if args.reviews else None
        arc = json.loads(args.arc_evidence.read_text()) if args.arc_evidence else None
        result = score_records(suite, protocol, codec, records, reviews=reviews, arc_evidence=arc)
        write_new(args.output, result)
        if args.review_template_output:
            write_new(args.review_template_output, review_template(suite, records))
        print(json.dumps({k: result[k] for k in ('mode', 'split', 'gate_outcome', 'release_decision')}))


if __name__ == '__main__':
    main()