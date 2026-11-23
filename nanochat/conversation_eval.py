"""Frozen conversational-evaluation plumbing. No model loading or network calls.

Backends see token IDs only, never checks/reference answers. The default CLI
in scripts/conversation_eval.py can score existing records; generation is a
Python API and acceptance is locked unless an explicit, validated permit exists.
"""
from __future__ import annotations

import copy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

PINNED_HASHES = {
    'development.json': '14e83fe79cf00b29521a1f217f38d8c2acd89c57832b2fe119518dc79818aa8e',
    'acceptance_primary.json': '24567fb37946c380e9ed75dd71990aa5bdb63b85e39c731e8936d98ef8af78d5',
    'acceptance_correction.json': '7d8114fc999fc638818387a31f355e68d8f9d5e305e18f4a293b2da6ba1b17b3',
    'protocol.json': 'a4851c73a61ad963586dfe2c915d02150e43121de5a3f5433160886c34adaf60',
}
CAPS = {'instruction': (40, 1), 'multi_turn': (20, 3),
        'grounded': (20, 1), 'human_review': (20, 3)}
CONTROL_NAMES = ('<|bos|>', '<|user_start|>', '<|user_end|>',
                 '<|assistant_start|>', '<|assistant_end|>', '<|python_start|>',
                 '<|python_end|>', '<|output_start|>', '<|output_end|>')
IMPERSONATION = re.compile(r'(?im)(?:^|\n)\s*(?:user|human|assistant)\s*:|<\|(?:user|assistant)_(?:start|end)\|>|\[/?INST\]')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def load_frozen(folder, split):
    if split not in ('development', 'acceptance_primary', 'acceptance_correction'):
        raise ValueError('Unknown frozen split')
    folder = Path(folder)
    # Check only requested payload, not either unused acceptance payload.
    for name in ('protocol.json', split + '.json'):
        if digest(folder / name) != PINNED_HASHES[name]:
            raise ValueError('Frozen identity mismatch: ' + name)
    suite = json.loads((folder / (split + '.json')).read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    validate_suite(suite)
    return suite, protocol


def validate_suite(suite):
    cases = suite['examples']
    if len(cases) != 100 or len({c['id'] for c in cases}) != 100:
        raise ValueError('Expected 100 unique cases')
    for capability, (count, turns) in CAPS.items():
        rows = [c for c in cases if c['capability'] == capability]
        if len(rows) != count:
            raise ValueError('Wrong capability count: ' + capability)
        for row in rows:
            if len(row['messages']) != turns or len(row['turn_checks']) != turns:
                raise ValueError('Wrong turn count')
            for message, check in zip(row['messages'], row['turn_checks']):
                if set(message) != {'role', 'content'} or message['role'] != 'user' or not isinstance(message['content'], str):
                    raise ValueError('Expected scripted user text only')
                if check['kind'] not in ('exact', 'json', 'human'):
                    raise ValueError('Unknown check kind')


class ChatCodec:
    """No silent truncation; previous assistant IDs are preserved verbatim."""
    def __init__(self, tokenizer, tokenizer_sha256=None):
        self.tokenizer = tokenizer
        self.tokenizer_sha256 = tokenizer_sha256
        self.special = {name: tokenizer.encode_special(name) for name in CONTROL_NAMES}
        self.end = self.special['<|assistant_end|>']
        self.controls = set(self.special.values())

    @classmethod
    def from_directory(cls, folder):
        from nanochat.tokenizer import RustBPETokenizer
        actual = digest(Path(folder) / 'tokenizer.pkl')
        if actual != 'a228803c68627f650e640c819e71ea0b90f49a96118858ed781a00a5ba63b6bf':
            raise ValueError('Tokenizer does not match pinned nanochat tokenizer')
        return cls(RustBPETokenizer.from_directory(str(folder)), actual)

    def prompt(self, history, user, context=2048, reserve=256):
        history = copy.deepcopy(history)
        def render():
            ids = [self.special['<|bos|>']]
            for pair in history:
                ids.extend([self.special['<|user_start|>'], *self.tokenizer.encode(pair['user']),
                            self.special['<|user_end|>'], self.special['<|assistant_start|>']])
                ids.extend(pair['assistant_ids'])
            ids.extend([self.special['<|user_start|>'], *self.tokenizer.encode(user),
                        self.special['<|user_end|>'], self.special['<|assistant_start|>']])
            return ids
        removed = []
        ids = render()
        while len(ids) + reserve > context and history:
            removed.append(history.pop(0)['turn'])
            ids = render()
        if len(ids) + reserve > context:
            raise ValueError('Current user prompt cannot fit without truncation')
        return ids, history, removed

    def text(self, ids):
        # Exclude only the actual terminal EOT. Never manufacture an EOT.
        return self.tokenizer.decode(ids[:-1] if ids and ids[-1] == self.end else ids)


class NativeGreedyBackend:
    """Reference full-prefix forward path, intentionally not optimized Engine.

    No calculator, sampling, implicit end markers, model loader or device moves.
    Caller supplies an already loaded, identity-verified model. GPU integration
    and performance have NOT been established by the synthetic CPU tests.
    """
    def __init__(self, model, *, synthetic=False):
        self.model = model
        self.synthetic = synthetic

    def stream(self, prompt_ids, max_new_tokens):
        import torch
        self.model.eval()
        with torch.inference_mode():
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.model.get_device())
            for _ in range(max_new_tokens):
                logits = self.model(ids)[:, -1, :]
                if not torch.isfinite(logits).all().item():
                    raise ValueError('Nonfinite generation logits')
                token = int(logits.argmax(dim=-1).item())
                yield token
                ids = torch.cat((ids, torch.tensor([[token]], device=ids.device)), dim=1)


def _valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def authorize(suite, protocol, identity, mode, *, ledger_path=None, permit=None):
    if mode == 'synthetic':
        if suite.get('split') != 'synthetic' or identity.get('kind') != 'synthetic':
            raise ValueError('Synthetic checks must use synthetic fixtures and identity')
        return
    if mode != 'model' or suite.get('split') not in ('development', 'acceptance_primary', 'acceptance_correction'):
        raise ValueError('Invalid generation mode/split')
    split = suite['split']
    for payload, name in ((suite, split + '.json'), (protocol, 'protocol.json')):
        if hashlib.sha256(json.dumps(payload, indent=2).encode()).hexdigest() != PINNED_HASHES[name]:
            raise ValueError('Real-model evaluation requires exact pinned fixtures/protocol')
    if not _valid_digest(identity.get('checkpoint_sha256')) or not _valid_digest(identity.get('tokenizer_sha256')):
        raise ValueError('Require immutable checkpoint/tokenizer SHA256 identities')
    if ledger_path is None:
        raise ValueError('Persistent evaluation ledger required')
    if split != 'development':
        if not permit or permit.get('phase') != split or permit.get('selected_checkpoint_sha256') != identity['checkpoint_sha256']:
            raise PermissionError('Acceptance locked: explicit matching selection permit required')
        for key in ('selected_by', 'selection_rationale', 'development_evidence', 'human_fixture_review_evidence'):
            if not isinstance(permit.get(key), str) or not permit[key].strip():
                raise PermissionError('Acceptance permit missing ' + key)
        if permit.get('protocol_sha256') != PINNED_HASHES['protocol.json']:
            raise PermissionError('Selection permit decoding/protocol mismatch')
        if split == 'acceptance_correction' and not permit.get('primary_run_evidence'):
            raise PermissionError('Correction requires prior primary evaluation evidence')
    with closing(sqlite3.connect(str(ledger_path), isolation_level=None)) as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS evaluations (split TEXT, checkpoint TEXT, status TEXT, permit TEXT)')
        checkpoint = identity['checkpoint_sha256']
        if split == 'development':
            primary = db.execute("SELECT checkpoint,status FROM evaluations WHERE split='acceptance_primary'").fetchone()
            dev_rows = list(db.execute("SELECT checkpoint,permit FROM evaluations WHERE split='development'"))
            if primary:
                if primary[1] != 'complete':
                    raise PermissionError('Interrupted primary acceptance requires operator review')
                if db.execute("SELECT 1 FROM evaluations WHERE split='acceptance_correction'").fetchone():
                    raise PermissionError('No further development after reserved correction acceptance')
                if (not permit or permit.get('phase') != 'correction_development' or
                    permit.get('correction_checkpoint_sha256') != checkpoint or
                    permit.get('base_selected_checkpoint_sha256') != primary[0] or
                    not permit.get('primary_run_evidence') or not permit.get('rationale')):
                    raise PermissionError('Post-primary development requires an explicit single-correction permit')
                correction_ids = {cp for cp, raw in dev_rows
                                  if raw and json.loads(raw) and json.loads(raw).get('phase') == 'correction_development'}
                if correction_ids and checkpoint not in correction_ids:
                    raise PermissionError('Only one correction checkpoint candidate is allowed')
            else:
                known = {r[0] for r in dev_rows}
                if checkpoint not in known and len(known) >= 4:
                    raise PermissionError('Development limit: initial model plus at most three SFT checkpoints')
        else:
            if db.execute('SELECT 1 FROM evaluations WHERE split=?', (split,)).fetchone():
                raise PermissionError('Acceptance already reserved; interrupted runs are not silently regenerated')
            if not db.execute("SELECT 1 FROM evaluations WHERE split='development' AND checkpoint=? AND status='complete'", (checkpoint,)).fetchone():
                raise PermissionError('Selected checkpoint lacks completed development record')
            if split == 'acceptance_correction' and not db.execute("SELECT 1 FROM evaluations WHERE split='acceptance_primary' AND status='complete'").fetchone():
                raise PermissionError('Primary evaluation must complete before correction acceptance')
            if split == 'acceptance_correction':
                candidates = [json.loads(raw) for raw, in db.execute("SELECT permit FROM evaluations WHERE split='development' AND checkpoint=? AND status='complete'", (checkpoint,)) if raw and json.loads(raw)]
                if not any(p.get('phase') == 'correction_development' for p in candidates):
                    raise PermissionError('Correction acceptance requires the single authorized correction development record')
        db.execute('INSERT INTO evaluations VALUES (?,?,?,?)', (split, checkpoint, 'reserved', json.dumps(permit)))
        db.execute('COMMIT')


def run_suite(suite, protocol, codec, backend, identity, *, mode='synthetic', output_path,
              ledger_path=None, permit=None):
    validate_suite(suite)
    if bool(getattr(backend, 'synthetic', False)) != (mode == 'synthetic'):
        raise ValueError('Backend provenance/mode mismatch')
    path = Path(output_path)
    # Exclusive creation prevents accidental replacement; reserve acceptance
    # before invoking any backend. Failures require explicit operator review.
    if path.exists():
        raise FileExistsError(str(path))
    context, limit = protocol['model_context'], protocol['decoding']['max_new_tokens_per_turn']
    if context != 2048 or limit != 256 or protocol['decoding']['temperature'] != 0:
        raise ValueError('Unsupported decoding protocol')
    if mode == 'model' and (identity.get('tokenizer_sha256') != codec.tokenizer_sha256 or
                           codec.tokenizer_sha256 != 'a228803c68627f650e640c819e71ea0b90f49a96118858ed781a00a5ba63b6bf'):
        raise ValueError('Real generation requires verified pinned tokenizer')
    authorize(suite, protocol, identity, mode, ledger_path=ledger_path, permit=permit)
    identity = dict(identity, runner_source_sha256=digest(__file__))
    artifact = {'schema': 'conversation-eval-records-v1', 'mode': mode,
                'split': suite['split'], 'identity': copy.deepcopy(identity),
                'suite_sha256': canonical_hash(suite), 'protocol_sha256': canonical_hash(protocol),
                'records': []}
    with path.open('x') as stream:
        stream.write(json.dumps({'header': {k: v for k, v in artifact.items() if k != 'records'}}) + '\n')
        stream.flush()
        for case in suite['examples']:
            history, aborted = [], False
            for turn, message in enumerate(case['messages']):
                start = time.perf_counter()
                record = {'case_id': case['id'], 'turn': turn, 'raw_ids': [], 'text': '',
                          'prompt_ids': [], 'removed_history_turns': [], 'error': None,
                          'stop_reason': 'not_generated'}
                try:
                    if aborted:
                        raise RuntimeError('Earlier turn failed; no silent retry or reference substitution')
                    prompt, history, removed = codec.prompt(history, message['content'], context, limit)
                    record['prompt_ids'], record['removed_history_turns'] = prompt, removed
                    iterator = iter(backend.stream(list(prompt), limit))
                    try:
                        for _ in range(limit):
                            try:
                                token = next(iterator)
                            except StopIteration:
                                record['stop_reason'] = 'backend_exhausted'
                                break
                            if type(token) is not int or not 0 <= token < codec.tokenizer.get_vocab_size():
                                raise ValueError('Invalid emitted token')
                            record['raw_ids'].append(token)
                            if token == codec.end:
                                record['stop_reason'] = 'assistant_end'
                                break
                            if token in codec.controls:
                                record['stop_reason'] = 'forbidden_control'
                                break
                        else:
                            record['stop_reason'] = 'length'
                    finally:
                        if hasattr(iterator, 'close'):
                            iterator.close()
                    record['text'] = codec.text(record['raw_ids'])
                    if not record['text'].strip():
                        raise ValueError('Empty response')
                    history.append({'turn': turn, 'user': message['content'], 'assistant_ids': list(record['raw_ids'])})
                except Exception as exc:
                    record['error'] = type(exc).__name__ + ': ' + str(exc)
                    record['stop_reason'] = 'error'
                    # Preserve any emitted prefix, but never continue a failed conversation.
                    record['text'] = codec.text(record['raw_ids'])
                    aborted = True
                record['latency_seconds'] = time.perf_counter() - start
                artifact['records'].append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
    if mode == 'model':
        with closing(sqlite3.connect(str(ledger_path), isolation_level=None)) as db:
            db.execute("UPDATE evaluations SET status='complete' WHERE split=? AND checkpoint=? AND status='reserved'",
                       (suite['split'], identity['checkpoint_sha256']))
    return artifact


def run_frozen(folder, split, codec, backend, identity, *, output_path, ledger_path, permit=None):
    suite, protocol = load_frozen(folder, split)
    identity = dict(identity, frozen_files_verified=True, suite_content_sha256=canonical_hash(suite),
                    suite_file_sha256=PINNED_HASHES[split + '.json'])
    # run_suite's guard verifies the backend and all identities before generation.
    return run_suite(suite, protocol, codec, backend, identity, mode='model',
                     output_path=output_path, ledger_path=ledger_path, permit=permit)


def has_severe_phrase_loop(text):
    words = text.split()
    for size in range(4, len(words) // 4 + 1):
        for i in range(len(words) - size * 4 + 1):
            phrase = words[i:i + size]
            if all(words[i + k * size:i + (k + 1) * size] == phrase for k in range(1, 4)):
                return True
    return False


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)


def _decision(annotation, name):
    value = annotation.get(name)
    if value is None:
        return None
    if type(value) is not bool or not annotation.get('reviewer') or not str(annotation.get('rationale', '')).strip():
        raise ValueError('Human decisions require bool, reviewer and written rationale')
    return value


def _gate(values, minimum):
    passed = sum(v is True for v in values)
    pending = sum(v is None for v in values)
    return {'passed': passed, 'failed': sum(v is False for v in values), 'pending': pending,
            'denominator': len(values), 'minimum': minimum,
            'status': 'fail' if passed + pending < minimum else ('pending' if pending else ('pass' if passed >= minimum else 'fail'))}


def score_records(suite, protocol, codec, artifact, *, reviews=None, arc_evidence=None):
    """Never reduce denominators; human checks/flags block passing until reviewed."""
    validate_suite(suite)
    if artifact['split'] != suite['split']:
        raise ValueError('Evidence split mismatch')
    if artifact['mode'] == 'synthetic':
        if suite['split'] != 'synthetic' or artifact['identity'].get('kind') != 'synthetic':
            raise ValueError('Synthetic evidence cannot establish a behavioral-suite result')
    elif artifact['mode'] == 'model':
        split = suite['split']
        for value, name in ((suite, split+'.json'), (protocol, 'protocol.json')):
            if name not in PINNED_HASHES or hashlib.sha256(json.dumps(value, indent=2).encode()).hexdigest() != PINNED_HASHES[name]:
                raise ValueError('Scoring requires frozen real-model fixture/protocol identities')
        if codec.tokenizer_sha256 != artifact['identity'].get('tokenizer_sha256') or not _valid_digest(codec.tokenizer_sha256):
            raise ValueError('Scoring tokenizer identity mismatch')
        if not _valid_digest(artifact['identity'].get('checkpoint_sha256')):
            raise ValueError('Scoring checkpoint identity required')
    else:
        raise ValueError('Unknown response evidence mode')
    if artifact['suite_sha256'] != canonical_hash(suite) or artifact['protocol_sha256'] != canonical_hash(protocol):
        raise ValueError('Response evidence is for different fixtures/protocol')
    reviews = reviews or {}
    if reviews and reviews.get('records_sha256') != canonical_hash(artifact):
        raise ValueError('Reviews must bind to these exact response records')
    annotations = reviews.get('responses', {})
    index = {}
    allowed = {(c['id'], i) for c in suite['examples'] for i in range(len(c['messages']))}
    if set(annotations) - {case_id + ':' + str(i) for case_id, i in allowed}:
        raise ValueError('Unknown response annotation')
    for row in artifact['records']:
        key = (row['case_id'], row['turn'])
        if key in index or key not in allowed:
            raise ValueError('Duplicate or unexpected response record')
        index[key] = row
    response_results, cases = [], []
    for case in suite['examples']:
        checks = []
        history, tainted_history = [], False
        for i, check in enumerate(case['turn_checks']):
            row = index.get((case['id'], i))
            key = case['id'] + ':' + str(i)
            ann = annotations.get(key, {})
            healthy = bool(not tainted_history and row and not row.get('error') and row.get('raw_ids') and row.get('text', '').strip())
            raw = row['raw_ids'] if row else []
            if healthy:
                try:
                    healthy = (len(raw) <= 256 and
                               all(type(t) is int and 0 <= t < codec.tokenizer.get_vocab_size() for t in raw) and
                               row['text'] == codec.text(raw))
                    prompt, kept, removed = codec.prompt(history, case['messages'][i]['content'])
                    healthy = (healthy and not tainted_history and row.get('prompt_ids') == prompt
                               and row.get('removed_history_turns') == removed)
                    if healthy:
                        history = kept + [{'turn': i, 'user': case['messages'][i]['content'],
                                           'assistant_ids': list(raw)}]
                except Exception:
                    healthy = False
            if not healthy:
                tainted_history = True
            boundary, loop_clean, correct = False, False, False
            phrase_flag = impersonation_flag = False
            if healthy:
                boundary = (len(raw) <= 256 and raw[-1] == codec.end and
                            not any(t in codec.controls for t in raw[:-1]) and row.get('stop_reason') == 'assistant_end')
                impersonation_flag = bool(IMPERSONATION.search(row['text']))
                judged_boundary = _decision(ann, 'boundary_ok')
                if boundary:
                    boundary = (judged_boundary if judged_boundary is not None else None) if impersonation_flag else judged_boundary is not False
                phrase_flag = has_severe_phrase_loop(row['text'])
                judged_loop = _decision(ann, 'loop_free')
                loop_clean = (judged_loop if judged_loop is not None else None) if phrase_flag else judged_loop is not False
                if check['kind'] == 'exact':
                    correct = row['text'].strip() == check['value']
                elif check['kind'] == 'json':
                    try:
                        correct = canonical_hash(strict_json(row['text'])) == canonical_hash(check['value'])
                    except (ValueError, TypeError):
                        correct = False
                else:
                    correct = _decision(ann, 'check_pass')
            checks.append(correct)
            response_results.append({'case_id': case['id'], 'turn': i, 'check_pass': correct,
                                     'healthy': healthy,
                                     'boundary_ok': boundary, 'loop_free': loop_clean,
                                     'phrase_loop_flag': phrase_flag, 'impersonation_flag': impersonation_flag})
        status = False if False in checks else (None if None in checks else True)
        cases.append({'case_id': case['id'], 'capability': case['capability'], 'pass': status})
    gates = {}
    keys = {'instruction': 'instruction_min_pass', 'multi_turn': 'multi_turn_min_pass', 'grounded': 'grounded_min_pass'}
    for cap, key in keys.items():
        gates[cap] = _gate([c['pass'] for c in cases if c['capability'] == cap], protocol['gates'][key])
    gates['repetition'] = _gate([r['loop_free'] for r in response_results], protocol['gates']['responses_without_severe_loops_min'])
    gates['boundaries'] = _gate([r['boundary_ok'] for r in response_results], protocol['gates']['correct_boundaries_min'])
    human_cases = [c for c in cases if c['capability'] == 'human_review']
    failures = reviews.get('human_conversations', {})
    categories = {}
    human_complete = True
    human_case_ids = {c['case_id'] for c in human_cases}
    if set(failures) - human_case_ids:
        raise ValueError('Unknown human review case')
    for case in human_cases:
        if any(not row['healthy'] for row in response_results if row['case_id'] == case['case_id']):
            categories['missing_or_failed_response'] = categories.get('missing_or_failed_response', 0) + 1
        ann = failures.get(case['case_id'], {})
        if not ann.get('reviewer') or not str(ann.get('rationale', '')).strip() or 'blocking_categories' not in ann:
            human_complete = False
            continue
        labels = ann['blocking_categories']
        if not isinstance(labels, list) or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError('Use a list of nonempty human failure-category labels')
        for label in {' '.join(label.casefold().split()) for label in labels} - {'missing_or_failed_response'}:
            categories[label] = categories.get(label, 0) + 1
    recurring = any(n >= 2 for n in categories.values())
    gates['human_usefulness'] = {'status': 'fail' if recurring else ('pass' if human_complete and all(c['pass'] is not None for c in human_cases) else 'pending'),
                                 'reviewed_conversations': sum(bool(failures.get(c['case_id'], {}).get('reviewer') and failures.get(c['case_id'], {}).get('rationale')) for c in human_cases),
                                 'denominator': 20, 'failure_categories': categories}
    if arc_evidence is None:
        gates['arc100'] = {'status': 'pending', 'reason': 'Historical ARC100 protocol evidence required'}
    else:
        if (type(arc_evidence.get('correct')) is not int or not 0 <= arc_evidence['correct'] <= 100 or
            arc_evidence.get('checkpoint_sha256') != artifact['identity'].get('checkpoint_sha256') or
            arc_evidence.get('benchmark_artifact') != 'wandb/nanochat-demo/e1384arc-benchmark-inputs:v0' or
            not arc_evidence.get('historical_protocol_evidence')):
            raise ValueError('ARC evidence must identify checkpoint, pinned benchmark and historical protocol')
        gates['arc100'] = {'status': 'review_required' if arc_evidence['correct'] <= 49 else 'pass',
                          'correct': arc_evidence['correct'], 'denominator': 100, 'baseline_correct': 55}
    states = [g['status'] for g in gates.values()]
    outcome = ('fail' if 'fail' in states else 'review_required' if 'review_required' in states
               else 'pending' if 'pending' in states else 'pass')
    return {'schema': 'conversation-eval-score-v1', 'mode': artifact['mode'],
            'split': suite['split'], 'records_sha256': canonical_hash(artifact),
            'gates': gates, 'gate_outcome': outcome,
            'release_decision': ('synthetic_not_release_evidence' if artifact['mode'] == 'synthetic'
                                 else 'not_acceptance' if suite['split'] == 'development'
                                 else 'acceptance_' + outcome + '_other_release_checks_still_required'),
            'case_results': cases, 'response_results': response_results,
            'training_approved': False, 'release_approved': False,
            'human_review_template': {'records_sha256': canonical_hash(artifact),
                                      'responses': {}, 'human_conversations': {}}}


def read_records(path):
    with Path(path).open() as stream:
        header = json.loads(next(stream))['header']
        return dict(header, records=[json.loads(line) for line in stream if line.strip()])
