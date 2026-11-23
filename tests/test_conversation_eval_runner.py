"""Scripted CPU plumbing tests ONLY. No nanochat weights or real suite responses."""
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from nanochat.conversation_eval import (ChatCodec, NativeGreedyBackend, PINNED_HASHES,
    authorize, canonical_hash, has_severe_phrase_loop, load_frozen, read_records,
    run_suite, score_records, strict_json, validate_suite)

ROOT = Path(__file__).resolve().parents[1]


from nanochat.conversation_eval_synthetic import CharacterTokenizer, ScriptedBackend, synthetic_suite


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.folder = Path(self.temp.name)
        self.codec = ChatCodec(CharacterTokenizer())
        # Only protocol and development fixture metadata are read here. No responses
        # are generated for any frozen behavioral suite, including development.
        self.protocol = json.loads((ROOT/'conversation_evals_v3/protocol.json').read_text())
        self.suite = synthetic_suite()
        self.identity = {'kind': 'synthetic', 'checkpoint_sha256': 'a'*64,
                         'tokenizer_sha256': 'b'*64, 'description': 'SCRIPTED TEST, NOT A MODEL RESULT'}
    def tearDown(self): self.temp.cleanup()
    def run_fake(self, custom=None, suite=None):
        backend = ScriptedBackend(self.codec, custom)
        path = self.folder/f'records-{len(list(self.folder.glob("records-*")))}.jsonl'
        suite = suite or self.suite
        artifact = run_suite(suite, self.protocol, self.codec, backend, self.identity, output_path=path)
        self.assertEqual(read_records(path), artifact)
        return artifact, backend
    def score(self, artifact, **kw):
        return score_records(self.suite, self.protocol, self.codec, artifact, **kw)
    def reviewed(self, artifact):
        responses, conversations = {}, {}
        for case in self.suite['examples']:
            if case['capability'] == 'human_review':
                conversations[case['id']] = {'reviewer': 'synthetic-test-reviewer',
                    'rationale': 'Fabricated test annotation; not a real human quality assessment.', 'blocking_categories': []}
                for turn in range(3):
                    responses[f'{case["id"]}:{turn}'] = {'check_pass': True, 'reviewer': 'synthetic-test-reviewer',
                        'rationale': 'Synthetic schema exercise only.'}
        return {'records_sha256': canonical_hash(artifact), 'responses': responses, 'human_conversations': conversations}
    def arc(self, correct):
        return {'correct': correct, 'checkpoint_sha256': 'a'*64,
                'benchmark_artifact': 'wandb/nanochat-demo/e1384arc-benchmark-inputs:v0',
                'historical_protocol_evidence': 'synthetic-placeholder-not-a-real-ARC-result'}

    def test_full_synthetic_suite_fixed_denominators_and_pending_human_arc(self):
        artifact, backend = self.run_fake()
        result = self.score(artifact)
        self.assertEqual(backend.calls, 180)
        self.assertEqual(len(result['case_results']), 100)
        self.assertEqual(result['gates']['boundaries']['denominator'], 180)
        self.assertEqual(result['gates']['human_usefulness']['status'], 'pending')
        self.assertEqual(result['gates']['arc100']['status'], 'pending')
        self.assertEqual(result['release_decision'], 'synthetic_not_release_evidence')

    def test_missing_record_fails_case_and_both_response_gates(self):
        artifact, _ = self.run_fake()
        artifact['records'].pop(0)
        result = self.score(artifact)
        self.assertIs(result['case_results'][0]['pass'], False)
        for gate in ('boundaries', 'repetition'):
            self.assertEqual(result['gates'][gate]['failed'], 1)
            self.assertEqual(result['gates'][gate]['denominator'], 180)

    def test_empty_answer_fails_all_checks(self):
        artifact, _ = self.run_fake({0: [self.codec.end]})
        self.assertIn('Empty response', artifact['records'][0]['error'])
        result = self.score(artifact)
        self.assertFalse(result['response_results'][0]['loop_free'])
        self.assertFalse(result['response_results'][0]['boundary_ok'])

    def test_stopping_never_appends_end_marker(self):
        artifact, _ = self.run_fake({0: self.codec.tokenizer.encode('synthetic-answer')})
        row = artifact['records'][0]
        self.assertNotIn(self.codec.end, row['raw_ids'])
        self.assertEqual(row['stop_reason'], 'backend_exhausted')
        self.assertFalse(self.score(artifact)['response_results'][0]['boundary_ok'])

    def test_length_and_last_token_end_boundary(self):
        artifact, _ = self.run_fake({0: [ord('x')]*256 + [self.codec.end],
                                    1: [ord('x')]*255 + [self.codec.end]})
        result = self.score(artifact)
        self.assertEqual(len(artifact['records'][0]['raw_ids']), 256)
        self.assertEqual(artifact['records'][0]['stop_reason'], 'length')
        self.assertFalse(result['response_results'][0]['boundary_ok'])
        self.assertTrue(result['response_results'][1]['boundary_ok'])

    def test_forbidden_control_stops_without_tools(self):
        forbidden = self.codec.special['<|python_start|>']
        artifact, _ = self.run_fake({0: [ord('x'), forbidden, self.codec.end]})
        self.assertEqual(artifact['records'][0]['raw_ids'], [ord('x'), forbidden])
        self.assertFalse(self.score(artifact)['response_results'][0]['boundary_ok'])

    def test_failed_multiturn_preserves_partial_tokens_and_never_retries(self):
        # First multi-turn case starts after 40 single-turn instructions.
        artifact, backend = self.run_fake({40: [ord('p'), RuntimeError('scripted failure')]})
        self.assertEqual(artifact['records'][40]['raw_ids'], [ord('p')])
        self.assertIn('Earlier turn failed', artifact['records'][41]['error'])
        self.assertIn('Earlier turn failed', artifact['records'][42]['error'])
        self.assertEqual(backend.calls, 178)
        result = self.score(artifact)
        self.assertEqual(result['gates']['boundaries']['failed'], 3)
        self.assertFalse(result['case_results'][40]['pass'])

    def test_history_actual_replies_only_not_expected_answers(self):
        suite = copy.deepcopy(self.suite)
        suite['examples'][40]['turn_checks'][0]['value'] = 'REFERENCE_MUST_NOT_LEAK'
        actual = self.codec.tokenizer.encode('actual-first-reply')+[self.codec.end]
        artifact, backend = self.run_fake({40: actual}, suite=suite)
        rendered = self.codec.tokenizer.decode(backend.prompts[41])
        self.assertIn('actual-first-reply', rendered)
        self.assertNotIn('REFERENCE_MUST_NOT_LEAK', rendered)
        self.assertEqual(artifact['records'][41]['removed_history_turns'], [])

    def test_history_pair_removal_preserves_current_and_raw_reply(self):
        history = [{'turn': i, 'user': 'old '+str(i)+'u'*600,
                    'assistant_ids': self.codec.tokenizer.encode('a'*600)+[self.codec.end]} for i in range(2)]
        ids, kept, removed = self.codec.prompt(history, 'CURRENT')
        self.assertEqual(removed, [0])
        self.assertEqual([p['turn'] for p in kept], [1])
        self.assertEqual(len(history), 2)
        self.assertLessEqual(len(ids)+256, 2048)
        self.assertIn('CURRENT', self.codec.tokenizer.decode(ids))
        self.assertEqual(ids.count(self.codec.end), 1)

    def test_oversized_current_prompt_fails_without_backend(self):
        suite = copy.deepcopy(self.suite)
        suite['examples'][0]['messages'][0]['content'] = 'z'*1800
        artifact, backend = self.run_fake(suite=suite)
        self.assertEqual(backend.calls, 179)
        self.assertIn('cannot fit', artifact['records'][0]['error'])

    def test_strict_json_rejects_duplicate_nan_prose_and_type_coercion(self):
        for text in ('{"a":1,"a":1}', '{"a":NaN}', 'prefix {"a":1}', '{"a":Infinity}', '{"a":1} suffix'):
            with self.subTest(text=text), self.assertRaises(ValueError): strict_json(text)
        self.assertEqual(strict_json(' {"a":"ok"} '), {'a':'ok'})
        self.assertNotEqual(canonical_hash(strict_json('{"a":true}')), canonical_hash({'a':1}))
        suite = copy.deepcopy(self.suite)
        suite['examples'][0]['turn_checks'] = [{'kind':'json','value':{'owner':'test'}}]
        artifact, _ = self.run_fake({0:self.codec.tokenizer.encode('{"owner":"test","extra":"bad"}')+[self.codec.end]},suite=suite)
        self.assertFalse(score_records(suite,self.protocol,self.codec,artifact)['case_results'][0]['pass'])

    def test_loop_flags_require_adjudication_and_semantic_override(self):
        self.assertTrue(has_severe_phrase_loop('one two three four '*4))
        self.assertFalse(has_severe_phrase_loop('one two three four '*3))
        ids = self.codec.tokenizer.encode('one two three four '*4)+[self.codec.end]
        artifact, _ = self.run_fake({0:ids})
        self.assertEqual(self.score(artifact)['gates']['repetition']['status'], 'pending')
        reviews = self.reviewed(artifact)
        reviews['responses']['synthetic-instruction-0:0'] = {'loop_free':False,'reviewer':'synthetic','rationale':'Scripted sustained loop.'}
        reviews['responses']['synthetic-instruction-1:0'] = {'loop_free':False,'reviewer':'synthetic','rationale':'Synthetic semantic-loop override.'}
        result = self.score(artifact,reviews=reviews)
        self.assertEqual(result['gates']['repetition']['failed'],2)

    def test_textual_impersonation_pending_but_cannot_override_invalid_raw(self):
        ids = self.codec.tokenizer.encode('answer\nUser: fake')+[self.codec.end]
        artifact, _ = self.run_fake({0:ids,1:self.codec.tokenizer.encode('no EOT')})
        result = self.score(artifact)
        self.assertIsNone(result['response_results'][0]['boundary_ok'])
        reviews = self.reviewed(artifact)
        for i in (0,1):
            reviews['responses'][f'synthetic-instruction-{i}:0'] = {'boundary_ok':True,'reviewer':'synthetic','rationale':'Synthetic override test.'}
        result = self.score(artifact,reviews=reviews)
        self.assertTrue(result['response_results'][0]['boundary_ok'])
        self.assertFalse(result['response_results'][1]['boundary_ok'])

    def test_stale_unattributed_and_unknown_reviews_rejected(self):
        artifact, _ = self.run_fake()
        reviews = self.reviewed(artifact)
        stale = copy.deepcopy(reviews); stale['records_sha256']='0'*64
        with self.assertRaises(ValueError): self.score(artifact,reviews=stale)
        reviews['responses']['synthetic-instruction-0:0']={'loop_free':False}
        with self.assertRaises(ValueError): self.score(artifact,reviews=reviews)
        reviews=self.reviewed(artifact); reviews['responses']['typo:0']={}
        with self.assertRaises(ValueError): self.score(artifact,reviews=reviews)

    def test_recurring_human_failure_fails_and_arc_49_requires_review(self):
        artifact, _ = self.run_fake()
        reviews = self.reviewed(artifact)
        self.assertEqual(self.score(artifact,reviews=reviews,arc_evidence=self.arc(50))['gate_outcome'],'pass')
        result = self.score(artifact,reviews=reviews,arc_evidence=self.arc(49))
        self.assertEqual(result['gates']['arc100']['status'],'review_required')
        self.assertEqual(result['release_decision'],'synthetic_not_release_evidence')
        for key in list(reviews['human_conversations'])[:2]:
            reviews['human_conversations'][key]['blocking_categories']=['synthetic-demo-blocker']
        self.assertEqual(self.score(artifact,reviews=reviews)['gates']['human_usefulness']['status'],'fail')
        wrong=self.arc(55);wrong['checkpoint_sha256']='c'*64
        with self.assertRaises(ValueError):self.score(artifact,arc_evidence=wrong)

    def test_duplicate_records_and_mutated_text_do_not_pass(self):
        artifact, _ = self.run_fake()
        modified=copy.deepcopy(artifact);modified['records'].append(modified['records'][0])
        with self.assertRaises(ValueError):self.score(modified)
        artifact['records'][0]['text']='forged rendering'
        self.assertFalse(self.score(artifact)['response_results'][0]['check_pass'])

    def test_score_reconstructs_prompt_history_and_rejects_tampering(self):
        artifact, _ = self.run_fake()
        artifact['records'][40]['prompt_ids'].append(ord('!'))
        result = self.score(artifact)
        self.assertEqual(result['gates']['boundaries']['failed'], 3)
        self.assertTrue(all(not row['healthy'] for row in result['response_results'][40:43]))

    def test_unknown_or_mislabeled_evidence_mode_does_not_score(self):
        artifact, _ = self.run_fake()
        artifact['mode'] = 'unverified'
        with self.assertRaises(ValueError): self.score(artifact)
        artifact['mode'] = 'model'
        with self.assertRaises(ValueError): self.score(artifact)

    def test_no_overwrite_and_synthetic_backend_cannot_run_acceptance(self):
        artifact, backend = self.run_fake()
        path=self.folder/'existing.jsonl';path.write_text('KEEP')
        with self.assertRaises(FileExistsError):
            run_suite(self.suite,self.protocol,self.codec,backend,self.identity,output_path=path)
        suite=copy.deepcopy(self.suite);suite['split']='acceptance_primary'
        with self.assertRaises(ValueError):
            run_suite(suite,self.protocol,self.codec,backend,self.identity,output_path=self.folder/'no.jsonl')
        self.assertEqual(path.read_text(),'KEEP')
        self.assertEqual(backend.calls,180)

    def test_frozen_development_integrity_and_schema_fail_closed(self):
        suite, protocol = load_frozen(ROOT/'conversation_evals_v3','development')
        self.assertEqual(len(suite['examples']),100)
        target=self.folder/'frozen';target.mkdir()
        (target/'protocol.json').write_text(json.dumps(protocol)+' ')
        with self.assertRaises(ValueError):load_frozen(target,'development')
        suite=synthetic_suite();suite['examples'][0]['id']=suite['examples'][1]['id']
        with self.assertRaises(ValueError):validate_suite(suite)

    def test_acceptance_permit_one_shot_and_dev_budget_without_generation(self):
        # Authorization only; patch pinned identities to SYNTHETIC content.
        # No acceptance payload is read and no backend is invoked by this test.
        suite=synthetic_suite();suite['split']='acceptance_primary'
        identity={'checkpoint_sha256':'a'*64,'tokenizer_sha256':'b'*64}
        ledger=self.folder/'ledger.sqlite'
        patched=dict(PINNED_HASHES)
        import hashlib
        patched['acceptance_primary.json']=hashlib.sha256(json.dumps(suite,indent=2).encode()).hexdigest()
        with patch.dict('nanochat.conversation_eval.PINNED_HASHES',patched):
            with self.assertRaises(PermissionError):authorize(suite,self.protocol,identity,'model',ledger_path=ledger)
            permit={'phase':'acceptance_primary','selected_checkpoint_sha256':'a'*64,
                'selected_by':'synthetic','selection_rationale':'test only','development_evidence':'synthetic',
                'human_fixture_review_evidence':'synthetic','protocol_sha256':PINNED_HASHES['protocol.json']}
            with self.assertRaises(PermissionError):authorize(suite,self.protocol,identity,'model',ledger_path=ledger,permit=permit)
            with closing(sqlite3.connect(ledger, isolation_level=None)) as db:
                db.execute('CREATE TABLE IF NOT EXISTS evaluations (split TEXT, checkpoint TEXT, status TEXT, permit TEXT)')
                db.execute("INSERT INTO evaluations VALUES ('development',?,'complete','null')",('a'*64,))
            authorize(suite,self.protocol,identity,'model',ledger_path=ledger,permit=permit)
            with self.assertRaises(PermissionError):authorize(suite,self.protocol,identity,'model',ledger_path=ledger,permit=permit)
        dev,proto=load_frozen(ROOT/'conversation_evals_v3','development')
        devledger=self.folder/'dev-budget.sqlite'
        for ch in 'abcd':authorize(dev,proto,dict(identity,checkpoint_sha256=ch*64),'model',ledger_path=devledger)
        with self.assertRaises(PermissionError):authorize(dev,proto,dict(identity,checkpoint_sha256='e'*64),'model',ledger_path=devledger)

    def test_only_one_post_primary_correction_candidate_no_generation(self):
        import hashlib
        ledger=self.folder/'correction.sqlite'
        dev,protocol=load_frozen(ROOT/'conversation_evals_v3','development')
        identity={'checkpoint_sha256':'b'*64,'tokenizer_sha256':'c'*64}
        with closing(sqlite3.connect(ledger,isolation_level=None)) as db:
            db.execute('CREATE TABLE evaluations (split TEXT, checkpoint TEXT, status TEXT, permit TEXT)')
            db.execute("INSERT INTO evaluations VALUES ('acceptance_primary',?,'complete','null')",('a'*64,))
        with self.assertRaises(PermissionError):authorize(dev,protocol,identity,'model',ledger_path=ledger)
        permit={'phase':'correction_development','correction_checkpoint_sha256':'b'*64,
                'base_selected_checkpoint_sha256':'a'*64,'primary_run_evidence':'synthetic only',
                'rationale':'Fabricated authorization check'}
        authorize(dev,protocol,identity,'model',ledger_path=ledger,permit=permit)
        second=dict(permit,correction_checkpoint_sha256='d'*64)
        with self.assertRaises(PermissionError):authorize(dev,protocol,dict(identity,checkpoint_sha256='d'*64),'model',ledger_path=ledger,permit=second)
        with closing(sqlite3.connect(ledger,isolation_level=None)) as db:
            db.execute("UPDATE evaluations SET status='complete' WHERE split='development'")
        suite=synthetic_suite();suite['split']='acceptance_correction'
        patched=dict(PINNED_HASHES,**{'acceptance_correction.json':hashlib.sha256(json.dumps(suite,indent=2).encode()).hexdigest()})
        permit={'phase':'acceptance_correction','selected_checkpoint_sha256':'b'*64,
                'selected_by':'synthetic','selection_rationale':'test','development_evidence':'synthetic',
                'human_fixture_review_evidence':'synthetic','protocol_sha256':PINNED_HASHES['protocol.json'],
                'primary_run_evidence':'synthetic'}
        with patch.dict('nanochat.conversation_eval.PINNED_HASHES',patched):
            authorize(suite,protocol,identity,'model',ledger_path=ledger,permit=permit)
            with self.assertRaises(PermissionError):authorize(suite,protocol,identity,'model',ledger_path=ledger,permit=permit)
        with self.assertRaises(PermissionError):authorize(dev,protocol,identity,'model',ledger_path=ledger,permit=second)

    def test_native_greedy_adapter_scripted_cpu_logits_only(self):
        import torch
        sequence=self.codec.tokenizer.encode('scripted')+[self.codec.end]
        class ScriptedLogits(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0;self.seen=[]
            def get_device(self):return torch.device('cpu')
            def forward(self,ids):
                self.seen.append(ids.tolist()[0]);out=torch.full((1,ids.shape[1],1024),-100.)
                out[0,-1,sequence[self.calls % len(sequence)]]=100.;self.calls+=1;return out
        model=ScriptedLogits()
        backend=NativeGreedyBackend(model,synthetic=True)
        output=self.folder/'native-scripted.jsonl'
        artifact=run_suite(self.suite,self.protocol,self.codec,backend,self.identity,output_path=output)
        self.assertEqual(artifact['records'][0]['raw_ids'],sequence)
        self.assertEqual(model.calls,180*len(sequence))
        self.assertFalse(model.training)
        self.assertEqual(model.seen[1],model.seen[0]+[sequence[0]])

    def test_native_nonfinite_logits_fail_not_silent_argmax(self):
        import torch
        class Bad(torch.nn.Module):
            def get_device(self):return torch.device('cpu')
            def forward(self,ids):return torch.full((1,ids.shape[1],1024),float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            next(NativeGreedyBackend(Bad(),synthetic=True).stream([1],256))

    @unittest.skipUnless((ROOT/'tokenizer/tokenizer.pkl').exists(), 'requires pinned tokenizer artifact')
    def test_pinned_real_tokenizer_matches_native_chat_template_without_model(self):
        codec=ChatCodec.from_directory(ROOT/'tokenizer')
        tokenizer=codec.tokenizer
        self.assertEqual(codec.end,32763)
        history=[{'turn':0,'user':'Synthetic earlier user.',
                  'assistant_ids':tokenizer.encode('Scripted earlier assistant.')+[codec.end]}]
        actual,kept,removed=codec.prompt(history,'Synthetic current user.')
        expected,_=tokenizer.render_conversation({'messages':[
            {'role':'user','content':'Synthetic earlier user.'},
            {'role':'assistant','content':'Scripted earlier assistant.'},
            {'role':'user','content':'Synthetic current user.'}]})
        expected.append(tokenizer.encode_special('<|assistant_start|>'))
        self.assertEqual(actual,expected)
        self.assertEqual(removed,[])

    def test_missing_human_outputs_cannot_be_hidden_by_empty_failure_labels(self):
        artifact,_=self.run_fake()
        keys={(self.suite['examples'][80]['id'],0),(self.suite['examples'][81]['id'],0)}
        artifact['records']=[r for r in artifact['records'] if (r['case_id'],r['turn']) not in keys]
        result=self.score(artifact,reviews=self.reviewed(artifact))
        self.assertEqual(result['gates']['human_usefulness']['status'],'fail')
        self.assertEqual(result['gates']['human_usefulness']['failure_categories']['missing_or_failed_response'],2)

    def test_cli_smoke_score_and_review_template_are_synthetic_only(self):
        from contextlib import redirect_stdout
        import io
        from scripts.conversation_eval import main
        folder=self.folder/'cli-smoke'
        with redirect_stdout(io.StringIO()):
            main(['synthetic-smoke','--suite-dir',str(ROOT/'conversation_evals_v3'),
                  '--output-dir',str(folder)])
            main(['score','--suite-dir',str(folder),'--records',str(folder/'synthetic-records.jsonl'),
                  '--output',str(folder/'rescored.json'),'--review-template-output',str(folder/'human-review.json')])
        result=json.loads((folder/'rescored.json').read_text())
        review=json.loads((folder/'human-review.json').read_text())
        self.assertEqual(result['release_decision'],'synthetic_not_release_evidence')
        self.assertEqual(result['gates']['human_usefulness']['status'],'pending')
        self.assertEqual(len(review['responses']),180)
        self.assertEqual(len(review['human_conversations']),20)
        with self.assertRaises(FileExistsError):
            main(['synthetic-smoke','--suite-dir',str(ROOT/'conversation_evals_v3'),'--output-dir',str(folder)])

    def test_human_reviews_cannot_hide_repeated_missing_conversations(self):
        artifact,_=self.run_fake()
        missing={c['id'] for c in self.suite['examples'] if c['capability']=='human_review'}
        missing=set(sorted(missing)[:2])
        artifact['records']=[r for r in artifact['records'] if r['case_id'] not in missing]
        reviews=self.reviewed(artifact)
        result=self.score(artifact,reviews=reviews)
        self.assertEqual(result['gates']['human_usefulness']['status'],'fail')

    def test_record_token_limit_cannot_be_bypassed_by_tampering(self):
        artifact,_=self.run_fake()
        row=artifact['records'][0]
        row['raw_ids']=[ord('x')]*256+[self.codec.end]
        row['text']=self.codec.text(row['raw_ids'])
        self.assertFalse(self.score(artifact)['response_results'][0]['loop_free'])


if __name__=='__main__':unittest.main(verbosity=2)