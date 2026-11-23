"""Fixture/protection tests only: not evaluation of model capability."""
import json,sys,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
try:
    from freeze_conversation_evals import BANKS,make_suite
except ModuleNotFoundError:
    BANKS = make_suite = None  # Protected authoring banks are not public source.
from nanochat.sft_runtime import sha256
from scripts.prepare_conversation_sft import normalize,prepare


@unittest.skipUnless(BANKS is not None and (ROOT/'conversation_evals_v3/acceptance_primary.json').exists()
                     and (ROOT/'tokenizer/tokenizer.pkl').exists(), 'requires private fixture authoring bank and tokenizer')
class EvalFreezeTests(unittest.TestCase):
    def test_suite_shapes_roles_and_disjoint_prompts(self):
        seen=set()
        for split,bank in BANKS.items():
            suite=make_suite(split,bank)
            counts={c:sum(r['capability']==c for r in suite['examples'])
                    for c in ('instruction','multi_turn','grounded','human_review')}
            self.assertEqual(counts,{'instruction':40,'multi_turn':20,'grounded':20,'human_review':20})
            self.assertEqual(sum(len(r['messages']) for r in suite['examples']),180)
            self.assertEqual(len({r['id'] for r in suite['examples']}),100)
            for row in suite['examples']:
                self.assertEqual(len(row['messages']),len(row['turn_checks']))
                for message in row['messages']:
                    self.assertEqual(set(message),{'role','content'})
                    self.assertEqual(message['role'],'user')
                    key=normalize(message['content'])
                    self.assertNotIn(key,seen)
                    seen.add(key)

    def test_frozen_files_and_fixed_denominators(self):
        folder=ROOT/'conversation_evals_v3'
        hashes=json.loads((folder/'SHA256.json').read_text())
        for name,digest in hashes.items():self.assertEqual(sha256(folder/name),digest)
        protocol=json.loads((folder/'protocol.json').read_text())
        self.assertEqual(protocol['denominators']['all_responses'],180)
        self.assertEqual(protocol['gates']['responses_without_severe_loops_min'],171)
        self.assertEqual(protocol['gates']['correct_boundaries_min'],171)
        self.assertEqual(protocol['gates']['arc100_review_at_or_below'],49)

    def test_embedded_protected_user_prompt_excluded(self):
        phrase='The unique amber observatory experiment recorded seventeen brass instruments beside its northern window'
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            folder=Path(name)
            source=folder/'input.jsonl'
            records=[{'messages':[{'role':'user','content':f'Return unique marker {i:08x}'},
                                   {'role':'assistant','content':f'{i:08x}'}],
                      'category':'general','source':'test-only','revision':'fixture-v1'} for i in range(120)]
            records.append({'messages':[{'role':'user','content':('irrelevant surrounding context '*80)+phrase+(' more unrelated details '*80)},
                                         {'role':'assistant','content':'must not be kept'}],
                            'category':'general','source':'test-only','revision':'fixture-v1'})
            source.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
            protected=folder/'protected.json'
            protected.write_text(json.dumps({'examples':[{'messages':[{'role':'user','content':phrase}]}]}))
            result=prepare([source],folder/'output',ROOT/'tokenizer',[protected])
            self.assertEqual(result['filter_counts']['protected_overlap'],1)
            self.assertEqual(result['filter_counts']['kept'],120)
            self.assertFalse(result['release_gate']['training_approved'])

if __name__=='__main__':unittest.main(verbosity=2)