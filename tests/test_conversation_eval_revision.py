"""v3 fixture integrity tests only. Never generate any frozen-suite outputs."""
import json,hashlib,unittest
from pathlib import Path
from nanochat.conversation_eval import PINNED_HASHES,load_frozen,validate_suite
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'conversation_evals_v1'
NEW=ROOT/'conversation_evals_v3'

@unittest.skipUnless((OLD/'acceptance_primary.json').exists() and (NEW/'acceptance_primary.json').exists(),
                     'requires private frozen fixture history')
class RevisionTests(unittest.TestCase):
    def test_all_pins_and_suite_denominators(self):
        for name,value in PINNED_HASHES.items():
            self.assertEqual(hashlib.sha256((NEW/name).read_bytes()).hexdigest(),value)
        for split in ('development','acceptance_primary','acceptance_correction'):
            suite,protocol=load_frozen(NEW,split)
            validate_suite(suite)
            self.assertEqual(sum(len(r['messages']) for r in suite['examples']),180)
            self.assertEqual(protocol['gates']['instruction_min_pass'],32)
            self.assertEqual(protocol['gates']['correct_boundaries_min'],171)

    def test_old_pins_rejected(self):
        with self.assertRaisesRegex(ValueError,'Frozen identity mismatch'):
            load_frozen(OLD,'development')

    def test_eight_cases_twelve_prompts_per_split_and_references_unchanged(self):
        for split in ('development','acceptance_primary','acceptance_correction'):
            old=json.loads((OLD/(split+'.json')).read_text())
            new=json.loads((NEW/(split+'.json')).read_text())
            changes=[]
            for before,after in zip(old['examples'],new['examples']):
                self.assertEqual(before['id'],after['id'])
                self.assertEqual(before['turn_checks'],after['turn_checks'])
                for i,(a,b) in enumerate(zip(before['messages'],after['messages'])):
                    if a!=b:changes.append((after['family'],i))
            self.assertEqual(sorted(changes),sorted([('exclude_distractors',0)]*4+[('reference_resolution',1)]*4+[('reference_resolution',2)]*4))

    def test_development_exact_prompts_match_output_contract(self):
        suite,_=load_frozen(NEW,'development')
        for row in suite['examples']:
            if row['family']=='exclude_distractors':
                self.assertIn('values explicitly labeled object',row['messages'][0]['content'])
            if row['family']=='reference_resolution':
                self.assertIn('exactly two words',row['messages'][1]['content'])
                self.assertEqual(len(row['turn_checks'][1]['value'].split()),2)

    def test_thresholds_and_decoding_unchanged(self):
        a=json.loads((OLD/'protocol.json').read_text());b=json.loads((NEW/'protocol.json').read_text())
        for key in ('gates','decoding','denominators','scoring','selection_policy'):
            self.assertEqual(a[key],b[key])

if __name__=='__main__':unittest.main()