import json
from pathlib import Path
import unittest
from remediation_checks import inspect

def exchange(prompt, answer, system=None):
    return ([dict(role='system', content=system)] if system else []) + [dict(role='user',content=prompt),dict(role='assistant',content=answer)]

class ContentChecks(unittest.TestCase):
    def kinds(self, m): return {f['kind'] for f in inspect(m)[0]}
    def test_literal_not_plural(self):
        self.assertIn('literal_word_minimum', self.kinds(exchange('Use the word "resource" at least 3 times.','resource resources resources')))
    def test_literal_case_punctuation(self):
        self.assertFalse(self.kinds(exchange('Use the word "resource" at least 3 times.','Resource, resource; RESOURCE.')))
    def test_word_boundary(self):
        self.assertFalse(self.kinds(exchange('Use the word "the" at least 1 times.','the')))
        self.assertIn('literal_word_minimum', self.kinds(exchange('Use the word "the" at least 1 times.','there')))
    def test_pronouns_and_possessives(self):
        for word in ['He','She','his','its',"It's",'your']:
            self.assertIn('pronoun_constraint_risk',self.kinds(exchange('Summarize.',word,'without using second or third person pronouns')))
    def test_pronoun_substrings(self):
        self.assertFalse(self.kinds(exchange('Summarize.','Miriam confirms the meeting.','without using second or third person pronouns')))
    def test_unrestricted_pronouns(self):
        self.assertFalse(self.kinds(exchange('Say hello.','She says hello.')))
    def test_word_limits(self):
        self.assertIn('strict_max_words',self.kinds(exchange('Your response should contain less than 3 words.','a b c')))
        self.assertIn('min_words',self.kinds(exchange('Your response should contain at least 3 words.','a b')))
    def test_postscript(self):
        self.assertIn('missing_postscript',self.kinds(exchange('Include a postscript starting with "P.S."','Hello.')))
        self.assertFalse(self.kinds(exchange('Include a postscript starting with "P.S."','Hello.\n\nP.S. Bye.')))
    def test_inline_and_lowercase_postscripts_are_not_false_flags(self):
        for answer in ['Hello. P.S. Bye.', 'hello. p.s. bye.', '"hello.\n\np.s. bye."']:
            self.assertFalse(self.kinds(exchange('Include a postscript starting with "P.S."',answer)))
    def test_transit_family(self):
        self.assertIn('unverified_local_transit_assertion',self.kinds(exchange('New subway lines in my city?','Two lines are under construction.')))
        self.assertFalse(self.kinds(exchange('New subway lines in my city?','Which city do you mean?')))
    def test_fish_family(self):
        self.assertIn('incomplete_fish_problem_family',self.kinds(exchange('Olga has an aquarium with fish in 5 different colors.','50')))
    def test_repetition(self):
        self.assertIn('repetitive_fourgram',self.kinds(exchange('Hello.','keep looping these words '*10)))
    def test_known_packet_and_false_positive_regressions(self):
        rows=json.loads((Path(__file__).parent/'readiness_review_v1/filtered-dataset-human-review-sample.json').read_text())['examples']
        for index in [13,20,32,49,69,87,89,90,93,94,98,99]:
            self.assertTrue(self.kinds(rows[index-1]['messages']),f'known defect family {index}')
        for index in [51,57,59,78]:
            self.assertFalse(self.kinds(rows[index-1]['messages']),f'known overturned false positive {index}')
        # Packet 88 is explicitly excluded by policy even though this conservative
        # lexical scan does not reproduce the earlier possessive interpretation.

if __name__ == '__main__': unittest.main()