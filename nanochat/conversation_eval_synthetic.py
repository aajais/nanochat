"""Explicitly fabricated fixtures and replay backend; never model-quality data."""
from nanochat.conversation_eval import CONTROL_NAMES


class CharacterTokenizer:
    def __init__(self):
        self.special = {name: 512 + i for i, name in enumerate(CONTROL_NAMES)}
    def encode_special(self, name): return self.special[name]
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, ids):
        inverse = {v: k for k, v in self.special.items()}
        return ''.join(inverse[i] if i in inverse else chr(i) for i in ids)
    def get_vocab_size(self): return 1024


class ScriptedBackend:
    synthetic = True
    def __init__(self, codec, custom=None):
        self.codec, self.custom, self.prompts, self.calls = codec, custom or {}, [], 0
    def stream(self, prompt_ids, max_new_tokens):
        self.prompts.append(list(prompt_ids))
        index = self.calls
        self.calls += 1
        response = self.custom.get(index, self.codec.tokenizer.encode('synthetic-answer') + [self.codec.end])
        if isinstance(response, Exception): raise response
        for token in response:
            if isinstance(token, Exception): raise token
            yield token


def synthetic_suite():
    examples = []
    for cap, count, turns in [('instruction', 40, 1), ('multi_turn', 20, 3),
                              ('grounded', 20, 1), ('human_review', 20, 3)]:
        for i in range(count):
            examples.append({'id': f'synthetic-{cap}-{i}', 'capability': cap,
                'messages': [{'role': 'user', 'content': f'Synthetic {cap} {i} turn {t}.'} for t in range(turns)],
                'turn_checks': [({'kind': 'human', 'criteria': ['Synthetic criterion only.']}
                                 if cap == 'human_review' else {'kind': 'exact', 'value': 'synthetic-answer'}) for _ in range(turns)]})
    return {'schema': 'scripted-user-turn-evaluation-v1', 'split': 'synthetic', 'examples': examples}


class ScriptedBackend:
    synthetic = True
    def __init__(self, codec, custom=None):
        self.codec, self.custom, self.prompts, self.calls = codec, custom or {}, [], 0
    def stream(self, prompt_ids, max_new_tokens):
        self.prompts.append(list(prompt_ids))
        index = self.calls
        self.calls += 1
        response = self.custom.get(index, self.codec.tokenizer.encode('synthetic-answer') + [self.codec.end])
        if isinstance(response, Exception): raise response
        for token in response:
            if isinstance(token, Exception): raise token
            yield token