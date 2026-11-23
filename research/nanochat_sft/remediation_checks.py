"""Deterministic content-risk checks. Flags are quarantine reasons, not truth judgments.

Only literal generated instruction forms are interpreted. No model calls, network,
holdout access, answer rewriting, or generic semantic correctness claims.
"""
import re
from collections import Counter

PRONOUNS = re.compile(r"\b(?:you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves)\b", re.I)
SENTENCES = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d"“])')


def inspect(messages):
    flags, checked = [], Counter()
    system = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
    for i, msg in enumerate(messages):
        if msg['role'] != 'assistant':
            continue
        answer = msg['content']
        prompt = messages[i-1]['content'] if i else ''
        def flag(kind, **evidence):
            flags.append(dict(kind=kind, assistant_turn=i, **evidence))
        checked['assistant_turns'] += 1
        if not answer.strip():
            flag('empty_answer')
        words = re.findall(r'\w+', answer.casefold())
        if len(words) >= 30:
            counts = Counter(tuple(words[j:j+4]) for j in range(len(words)-3))
            if max(counts.values(), default=0) >= 8:
                flag('repetitive_fourgram', maximum=max(counts.values()))
        if re.search(r'without using second or third person pronouns', system, re.I):
            checked['pronoun_prohibition'] += 1
            found = sorted(set(x.casefold() for x in PRONOUNS.findall(answer)))
            if found:
                flag('pronoun_constraint_risk', tokens=found,
                     interpretation='Conservative lexical exclusion; includes possessives, quotations and contractions.')
        if 'up to three sentences' in system:
            checked['summary_sentence_limit'] += 1
            n = len([x for x in SENTENCES.split(answer.strip()) if x.strip()])
            if n > 3:
                flag('summary_sentence_limit_risk', observed=n, maximum=3)
        # Apply only exact generated directive forms, not arbitrary prose mentioning a constraint.
        for pattern, kind in [
            (r'Your response should contain less than (\d+) words\.', 'strict_max_words'),
            (r'Your response should contain at least (\d+) words\.', 'min_words')]:
            for match in re.finditer(pattern, prompt):
                checked[kind] += 1
                limit, observed = int(match[1]), len(answer.split())
                if (kind == 'strict_max_words' and observed >= limit) or (kind == 'min_words' and observed < limit):
                    flag(kind, observed=observed, limit=limit, counting='whitespace')
        for match in re.finditer(r'Use the word ["“]([\w-]+)["”] at least (\d+) times\.', prompt):
            checked['literal_word_minimum'] += 1
            word, minimum = match[1], int(match[2])
            n = len(re.findall(r'(?<!\w)' + re.escape(word) + r'(?!\w)', answer, flags=re.I))
            if n < minimum:
                flag('literal_word_minimum', word=word, observed=n, minimum=minimum)
        if 'Include a postscript starting with "P.S."' in prompt:
            checked['postscript'] += 1
            # Neither a separate line nor capitalization is inferred. Some
            # generated tasks explicitly require all-lowercase output as well.
            if not re.search(r'(?<!\w)p\.s\.', answer, re.I):
                flag('missing_postscript')
        # Risk families discovered in the rejected packet, not general fact checking.
        if re.search(r'\b(?:my|our) city\b', prompt, re.I) and re.search(r'\b(?:subway|metro|transit)\b', prompt, re.I):
            checked['unspecified_local_transit_risk'] += 1
            if re.search(r'\b(?:under construction|currently being built|expected to open|will (?:connect|extend))\b', answer, re.I):
                flag('unverified_local_transit_assertion', interpretation='Conservative risk hold, not proof that no location was supplied elsewhere.')
        if re.search(r'aquarium with fish in 5 different colors', prompt, re.I):
            checked['incomplete_fish_problem_family'] += 1
            flag('incomplete_fish_problem_family', interpretation='Known underdetermined problem family; quarantine rather than repair labels.')
        terms = ['Grandparents Visitation Rights Act', "Coursera's Technical Leadership Specialization", "Leadership Circle's Technical Leadership Program"]
        for term in terms:
            if term.casefold() in answer.casefold():
                checked['unverified_named_claim_family'] += 1
                flag('unverified_named_claim_family', term=term)
    return flags, checked