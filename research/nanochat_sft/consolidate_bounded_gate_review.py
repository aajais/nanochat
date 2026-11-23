"""Record this bounded inspection without changing any training/evaluation input."""
import hashlib,json,re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'bounded_gate_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2))
sample=ROOT/'readiness_review_v1/filtered-dataset-human-review-sample.json'
examples=json.loads(sample.read_text())['examples']
raw={r['id']:r for f in sorted(OUT.glob('sample-batch-*.json')) for r in json.loads(f.read_text())['reviews']}
assert len(raw)==80
# The 20 rows whose batch evidence validation failed were inspected directly
# by ARIA, not regenerated and not reviewed/approved by a human.
overrides={
13:('reject','Unsupported local facts','Invents two subway lines and 2025/2027 completion dates without being given a city.'),
20:('uncertain','Unverified jurisdiction-specific law','Names a US Grandparents Visitation Rights Act without a jurisdiction or verified source; owner/legal-content review required.'),
51:('pass','Automated false positive overturned','The full answer has three sentences, two placeholders and two italicized important tokens; copied wording is not an explicit violation.'),
57:('pass','Automated false positive overturned','The answer exceeds 300 words, includes multiple italic and bold sections, and ends with the required phrase.'),
59:('pass','Automated false positive overturned','No whole-word the appears; there is not the prohibited word. The answer is under 50 words and ends correctly.'),
78:('pass','Automated false positive overturned','The user requests transformative fiction that explicitly changes game events; canon deviation alone is not a blocking factual defect.'),
88:('uncertain','Pronoun-policy interpretation','The possessive form its appears under a no-third-person-pronouns requirement; retain as an interpretation hold rather than asserting an unequivocal defect.'),
93:('reject','Explicit pronoun constraint violated','Uses He and his despite the system prohibition on second/third-person pronouns.'),
94:('reject','Explicit pronoun constraint violated','Uses She despite the system prohibition on second/third-person pronouns.'),
98:('reject','Explicit pronoun constraint violated','Uses She despite the system prohibition on second/third-person pronouns.'),
99:('reject','Explicit pronoun constraint violated','Uses He despite the system prohibition on second/third-person pronouns.'),
}
adjudicated=[]
for i,e in enumerate(examples,1):
    prior=raw.get(e['id'])
    record={'packet_index':i,'id':e['id'],'source':e['source'],'source_row':e['source_row'],'split':e['split'],
            'reviewer':'ARIA AI-assisted inspection; not a human reviewer',
            'initial_model_review':prior,
            'review_mode':'validated nested-model review plus ARIA flag inspection' if prior else 'ARIA direct source-text inspection after invalid batch evidence',
            'decision':prior['decision'] if prior else 'pass',
            'rationale':prior['rationale'] if prior else 'No material defect found in direct inspection; ordinary sampled-target usability only.'}
    if i in overrides:
        record['decision'],record['defect_or_override'],record['rationale']=overrides[i]
    if record['decision']=='reject':
        record['proposed_action']='Exclude this retained candidate ID before any future approved cache rebuild; existing v2 is unchanged and entirely training-blocked.'
    if record['decision']=='uncertain':record['proposed_action']='Owner/content adjudication or conservative exclusion required before approval.'
    adjudicated.append(record)
counts=dict(Counter(r['decision'] for r in adjudicated))
assert counts=={'pass':87,'reject':10,'uncertain':3},counts
# Recheck mechanical claims used to overrule the model, not substring guesses.
answers={i:'\n'.join(m['content'] for m in e['messages'] if m['role']=='assistant') for i,e in enumerate(examples,1)}
mechanical={
    'row_49_exact_resource_count':len(re.findall(r'\bresource\b',answers[49],flags=re.I)),
    'row_51_sentences':len([x for x in re.split(r'(?<=[.!?])\s+',answers[51]) if x]),
    'row_51_important_count':len(re.findall(r'\bimportant\b',answers[51],flags=re.I)),
    'row_57_whitespace_words':len(answers[57].split()),
    'row_57_markdown_italic_headings':len(re.findall(r'^\*[^*\n]+\*\s*$',answers[57],flags=re.M)),
    'row_57_correct_suffix':answers[57].endswith('I hope this information is helpful.'),
    'row_59_whole_word_the_count':len(re.findall(r'\bthe\b',answers[59],flags=re.I)),
    'row_59_whitespace_words':len(answers[59].split()),
    'row_59_correct_suffix':answers[59].endswith('Is there anything else I can help with?'),
}
assert mechanical['row_49_exact_resource_count']==1
assert mechanical['row_51_sentences']==3 and mechanical['row_51_important_count']==2
assert mechanical['row_57_whitespace_words']>=300 and mechanical['row_57_markdown_italic_headings']>=2 and mechanical['row_57_correct_suffix']
assert mechanical['row_59_whole_word_the_count']==0 and mechanical['row_59_whitespace_words']<50 and mechanical['row_59_correct_suffix']
for i in [87,89,90,93,94,98,99]:assert re.search(r'\b(?:he|she)\b',answers[i],flags=re.I)
write('sample-adjudication.json',{'scope':'100-example retained-training-data review only; no model-under-test generation',
    'sample_sha256':sha(sample),'counts':counts,'rows':adjudicated,'mechanical_crosschecks':mechanical,
    'automated_batches':10,'validated_batches':8,'invalid_batches':[1,9],
    'invalid_batch_handling':'No retry; all 20 affected examples directly inspected by ARIA. Invalid batch outputs were not retained or relied on.',
    'gate':'FAIL','independent_human_approval':False,
    'limitations':['AI-assisted inspection, not independent human sign-off or an exhaustive factual audit.',
      '20 examples per source: stratified review, not token-weighted representative sampling. Counts are not estimated corpus defect rates.',
      'Model flags included four false positives overturned from source text; unvalidated judge outputs cannot be used as release approval.']})
write('proposed-additional-exclusions.json',{'cache_mutated':False,'training_approved':False,
    'hard_rejects':[r for r in adjudicated if r['decision']=='reject'],
    'review_holds':[r for r in adjudicated if r['decision']=='uncertain'],
    'existing_1166_quarantine':'Retained unchanged. These are additional findings in the currently retained set, not replacements for the old quarantine.'})

# Static fixture inspection only: references are specifications, not generations.
newdir=ROOT/'readiness_bundle_v1/conversation_evals_v3'
olddir=ROOT/'conversation_evals_v2'
audit=json.loads((newdir/'audit.json').read_text())
fixture_results=[]
for split in ['development','acceptance_primary','acceptance_correction']:
    new=json.loads((newdir/f'{split}.json').read_text())
    old=json.loads((olddir/f'{split}.json').read_text())
    prev={e['id']:e for e in old['examples']}
    changed=[]
    for e in new['examples']:
        assert e['turn_checks']==prev[e['id']]['turn_checks']
        if e['messages']!=prev[e['id']]['messages']:changed.append(e['id'])
        if e['family']=='exclude_distractors':
            values=re.findall(r'(?:^|[;:])\s*object=([^;.]+)',e['messages'][0]['content'])
            assert len(values)==2,(split,e['id'],values)
            assert e['turn_checks'][0]['value']==', '.join(v.strip() for v in values)
        if e['family']=='reference_resolution':
            text=e['messages'][0]['content']
            direction=re.search(r'on the (\w+) shelf',text).group(1)
            assert e['turn_checks'][1]['value']==direction+' shelf'
            assert 'exactly two words' in e['messages'][1]['content']
            assert 'Do not include an article or punctuation.' in e['messages'][1]['content']
        assert len(e['messages'])==len(e['turn_checks'])
    expected={x['case_id'] for x in audit['changed_cases'] if x['split']==split}
    assert set(changed)==expected and len(changed)==8
    fixture_results.append({'split':split,'conversations':len(new['examples']),
         'responses':sum(len(e['messages']) for e in new['examples']),
         'changed_case_definitions_checked':len(changed),'all_reference_checks_unchanged':True,
         'sha256':sha(newdir/f'{split}.json')})
assert [r['responses'] for r in fixture_results]==[180]*3
write('fixture-definition-review.json',{'gate':'PASS for targeted v3 wording repairs only; human review remains pending',
    'scope':'Static checks on 24 changed case definitions plus unchanged-reference checks across 300 conversations; no generated responses.',
    'checks':['Explicit object labels remove noun/color ambiguity.',
       'Explicit DIRECTION shelf two-word contract matches the unchanged strict reference.',
       'No reference answer changes across the v2-to-v3 wording repairs.'],
    'splits':fixture_results,'independent_human_approval':False,'acceptance_generations':0,
    'limitations':['Shared templates across splits limit generalization claims.',
       'Exact match intentionally rejects extra words/punctuation; not a broad semantic quality measure.',
       'Human-scored turns and the human-demo gate still require a human reviewer.']})
cache=ROOT/'conversation_cache_quality_v1'
manifest=json.loads((cache/'manifest.json').read_text())
assert manifest['release_gate']['training_approved'] is False
assert manifest['quality_quarantine']['exclusions']==1166
for f,d in manifest['files'].items():assert sha(cache/f)==d
write('input-integrity.json',{'candidate_ref':'wandb/nanochat-demo/conversation-sft-cache-candidate:v2',
      'files':manifest['files'],'manifest_sha256':sha(cache/'manifest.json'),
      'quarantine_ids_sha256':sha(cache/'quarantine-ids.json'),'original_quarantine_count':1166,
      'training_approved':False,'candidate_files_unchanged':True})
print(json.dumps({'sample_counts':counts,'content_gate':'FAIL','fixture_changed_cases_checked':24,
                  'training_approved':False,'gpu_jobs_launched':0}),flush=True)