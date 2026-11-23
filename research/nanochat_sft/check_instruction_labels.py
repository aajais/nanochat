"""Audit only explicit whitespace-word bounds and flag sentence-limit risks."""
import hashlib,json,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prepared'))
from scripts.prepare_conversation_sft import normalize,text_of
out=ROOT/'readiness_review_v1'
ids=set()
for split in ('train','dev'):
    ids.update(json.loads(line)['id'] for line in (ROOT/'conversation_cache_protected_v1'/(split+'.jsonl')).open())
flags=[];seen=set();checked=Counter();systems=Counter()
for line in (ROOT/'conversation_candidates/selected.jsonl').open():
    row=json.loads(line);key=hashlib.sha256(normalize(text_of(row['messages'],True)).encode()).hexdigest()
    if key not in ids or key in seen:continue
    seen.add(key);component=row['source'].rsplit('/',1)[1]
    for i,msg in enumerate(row['messages']):
        if msg['role']!='assistant':continue
        prompt=row['messages'][i-1]['content'];answer=msg['content'];words=len(answer.split())
        for pattern,kind in ((r'Your response should contain less than (\d+) words\.', 'strict_max_words'),(r'Your response should contain at least (\d+) words\.', 'min_words')):
            match=re.search(pattern,prompt)
            if match:
                checked[kind]+=1;limit=int(match[1])
                if (kind=='strict_max_words' and words>=limit) or (kind=='min_words' and words<limit):
                    flags.append(dict(id=key,source=component,assistant_turn=i,kind=kind,word_count=words,limit=limit,prompt=prompt,answer=answer,status='high_confidence_under_whitespace_word_count'))
        if component=='smol-summarize':
            system=next((m['content'] for m in row['messages'] if m['role']=='system'),'')
            systems[system]+=1
            if 'up to three sentences' in system:
                checked['summary_sentence_limit']+=1
                count=len([s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\d"“])',answer.strip()) if s.strip()])
                if count>3:flags.append(dict(id=key,source=component,assistant_turn=i,kind='summary_sentence_limit_heuristic',sentence_count=count,limit=3,prompt=system,answer=answer,status='human_adjudication_required_abbreviations_and_formatting'))
result=dict(checked=dict(checked),flags_by_kind=dict(Counter(x['kind'] for x in flags)),unique_flagged_conversations=len({x['id'] for x in flags}),summary_systems=dict(systems),flags=flags,human_approval=False,training_approved=False)
(out/'instruction-label-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('flags','summary_systems')},indent=2))
for kind in sorted({x['kind'] for x in flags}):
    row=next(x for x in flags if x['kind']==kind)
    print(json.dumps(row)[:4200])