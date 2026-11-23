"""Prepare offline, tokenized SFT splits from version-pinned local JSONL.

Input record: {messages:[{role,content}], category:general|context|reasoning,
               source:string, revision:immutable-version}.
Categories are supplied by curated source metadata, not guessed from model calls.
No datasets are downloaded by this script. No acceptance data is generated here.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from nanochat.sft_runtime import sha256, atomic_json


def normalize(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def text_of(messages, prompts_only=False):
    return '\n'.join(m['content'] for m in messages if not prompts_only or m['role'] != 'assistant')


@lru_cache(maxsize=8192)
def fingerprint(text):
    words = normalize(text).split()
    shingles = [' '.join(words[i:i+5]) for i in range(max(1,len(words)-4))]
    # Same bit-for-bit SimHash as the reference implementation, without the
    # Python loop over 64 bits for every shingle in every document.
    hashes=b''.join(hashlib.blake2b(s.encode(),digest_size=8).digest() for s in set(shingles))
    bits=np.unpackbits(np.frombuffer(hashes,dtype=np.uint8)).reshape(-1,64)
    positive=bits.sum(axis=0)*2>=len(bits)
    return int.from_bytes(np.packbits(positive).tobytes(),'big')


class NearIndex:
    """Heuristic near dedup: 64-bit 5-word SimHash, Hamming distance <=3.

    Four exact 16-bit bands give candidate recall for that Hamming threshold.
    This is not a guarantee of semantic deduplication.
    """
    def __init__(self):
        self.exact = set()
        self.bands = defaultdict(list)

    def contains(self,text):
        text = normalize(text)
        if text in self.exact:
            return True
        if len(text.split()) < 10:
            return False  # Short prompts use exact matching; SimHash too coarse.
        fp = fingerprint(text)
        candidates = set()
        for i in range(4):
            candidates.update(self.bands[(i,(fp>>(16*i)) & 65535)])
        return any((fp^other).bit_count()<=3 for other in candidates)

    def add(self,text):
        text = normalize(text)
        self.exact.add(text)
        if len(text.split()) >= 10:
            fp = fingerprint(text)
            for i in range(4):
                self.bands[(i,(fp>>(16*i)) & 65535)].append(fp)


def validate_messages(messages):
    if not isinstance(messages,list) or not messages:
        return 'missing_messages'
    if any(not isinstance(m,dict) for m in messages):
        return 'malformed_message'
    start = int(messages[0].get('role')=='system')
    rest = messages[start:]
    if len(rest)<2 or len(rest)%2:
        return 'incomplete_conversation'
    for i,m in enumerate(messages):
        if not isinstance(m.get('content'),str) or not m['content'].strip():
            return 'empty_or_tool_content'
        if any(t in m['content'] for t in SPECIAL_TOKENS):
            return 'embedded_control_token'
    for i,m in enumerate(rest):
        if m.get('role') != ('user' if i%2==0 else 'assistant'):
            return 'invalid_roles'
        if m['role']=='assistant':
            words=normalize(m['content']).split()
            if len(words)>=30:
                grams=[' '.join(words[j:j+4]) for j in range(len(words)-3)]
                if max(Counter(grams).values())>=8:
                    return 'repetitive_answer'
    return None


def prepare(inputs, output, tokenizer_dir, protected_files, context=2048, dev_fraction=.05, progress=None):
    if not protected_files:
        raise ValueError('Protected ARC + development + both acceptance manifests are required')
    tokenizer=RustBPETokenizer.from_directory(tokenizer_dir)
    banned=NearIndex()
    protected_spans=set()
    for path in protected_files:
        payload=json.loads(Path(path).read_text())
        # ARC snapshot uses examples/query. Conversation manifests use examples/messages.
        for row in payload['examples']:
            if 'messages' in row:
                combined=text_of(row['messages'],True)
                banned.add(combined)
                for m in row['messages']:
                    if m['role'] in ('user','system'):
                        banned.add(m['content'])
                        norm=normalize(m['content'])
                        if len(norm.split())>=12:protected_spans.add(norm)
            elif 'query' in row:
                banned.add(row['query'])
                banned.add(re.sub(r'^Question:\s*','',row['query']))
                norm=normalize(re.sub(r'^Question:\s*','',row['query']))
                if len(norm.split())>=12:protected_spans.add(norm)
            else:
                raise ValueError('Unsupported protected example schema')
    seen=NearIndex()
    seen_prompts=NearIndex()
    kept=[]
    counters=Counter()
    provenance=set()
    for path in inputs:
        with open(path) as f:
            for row_number,line in enumerate(f,1):
                if progress and row_number%500==0:
                    progress({'rows_scanned':row_number,'filter_counts':dict(counters)})
                row=json.loads(line)
                messages=row.get('messages')
                reason=validate_messages(messages)
                if reason:
                    counters[reason]+=1
                    continue
                if not row.get('source') or not row.get('revision') or row['revision'] in ('main','latest','master'):
                    raise ValueError('Each source requires an immutable revision')
                category=row['category']
                if category not in {'general','context','reasoning'}:
                    raise ValueError('Unknown category')
                prompt=text_of(messages,True)
                full=text_of(messages)
                # Exclude examples containing any protected user prompt, not just full-dialogue matches.
                prompt_norm=normalize(prompt)
                if (banned.contains(prompt) or
                    any(banned.contains(m['content']) for m in messages if m['role'] in ('user','system')) or
                    any((' '+span+' ') in (' '+prompt_norm+' ') for span in protected_spans)):
                    counters['protected_overlap']+=1
                    continue
                if seen.contains(full) or seen_prompts.contains(prompt):
                    counters['duplicate_or_near_duplicate']+=1
                    continue
                # Render beyond context first; never slice an assistant answer or its EOT.
                ids,mask=tokenizer.render_conversation({'messages':messages},max_tokens=10_000_000)
                if len(ids)>context+1 or len(ids)<2:
                    counters['over_context_or_empty']+=1
                    continue
                if ids[-1]!=tokenizer.encode_special('<|assistant_end|>') or mask[-1]!=1:
                    raise ValueError('Missing supervised assistant termination')
                if not any(mask[1:]):
                    counters['zero_assistant_tokens']+=1
                    continue
                seen.add(full)
                seen_prompts.add(prompt)
                provenance.add((row['source'],row['revision']))
                key=hashlib.sha256(normalize(prompt).encode()).hexdigest()
                kept.append({'id':key,'ids':ids,'mask':mask,'category':category,
                             'source':row['source'],'revision':row['revision']})
                counters['kept']+=1
    # Deduplication precedes split assignment. Hash split is stable under input reordering
    # except for which near-duplicate representative survives the input ordering.
    splits={'train':[],'dev':[]}
    for record in kept:
        split='dev' if int(record['id'][:8],16)/2**32 < dev_fraction else 'train'
        splits[split].append(record)
    for name,rows in splits.items():
        if not rows:
            raise ValueError('Empty '+name+' split; more suitable data is required')
    requested={'general':.7,'context':.2,'reasoning':.1}
    available={r['category'] for r in splits['train']}
    available_dev={r['category'] for r in splits['dev']}
    if not available.issubset(available_dev):
        raise ValueError('A training category has no development examples')
    denominator=sum(requested[c] for c in available)
    effective={c:requested[c]/denominator for c in sorted(available)}
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for split,rows in splits.items():
        with open(output/f'{split}.jsonl','w') as f:
            for row in rows:
                f.write(json.dumps(row)+'\n')
    stats={name:{'examples':len(rows),'input_tokens':sum(len(r['ids'])-1 for r in rows),
                 'assistant_tokens':sum(sum(r['mask'][1:]) for r in rows),
                 'near_context_examples':sum(len(r['ids'])>=int(.9*context) for r in rows),
                 'categories':dict(Counter(r['category'] for r in rows))}
           for name,rows in splits.items()}
    manifest={'format':'nanochat-conversation-cache-v1','context':context,
              'release_gate':{'training_approved':False,
                              'reason':'Acceptance manifests and independent dataset audit must be completed before approval'},
              'tokenizer_sha256':sha256(Path(tokenizer_dir)/'tokenizer.pkl'),
              'files':{f'{s}.jsonl':sha256(output/f'{s}.jsonl') for s in splits},
              'raw_inputs':{str(p):sha256(p) for p in inputs},
              'protected_manifests':{str(p):sha256(p) for p in protected_files},
              'sources':[{'source':s,'revision':r} for s,r in sorted(provenance)],
              'requested_shares':requested,'effective_shares':effective,
              'unavailable_categories':sorted(set(requested)-available),
              'filter_counts':dict(counters),'statistics':stats,
              'dedup_method':'normalized exact + 64-bit 5-word SimHash Hamming<=3; heuristic, not semantic proof'}
    atomic_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',action='append',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--tokenizer-dir',required=True)
    p.add_argument('--protected',action='append',required=True,
                   help='ARC snapshot and sealed development/acceptance prompt manifests')
    args=p.parse_args()
    print(json.dumps(prepare(args.input,args.output,args.tokenizer_dir,args.protected),indent=2))