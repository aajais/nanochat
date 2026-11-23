"""Strict-subset candidate rebuild; preserves parent v2, token bytes and splits."""
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

from remediation_checks import inspect

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'prepared'))
from nanochat.sft_runtime import PackedLoader, sha256
from nanochat.tokenizer import RustBPETokenizer
import torch

OUT = ROOT/'cpu_remediation_v2'
CACHE = OUT/'candidate'
PARENT = ROOT/'conversation_cache_quality_v1'

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def main():
    OUT.mkdir(exist_ok=False)
    CACHE.mkdir()
    parent = json.loads((PARENT/'manifest.json').read_text())
    original_hashes = {p:sha256(PARENT/p) for p in ['train.jsonl','dev.jsonl','manifest.json','quarantine-ids.json']}
    assert all(original_hashes[p] == h for p,h in parent['files'].items())
    binding = json.loads((ROOT/'authorized_holdout_check_v1/candidate-binding-evidence.json').read_text())
    rawpath = ROOT/'authorized_holdout_check_v1/retained-candidate-verified-texts.jsonl'
    assert sha256(rawpath) == binding['candidate_text_sha256']
    assert binding['candidate_cache_files'] == parent['files']
    sources = {x['id']:x for x in json.loads((ROOT/'authorized_holdout_check_v1/candidate-line-map.json').read_text())}
    review = json.loads((ROOT/'bounded_gate_review_v1/sample-adjudication.json').read_text())
    mandatory = {r['id']:r for r in review['rows'] if r['decision'] in ('reject','uncertain')}
    assert len(mandatory) == 13
    flagged, checked, ids, raw_kept = {}, Counter(), set(), {}
    by_kind, by_source = Counter(), Counter()
    for line in rawpath.open():
        row = json.loads(line); key = row['id']
        assert key not in ids; ids.add(key)
        reasons, coverage = inspect(row['messages']); checked.update(coverage)
        if key in mandatory:
            r = mandatory[key]
            reasons.append(dict(kind='mandatory_packet_'+r['decision'], packet_index=r['packet_index'], rationale=r['rationale']))
        if reasons:
            flagged[key] = dict(id=key, split=row['split'], source=sources[key]['source'], source_row=sources[key]['source_row'], reasons=reasons)
            by_kind.update(set(r['kind'] for r in reasons))
            by_source[sources[key]['source'].rsplit('/',1)[-1]] += 1
        else:
            raw_kept[key] = row
    assert len(ids) == 17018 and mandatory.keys() <= flagged.keys()
    old_quarantine = set(json.loads((PARENT/'quarantine-ids.json').read_text()))
    assert len(old_quarantine) == 1166 and not old_quarantine & ids
    manifest = copy.deepcopy(parent)
    stats, removed, rows_by_split, seen = {}, {}, {}, set()
    for split in ['train','dev']:
        retained, drop = [], 0
        with (PARENT/(split+'.jsonl')).open() as stream, (CACHE/(split+'.jsonl')).open('x') as sink:
            for line in stream:
                row = json.loads(line); key = row['id']
                assert key not in seen; seen.add(key)
                assert key in ids and sources[key]['split'] == split
                if key in flagged:
                    drop += 1; continue
                # Original serialized token/mask rows are copied, never re-encoded.
                assert raw_kept[key]['split'] == split
                sink.write(line); retained.append(row)
        removed[split] = drop; rows_by_split[split] = retained
        stats[split] = dict(examples=len(retained), input_tokens=sum(len(r['ids'])-1 for r in retained),
                            assistant_tokens=sum(sum(r['mask'][1:]) for r in retained),
                            near_context_examples=sum(len(r['ids'])>=1843 for r in retained),
                            categories=dict(Counter(r['category'] for r in retained)))
        manifest['files'][split+'.jsonl'] = sha256(CACHE/(split+'.jsonl'))
    assert seen == ids and sum(removed.values()) == len(flagged)
    new_ids = {r['id'] for rows in rows_by_split.values() for r in rows}
    assert new_ids == ids-set(flagged)
    assert not new_ids & (old_quarantine | set(mandatory))
    # Every retained answer is also re-scanned, independently of decisions used above.
    assert all(not inspect(raw_kept[key]['messages'])[0] for key in new_ids)
    parent_comparison = json.loads((ROOT/'authorized_holdout_check_v1/comparison.json').read_text())
    assert parent_comparison['candidates_checked'] == len(ids) and not parent_comparison['lexical_flags']
    assert sha256(rawpath) == parent_comparison['candidate_sha256']
    holdout = dict(method='Strict subset of previously checked parent, identical prompts/answers/tokens; no holdout payload or fingerprint index read',
                   parent_ref='wandb/nanochat-demo/nanochat-sft-holdout-comparison:v0',
                   parent_comparison_sha256=sha256(ROOT/'authorized_holdout_check_v1/comparison.json'),
                   all_candidate_ids_subset_of_verified_parent=True, lexical_matches=0,
                   semantic_overlap_excluded=False, independent_custodian_approval=False)
    manifest['statistics'] = stats
    manifest['filter_counts']['content_remediation_quarantined'] = len(flagged)
    manifest['filter_counts']['kept'] = len(new_ids)
    manifest['parent_candidate'] = dict(ref='wandb/nanochat-demo/conversation-sft-cache-candidate:v2', digest='3f356c1f3cb87a99d5c95dce19dc1d66', file_sha256=original_hashes)
    manifest['content_remediation'] = dict(rule_code_sha256=sha256(ROOT/'remediation_checks.py'), additional_exclusions=len(flagged),
        mandatory_packet_exclusions=13, parent_exclusions_preserved=1166,
        strict_subset=True, original_tokens_masks_and_splits_preserved=True, labels_rewritten=False,
        semantic_truth_verified=False, human_review='pending')
    manifest['intended_use'] = dict(scope='internal demo only', responsible_owner='Aayush Jaiswal',
        recorded_from_user_message_utc='2026-11-23T17:06:36Z',
        license_and_notice_disposition='Owner and scope confirmed; no explicit disposition of outstanding constituent/generator/notice questions inferred.')
    manifest['release_gate']['training_approved'] = False
    manifest['release_gate']['completed'] += ['all 17018 parent conversations screened with recorded mechanical/risk rules', '13 packet reject/uncertain IDs conservatively excluded', 'strict-subset holdout lexical non-match inheritance']
    manifest['release_gate']['pending'] = ['human dataset content review of rebuilt candidate', 'human evaluation fixture review', 'responsible-owner disposition of recorded license/attribution questions', 'production SFT entrypoint save/resume/shutdown/upload check as first stage of shared final budget']
    manifest['holdout_comparison'] = holdout
    write(CACHE/'manifest.json', manifest)
    write(CACHE/'quarantine-ids.json', sorted(old_quarantine | set(flagged)))
    write(OUT/'additional-quarantine.json', dict(count=len(flagged), rows=list(flagged.values()), by_reason=dict(by_kind), by_source=dict(by_source)))
    # Preserve excluded source text for audit, outside the usable candidate directory.
    with rawpath.open() as stream, (OUT/'quarantined-conversations.jsonl').open('x') as sink:
        for line in stream:
            if json.loads(line)['id'] in flagged: sink.write(line)
    torch.set_num_threads(1)
    tok = RustBPETokenizer.from_directory(ROOT/'tokenizer')
    assert sha256(ROOT/'tokenizer/tokenizer.pkl') == manifest['tokenizer_sha256']
    loader = PackedLoader(rows_by_split['train'],2048,8,tok.get_bos_token_id(),manifest['effective_shares'],seed=20261123)
    assistant = 0
    for _ in range(128):
        x,y = loader.next('cpu'); assert x.shape == y.shape == (8,2048)
        assistant += int((y!=-1).sum())
    packing = dict(updates=128, input_slots=128*16384, assistant_tokens=assistant,
        assistant_category_shares={k:v/assistant for k,v in loader.category_tokens.items()},
        requested_shares=manifest['requested_shares'], training_or_gradients=False)
    samples = defaultdict(list)
    for key in new_ids:
        row = raw_kept[key]
        samples[sources[key]['source']].append(dict(row,source=sources[key]['source'],source_row=sources[key]['source_row']))
    selected = []
    for source, rows in sorted(samples.items()):
        rows.sort(key=lambda r:hashlib.sha256(('review-remediation-v2:'+r['id']).encode()).hexdigest())
        selected += rows[:20]
    write(OUT/'rebuilt-human-review-sample.json', dict(scope='20 per source; source-stratified, not representative under weighted sampling',human_approval=False,candidate_files=manifest['files'],examples=selected))
    assert original_hashes == {p:sha256(PARENT/p) for p in original_hashes}
    result = dict(parent_conversations=len(ids), checked=dict(checked), mandatory_quarantined=13,
        additional_quarantined=len(flagged), cumulative_quarantined=len(old_quarantine)+len(flagged),
        quarantined_by_split=removed, quarantined_by_reason=dict(by_kind), quarantined_by_source=dict(by_source),
        statistics=stats, packing=packing, parent_bytes_unchanged=True, rescan_remaining_flags=0,
        holdout=holdout, training_approved=False, gpu_jobs_launched=0, model_calls=0,
        coverage_limit='Mechanical constraints and named defect families only. No general factual, arithmetic, grounding or semantic correctness guarantee. Raw input is the entire retained 17018-row candidate, not all upstream SmolTalk.')
    write(OUT/'remediation-result.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__': main()