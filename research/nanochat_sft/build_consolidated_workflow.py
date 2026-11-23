"""Prepare immutable manifests/decision packet only. Never submits a queue item."""
import json
import hashlib
from pathlib import Path
import shutil
import sys
import wandb

ROOT=Path('/workspace/nanochat_sft');P=ROOT/'final_workflow_v1'
sys.path.insert(0,str(P))
from nanochat.sft_runtime import sha256,atomic_json
from nanochat.workflow_policy import REQUIRED_DECISIONS,Allocation

def main():
    api=wandb.Api()
    refs=dict(candidate='conversation-sft-cache-candidate:v3',evaluator='nanochat-conversation-evaluation-runner:v2',
        fixtures='nanochat-conversation-demo-evals:v1',tokenizer='pr1-smoke-tokenizer:v0',base='b1384b1-checkpoint:v2',
        arc='e1384arc-benchmark-inputs:v0',historical_arc_code='code-nanochat-1384m-arc100-text10-20261123:v0')
    assets={key:dict(ref='wandb/nanochat-demo/'+name,digest=api.artifact('wandb/nanochat-demo/'+name).digest) for key,name in refs.items()}
    assets['base']['sha256']={name:sha256(ROOT/'base_metadata'/name) for name in ('model_133632.pt','meta_133632.json')}
    assets['tokenizer']['sha256']={'tokenizer.pkl':'a228803c68627f650e640c819e71ea0b90f49a96118858ed781a00a5ba63b6bf'}
    assets['arc']['sha256']={'benchmark-inputs.json':sha256(ROOT/'arc_inputs/benchmark-inputs.json')}
    assets['candidate']['sha256']={name:sha256(ROOT/'cpu_remediation_v2/candidate'/name) for name in ('manifest.json','train.jsonl','dev.jsonl')}
    assets['fixtures']['sha256']={name:sha256(ROOT/'readiness_bundle_v1/conversation_evals_v3'/name) for name in ('development.json','acceptance_primary.json','acceptance_correction.json','protocol.json')}
    # Read back the actual corrected server source, not only a local label.
    evaluator=api.artifact(assets['evaluator']['ref']);verify=ROOT/'final-workflow-server-source'
    for name in ('nanochat/conversation_eval.py','nanochat/conversation_eval_checkpoint.py','scripts/conversation_eval.py'):
        matches=[n for n in evaluator.manifest.entries if n.endswith('/'+name) or n==name]
        if len(matches)!=1:raise ValueError(('Corrected evaluator source ambiguous',name,matches))
        downloaded=Path(evaluator.get_path(matches[0]).download(root=str(verify)))
        if sha256(downloaded)!=sha256(P/name):raise ValueError('Corrected evaluator changed: '+name)
    # Freeze source and all entrypoints, including this provenance-producing build.
    packet=P/'decision';packet.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'bounded_gate_review_v1/LICENSE_AND_ATTRIBUTION_REVIEW.md',packet/'historical-license-inventory.md')
    shutil.copy2(ROOT/'readiness_review_v1/source-evidence.json',packet/'source-evidence.json')
    review=json.loads((ROOT/'cpu_remediation_v2/rebuilt-human-review-sample.json').read_text())
    retained={}
    for split in ('train','dev'):
        for line in (ROOT/'cpu_remediation_v2/candidate'/f'{split}.jsonl').open():
            row=json.loads(line);retained[row['id']]=dict(split=split,category=row['category'])
    examples=[]
    for source in sorted({e['source'] for e in review['examples']}):
        options=[e for e in review['examples'] if e['source']==source and e['id'] in retained]
        selected=min(options,key=lambda e:len(json.dumps(e['messages'])))
        examples.append(dict(**selected,retained_identity=retained[selected['id']]))
    development=json.loads((ROOT/'readiness_bundle_v1/conversation_evals_v3/development.json').read_text())
    fixtures=[e for e in development['examples'] if 'label' in json.dumps(e) or 'DIRECTION' in json.dumps(e)]
    # Only representative corrected development fixtures are opened in this packet.
    repaired=[]
    for family in ('object','shelf'):
        match=next((e for e in fixtures if family in json.dumps(e)),None)
        if match and match not in repaired:repaired.append(match)
    if len(repaired)<2:
        repaired=fixtures[:2]
    atomic_json(packet/'REVIEW_EXAMPLES.json',dict(candidate=assets['candidate'],
        scope='One shortest complete retained example per source: coverage sample, NOT weighted prevalence estimate.',
        content_approval=False,retained_examples=examples,corrected_development_fixtures=repaired,
        full_review_sample_ref='wandb/nanochat-demo/nanochat-sft-cpu-remediation:v0/rebuilt-human-review-sample.json'))
    lines=['# Decision packet — HOLD; no GPU launch',
        '\nOwner: **Aayush Jaiswal**. Scope: **internal demo only**. Neither is acceptance.',
        '\n## Required decisions — all explicitly pending',
        '1. **Retained content:** approve the pinned v3 basis or identify exclusions. There are 15,375 train + 789 dev conversations. The five complete examples in `REVIEW_EXAMPLES.json` cover all sources, not statistical/weighted representativeness. Full 100-example source-stratified review remains linked there. Rule screening is not factual/content approval.',
        '2. **Corrected fixtures:** approve the exact v1 evaluation artifact/v3 fixture bytes. Object labels and DIRECTION shelf wording now match unchanged strict references. Review the development examples below and the complete approved definitions; thresholds/decoding are unchanged. No final-acceptance generations have been run in preparation.',
        '3. **License/notice disposition:** resolve every row below, provide applicable authoritative notice text, and explicitly approve the resulting internal-use basis. No blanket clearance is inferred.',
        '4. **Selection:** explicitly approve `mechanical_development_then_human_release_hold`: initial model + at most three SFT checkpoints; rank development mechanical turn passes (including multi-turn recall/correction), boundaries, then loop-free responses; stable earlier ties. This is NOT human usefulness selection. Primary acceptance runs once only after that selection. Package stays research-only pending human review; no automated release, correction or retry.',
        '5. **GPU execution:** authorize this exact pinned workflow, one GPU on nanochat-demo-k8s, 7,200 total seconds including 87 already charged. This preparation does not record authorization.',
        '\n## Exact unresolved license/notice work',
        '| Component | Required resolution / applicable obligation |',
        '|---|---|',
        '| Apache-2.0: magpie, constraints, summarize, everyday | Determine applicability to internal serving, dataset/code/checkpoint handling. Where distribution terms apply: include Apache license; preserve applicable copyright/patent/trademark/attribution notices; mark modified files; reproduce applicable upstream NOTICE. Locate authoritative texts/NOTICE or document authoritative absence. Summarization constituent rights remain part of this disposition. |',
        '| MetaMathQA augmentation (MIT) | Locate authoritative copyright holder/notice and complete permission text; retain both in covered copies/substantial portions. Card label is insufficient; never fabricate the copyright line. |',
        '| Underlying GSM8K | Establish exact source/version, applicable terms and notices, and compatibility of intended internal use. Not cleared by MIT augmentation label. |',
        '| Underlying MATH | Same: exact source/version, applicable terms/notices and intended-use basis remain unresolved. |',
        '| Llama-3.1-generated portions | Record whether generator/output terms apply in the actual generation/use context and why. If applicable, fulfill the actual attribution/notice/use conditions. Generator name alone establishes neither blanket restriction nor permission. |',
        '| Original nanochat source and dependencies | The immutable image is pinned, not a third-party notice audit. Preserve applicable original source/dependency licenses when redistributing code/image; confirm authoritative source license/NOTICE in the approved bundle. Base pretraining corpus rights are not newly certified by SFT. |',
        '\nThe historical inventory is evidence only; its old “owner not provided” text is superseded by the owner above, NOT by acceptance. The authoritative notice bundle and reasoned dispositions must accompany explicit approval.',
        '\n## Retained examples — exact excerpts; full complete conversations in REVIEW_EXAMPLES.json']
    for e in examples:
        lines+=['\n### '+e['source'].split('/')[-1]+' — '+e['id'][:12],
                'Retained '+retained[e['id']]['split']+' / '+retained[e['id']]['category']+'.',
                '\n'.join('> '+m['role']+': '+m['content'][:650].replace('\n','\n> ')+(' [excerpt]' if len(m['content'])>650 else '') for m in e['messages'][:2])]
    lines+=['\n## Corrected development fixtures (not final acceptance)']
    for e in repaired:lines+=['\n### '+e['id'],'```json',json.dumps(e,indent=2),'```']
    lines+=['\n## Informational caveats — not additional owner sign-offs',
        '- Zero residual implemented flags is not a semantic guarantee. Shared fixture templates limit generalization claims; strict exact-match intentionally rejects extra text.',
        '- CPU tests use tiny native GPT/MuonAdamW and mocked service boundaries. Real 1.384B BF16, upload time, memory fit and exact GPU replay remain readiness proofs; any missing proof exits with evidence.',
        '- One GPU; no distributed-resume portability claim. Startup/queue time is conservatively charged from submission. No remote training dataset reads or kernel acquisition.',
        '- Evaluation is capped at 25 minutes, packaging/upload at 10 minutes, exit at 1 minute. If a stage cannot finish within its reserve, stop incomplete; never reduce denominators or claim a release.',
        '- Historical ARC100 is the same cached 100, same zero-shot length-normalized choice NLL. The historical ten unrelated text completions are not part of retention scoring.',
        '- No live queue item, GPU training or acceptance generation is created by this packet.']
    (packet/'DECISIONS.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(ROOT/'final_workflow_v1-cpu-tests.log',packet/'production-cpu-tests.log')
    shutil.copy2(ROOT/'final_workflow_v1-evaluator-tests.log',packet/'corrected-evaluator-tests.log')
    source={str(p.relative_to(P)):sha256(p) for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('WORKFLOW.json','APPROVAL_TEMPLATE.json') and 'wandb' not in p.parts}
    manifest=dict(schema='nanochat-consolidated-sft-v1',assets=assets,source_sha256=source,
        image='wandb/autoresearch-base@sha256:05578cc98ed0629b6518482bfe8de57935e5b852ed6057446b2a09a7f60edeca',
        candidate_manifest_sha256=assets['candidate']['sha256']['manifest.json'],gpu_count=1,queue='nanochat-demo-k8s',
        entrypoint=['python','-u','workflow.py'],budget=Allocation(__import__('time').time()).spec(),
        launch_authorized=False,release_approved=False,approval_required=True,corrected_evaluator_bytes_server_verified=True)
    # No prepared timestamp is an allocation: actual clock is stamped only at future submission.
    for key in ('allocation_started_unix','allocation_ends_unix'):manifest['budget'].pop(key)
    atomic_json(P/'WORKFLOW.json',manifest)
    decisions={name:dict(status='pending',reviewer='',evidence='') for name in REQUIRED_DECISIONS}
    decisions['selection_policy']['policy']='mechanical_development_then_human_release_hold'
    decisions['license_notice_disposition'].update(dispositions={k:'' for k in ('apache_components','metamath_mit','gsm8k_underlying','math_underlying','llama31_outputs','source_dependencies')},notice_bundle={})
    decisions['gpu_execution'].update(total_seconds_including_prior=7200,gpu_count=1)
    atomic_json(P/'APPROVAL_TEMPLATE.json',dict(owner='Aayush Jaiswal',scope='internal demo only',workflow_sha256=sha256(P/'WORKFLOW.json'),assets=assets,decisions=decisions))
    print(json.dumps(dict(workflow_sha256=sha256(P/'WORKFLOW.json'),files=len(source),gpu_launched=False)))

if __name__=='__main__':main()