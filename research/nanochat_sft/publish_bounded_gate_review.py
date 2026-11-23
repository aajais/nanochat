"""Publish a single non-training audit and verify it back from W&B."""
import hashlib,json,os,re
from pathlib import Path
import wandb

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'bounded_gate_review_v1'
RECEIPT=ROOT/'published_bounded_gate_review.json'
if RECEIPT.exists():raise FileExistsError('Existing audit receipt: do not create another run.')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
scope='wandb/nanochat-demo'
api=wandb.Api()
candidate=api.artifact(scope+'/conversation-sft-cache-candidate:v2')
assert candidate.digest=='3f356c1f3cb87a99d5c95dce19dc1d66'
assert candidate.metadata['training_approved'] is False
sample=json.loads((OUT/'sample-adjudication.json').read_text())
fixtures=json.loads((OUT/'fixture-definition-review.json').read_text())
assert sample['counts']=={'pass':87,'reject':10,'uncertain':3} and sample['gate']=='FAIL'
tests={}
for name,n in [('preparation-tests.log',8),('pinned-evaluator-tests.log',63)]:
    text=(OUT/name).read_text()
    assert re.search(rf'Ran {n} tests? in ',text) and text.rstrip().endswith('OK')
    tests[name]={'passed':n,'sha256':sha(OUT/name)}
assert sha(ROOT/'prepared/tests/test_sft_preparation.py')==sha(ROOT/'readiness_bundle_v1/prepared/tests/test_sft_preparation.py')
assert sha(ROOT/'prepared/nanochat/sft_runtime.py')==sha(ROOT/'readiness_bundle_v1/prepared/nanochat/sft_runtime.py')
holdout=json.loads((ROOT/'published_authorized_holdout_check.json').read_text())
summary={
 'decision':'NO-GO: GPU work stays paused','training_approved':False,'launch_ready':False,
 'content_review':{'gate':'FAIL','sample':100,'counts':sample['counts'],'independent_human_approval':False},
 'fixture_review':{'targeted_wording_repairs':'PASS','changed_definitions_checked':24,'human_approval':False},
 'license_review':{'gate':'HOLD','named_owner':None,'intended_use_scope':None,'owner_acceptance':False},
 'holdout_lexical_comparison':{'status':'previously completed; not repeated this pass','candidates':17018,'holdout_documents':2000,
   'normalized_exact_matches':0,'shared_13word_matches':0,'semantic_nonoverlap_proven':False,
   'evidence_ref':holdout['artifacts']['nanochat-sft-holdout-comparison']['ref'],
   'evidence_url':holdout['artifacts']['nanochat-sft-holdout-comparison']['url'],
   'method':'NFKC/casefold word-normalized document SHA256 plus contiguous 13-word BLAKE2b128; independently cross-checked using direct normalized strings and SHA256 13-grams.',
   'custody_disclosure':'Previous authorized automated export read protected payload programmatically; no independent human-custodian approval claimed. No holdout payload read or sent to a model in this pass.'},
 'sft_runner':{'gate':'UNVERIFIED','successful_production_gpu_resume_test':False,'remaining_smoke_max_seconds':780},
 'existing_summary_quarantine':{'count':1166,'unchanged':True},'candidate_v2_unchanged':True,
 'tests':tests,'gpu_jobs_launched':0,'sft_updates':0,'acceptance_generations':0,
 'nested_model_review_requests':10,'validated_nested_review_batches':8,'direct_ARIA_fallback_examples':20,
 'human_approval_claimed':False,
 'next_actions':['Remediate the 10 clear rejects and adjudicate or conservatively exclude the three unresolved examples; address recurring constraint-violation pattern before building a new candidate version.',
  'Responsible owner must confirm intended use/distribution scope and accept the source-license/notice and constituent/generator-terms disposition.',
  'Record independent human review where required; targeted AI fixture checks are not human approval.',
  'Only after data/license gates clear, pin an approved cache, evaluator and exact fixtures into a reconciled bounded smoke job; verify actual save/resume/deadline/upload and failure propagation before conditional SFT.']}
(OUT/'gate-decision.json').write_text(json.dumps(summary,indent=2))
readme='''# Bounded SFT readiness pass — NO-GO

GPU work remains paused. No SFT update, GPU job or acceptance response was generated.
This audit uses ten small project-scoped language-model review requests on the
existing training-data sample only. Eight batch outputs passed evidence validation;
two failed validation and were not retried or relied on. ARIA directly inspected
the affected twenty examples. All ten rejects and three uncertain cases have
specific source IDs and rationales. Four automated false positives were overturned.

Final AI-assisted sampled-target decisions: 87 pass, 10 reject, 3 uncertain.
These are not independent human approvals, corpus error-rate estimates or model
performance results. The sample is twenty examples from each of five sources.
Content gate FAIL: unsupported city-specific subway claims, a fish-count problem
omitting two stated color groups, an exact-word repetition miss, and seven
summaries using He/She despite an explicit prohibition. Three unresolved cases
concern an unverified legal citation, named course recommendations, and the
interpretation of a possessive form under the pronoun ban.

The existing 1,166-item quarantine and all candidate-v2 cache bytes remain unchanged.
Additional rejects/review holds are recommendations for remediation, not claims
that the immutable v2 cache has already been rebuilt or those rows removed.

Static fixture repair checks PASS: 24 changed definitions across three splits,
with reference checks unchanged across all 300 definitions. This does not supply
human sign-off or any model acceptance result. Re-ran 63 pinned evaluator and
eight preparation tests, all passing. The SFT runner's real periodic checkpoint,
fresh-process GPU restore and graceful deadline/signal/upload path remain unverified.

Source cards identify Apache-2.0 for four components and MIT for MetaMathQA;
the accompanying license review records concrete notice/provenance obligations
and unresolved constituent/generator terms. No responsible owner or intended
distribution scope has been supplied; owner acceptance is pending.

The separate completed holdout lexical audit found zero exact-document/13-word
matches across 17,018 retained candidates and 2,000 holdout documents. It was
not repeated here, does not prove absence of semantic overlap, and was an
authorized automated export, not independent human-custodian sign-off. Earlier
pending fields in the immutable candidate/readiness metadata are historical.

Use gate-decision.json for status; sample-adjudication.json for row-level results;
LICENSE_AND_ATTRIBUTION_REVIEW.md and SFT_RUNNER_EVIDENCE.md for exact missing
decisions/evidence. Do not treat a successful audit publication as training approval.
'''
(OUT/'README.md').write_text(readme)
files=[OUT/name for name in ['README.md','gate-decision.json','sample-adjudication.json',
 'proposed-additional-exclusions.json','fixture-definition-review.json','input-integrity.json',
 'LICENSE_AND_ATTRIBUTION_REVIEW.md','SFT_RUNNER_EVIDENCE.md','preparation-tests.log',
 'pinned-evaluator-tests.log','sample-review-errors.json','reviewer-instructions.txt']]
extras={ROOT/'readiness_review_v1/source-evidence.json':'source-evidence.json',
 ROOT/'consolidate_bounded_gate_review.py':'consolidate_bounded_gate_review.py',
 ROOT/'bounded_review_samples.py':'bounded_review_samples.py',
 ROOT/'readiness_review_v1/filtered-dataset-human-review-sample.json':'reviewed-100-example-packet.json'}
hashes={p.name:sha(p) for p in files}
hashes.update({name:sha(p) for p,name in extras.items()})
provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if v in os.environ}
run=wandb.init(entity='wandb',project='nanochat-demo',name='bounded-readiness-pass-no-go-no-gpu',
 job_type='sft-readiness-audit',dir=str(ROOT),save_code=False,
 config={'_wb_agent':provenance,'training_approved':False,'launch_ready':False,'scope':'bounded AI-assisted content/fixture/license/runner evidence audit; no GPU or acceptance generation'})
try:
 for ref in ['conversation-sft-cache-candidate:v2','nanochat-conversation-evaluation-runner:v2',
             'nanochat-conversation-demo-evals:v1','nanochat-sft-holdout-comparison:v0']:
  run.use_artifact(scope+'/'+ref)
 artifact=wandb.Artifact('nanochat-sft-bounded-gate-review',type='evaluation',
  metadata={'decision':'NO-GO','training_approved':False,'human_approval':False,
    'sample_counts':sample['counts'],'existing_quarantine_unchanged':1166,'gpu_jobs_launched':0,'file_sha256':hashes})
 for p in files:artifact.add_file(str(p),name=p.name)
 for p,name in extras.items():artifact.add_file(str(p),name=name)
 logged=run.log_artifact(artifact);logged.wait()
 ref=f'{scope}/{logged.name}'
 run.summary.update({'decision':'NO-GO','training_approved':False,'sample_pass':87,'sample_reject':10,
    'sample_uncertain':3,'gpu_jobs_launched':0,'acceptance_generations':0,'unit_tests_passed':71})
 run_id,run_url=run.id,run.url
 run.finish()
except BaseException:
 run.finish(exit_code=1)
 raise
server=api.artifact(ref)
assert server.metadata['decision']=='NO-GO' and server.metadata['training_approved'] is False
assert server.digest==logged.digest
verification=OUT/'server-verification'
download=Path(server.download(root=str(verification)))
for name,digest in hashes.items():assert sha(download/name)==digest,name
server_run=api.run(scope+'/'+run_id)
assert server_run.state=='finished' and server_run.summary.get('decision')=='NO-GO'
again=api.artifact(scope+'/conversation-sft-cache-candidate:v2')
assert again.digest==candidate.digest and again.metadata['training_approved'] is False
url=f'https://forge.coreweave.com/wandb/{scope}/artifacts/evaluation/nanochat-sft-bounded-gate-review/{server.version}'
receipt={'run_id':run_id,'run_url':run_url,'artifact_ref':ref,'artifact_url':url,'artifact_digest':server.digest,
 'published_file_count':len(hashes),'all_published_files_server_byte_verified':True,
 'server_run_state':server_run.state,'candidate_approval_unchanged':True,'training_approved':False,'gpu_jobs_launched':0}
RECEIPT.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)