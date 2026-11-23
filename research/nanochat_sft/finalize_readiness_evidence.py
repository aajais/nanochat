"""Refresh staged source manifest and audit receipts. CPU hashes only; no loading/generation."""
import importlib.util,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
bundle=ROOT/'readiness_bundle_v1';review=ROOT/'readiness_review_v1'
sys.path.insert(0,str(bundle/'prepared'))
from nanochat import conversation_eval_checkpoint as ck
from nanochat.conversation_eval import PINNED_HASHES

tests={}
for name,count in [('runner-v3-tests.log',63),('preparation-v3-tests.log',8),('fingerprint-tests.log',3)]:
    path=review/name;log=path.read_text()
    found=re.search(r'Ran (\d+) tests? in',log)
    assert found and int(found[1])==count and log.rstrip().endswith('OK')
    tests[name]={'passed':count,'log_sha256':ck.sha256(path)}
assert {name:ck.sha256(bundle/'conversation_evals_v3'/name) for name in PINNED_HASHES}==PINNED_HASHES
manifest=dict(schema='nanochat-eval-checkpoint-v1',kind='base',model_config=ck.MODEL_CONFIG,
              base_weights_sha256=ck.BASE_WEIGHTS,checkpoint_sha256=ck.BASE_WEIGHTS,
              tokenizer_sha256=ck.TOKENIZER,weights_file='base_metadata/model_133632.pt',
              metadata_file='base_metadata/meta_133632.json',
              source_sha256={name:ck.sha256(bundle/'prepared/nanochat'/name) for name in ck.SOURCE_NAMES})
path=review/'base-evaluation-manifest-v3.json';path.write_text(json.dumps(manifest,indent=2))
trust=ck.sha256(path)
checked,weights=ck.verify_manifest(ROOT,path,trust,ROOT/'tokenizer')
assert checked==manifest and weights.is_file()
quality=json.loads((review/'quality-filtered-cache-audit.json').read_text())
cache=ROOT/'conversation_cache_quality_v1'
cache_manifest=json.loads((cache/'manifest.json').read_text())
for name,value in cache_manifest['files'].items():assert ck.sha256(cache/name)==value
assert cache_manifest['release_gate']['training_approved'] is False
assert quality['original_cache_unchanged']
assert len(json.loads((review/'filtered-dataset-human-review-sample.json').read_text())['examples'])==100
evidence={'scope':'CPU-only dataset/evaluation readiness; no GPU attempt','targeted_eval_and_preparation_tests_passed':71,
          'fingerprint_utility_synthetic_tests_passed':3,'test_logs':tests,
          'base_manifest_sha256':trust,'base_file_and_source_hashes_verified':True,
          'base_model_loaded_this_audit':False,'source_artifact_publication':'pending',
          'evaluation_revision':'v3; wording fixes, unchanged reference answers, thresholds and decoding',
          'development_and_acceptance_files':PINNED_HASHES,
          'dataset_cache_files':cache_manifest['files'],'dataset_training_approved':False,
          'holdout_payloads_opened':False,'holdout_export_executed':False,
          'gpu_jobs_launched':0,'model_generation_calls':0,'acceptance_generations':0,
          'remaining_gates':['human review of retained dataset sample and revised fixtures',
                             'dataset/source attribution and license acceptance',
                             'custodian holdout fingerprint export and recorded candidate comparison',
                             'review/update GPU package references then bounded native train/save/fresh-process-restore smoke']}
(review/'test-evidence.json').write_text(json.dumps(evidence,indent=2))
(bundle/'base-evaluation-manifest-v3.json').write_text(path.read_text())
print(json.dumps(evidence,indent=2),flush=True)