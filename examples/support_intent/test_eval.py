import copy,json,tempfile,unittest
from pathlib import Path
from eval_core import LABELS,LexicalNB,load_rows,parse_output,score,candidate_loss
from run_eval import require_pinned,generate_label,rank_labels

class ScoringTests(unittest.TestCase):
 def setUp(self):self.rows=[{'label':l} for l in LABELS]
 def test_perfect(self):
  r=score(self.rows,list(LABELS));self.assertEqual(r['macro_f1'],1);self.assertEqual(r['accuracy'],1)
 def test_invalid_counts_as_error(self):
  r=score(self.rows,['garbage']*5)
  self.assertEqual(r['macro_f1'],0);self.assertEqual(r['invalid_rate'],1)
 def test_unknown_only_not_fiveway_macro(self):
  r=score([{'label':'unknown'}],['unknown']);self.assertIsNone(r['macro_f1']);self.assertEqual(r['unknown_recall'],1)
  self.assertIsNone(r['supported_false_abstention_rate'])
 def test_false_abstention(self):
  r=score(self.rows,['unknown']*5);self.assertEqual(r['supported_false_abstention_rate'],1)
 def test_strict_output_contract(self):
  self.assertEqual(parse_output(' billing\n'),'billing')
  for s in ['Billing','"billing"','billing because it is a bill','billing\nunknown','']:
   self.assertEqual(parse_output(s),'__invalid__')
 def test_length_mismatch(self):
  with self.assertRaises(ValueError):score(self.rows,[])
 def test_mean_continuation_loss(self):self.assertEqual(candidate_loss([1,3]),2)
 def test_pinned_refs(self):
  require_pinned('wandb/nanochat-demo/model:v0')
  for s in ['wandb/nanochat-demo/model:latest',None,'model:v0','wandb/nanochat-demo/model']:
   with self.assertRaises(ValueError):require_pinned(s)
 def test_latency_quantile(self):
  r=score(self.rows,list(LABELS),[.1,.2,.3,.4,.5]);self.assertEqual(r['p95_serial_model_seconds'],.5)

class DatasetTests(unittest.TestCase):
 def test_integrity_and_disjointness(self):
  root=Path(__file__).parent/'dataset';seen=set()
  for fold in ['train','dev','test','challenge_dev','challenge_test']:
   rows=load_rows(root,fold);keys={r['normalized_sha256'] for r in rows}
   self.assertEqual(len(keys),len(rows));self.assertFalse(keys&seen);seen|=keys
  self.assertEqual({r['label'] for r in load_rows(root,'train')},set(LABELS))
 def test_no_reassigned_official_splits(self):
  root=Path(__file__).parent/'dataset'
  for fold,source in [('train','train'),('dev','val'),('test','test'),('challenge_dev','oos_val'),('challenge_test','oos_test')]:
   self.assertTrue(all(r['source_split']==source for r in load_rows(root,fold)))
 def test_nb_learns_and_is_deterministic(self):
  rows=[{'label':l,'text':l+' unique_'+l} for l in LABELS]
  m=LexicalNB().fit(rows)
  for r in rows:self.assertEqual(m.predict(r['text']),r['label'])
  self.assertEqual(m.predict('unseenword'),m.predict('unseenword'))
 def test_checksum_tampering_rejected(self):
  import shutil
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
   root=Path(folder);shutil.copyfile(Path(__file__).parent/'dataset/manifest.json',root/'manifest.json')
   (root/'dev.jsonl').write_text('{}\n')
   with self.assertRaises(ValueError):load_rows(root,'dev')

class FakeTokenizer:
 def get_bos_token_id(self):return 1
 def encode_special(self,s):return 2
 def encode(self,s):return [3]*len(s)
 def decode(self,ids):return 'billing\n'

class FakeForward:
 def __call__(self,ids):
  import torch
  x=torch.zeros((ids.shape[0],ids.shape[1],5));x[:,:,-1]=10
  return x

class ModelAdapterTests(unittest.TestCase):
 def test_greedy_adapter_no_external_model(self):
  raw,prompt_len,generated=generate_label(FakeForward(),FakeTokenizer(),'test','cpu',12,2048)
  self.assertEqual(raw,'billing\n');self.assertEqual(generated,1)
 def test_likelihood_adapter_no_external_model(self):
  label,losses=rank_labels(FakeForward(),FakeTokenizer(),'test','cpu',2048)
  self.assertIn(label,LABELS);self.assertEqual(len(losses),5)
  self.assertTrue(all(abs(losses[0]-x)<1e-5 for x in losses))
 def test_context_overflow_rejected(self):
  for fn,args in [(generate_label,(FakeForward(),FakeTokenizer(),'test','cpu',12,2)),
                  (rank_labels,(FakeForward(),FakeTokenizer(),'test','cpu',2))]:
   with self.assertRaises(ValueError):fn(*args)

if __name__=='__main__':unittest.main()