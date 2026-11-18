import asyncio
import unittest
from unittest.mock import patch
import weave
from eval_core import LABELS, PROMPT
from structured_eval import SupportModel, SupportScorer, execute


IDENTITY = {'checkpoint_artifact':'wandb/nanochat-demo/model:v0',
            'model_weight_sha256':'abc','tokenizer_artifact':'wandb/nanochat-demo/tokenizer:v0',
            'method':'greedy_label'}


class StructuredTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'id':str(i),'text':label,'label':label} for i,label in enumerate(LABELS)]

    def predictor(self, text, example_id):
        return {'prediction':text,'raw_output':text,'latency_seconds':0.1,
                'prompt':PROMPT.format(text=text),'inference_mode':'live'}

    def test_unconfigured_runs_once_and_retains_outputs(self):
        seen=[]
        def predict(text,example_id):
            seen.append(example_id)
            return self.predictor(text,example_id)
        outputs,receipt=execute(self.rows,predict,IDENTITY,dataset_artifact='pinned:v0',split='dev',trace_enabled=False)
        self.assertEqual(seen,[r['id'] for r in self.rows])
        self.assertEqual([x['raw_output'] for x in outputs],list(LABELS))
        self.assertFalse(receipt['backend_verified'])

    def test_unreachable_init_does_not_change_inference(self):
        def refused(_):raise ConnectionRefusedError('synthetic refused endpoint')
        with self.assertWarns(UserWarning):
            outputs,receipt=execute(self.rows,self.predictor,IDENTITY,dataset_artifact='pinned:v0',split='dev',init_fn=refused)
        self.assertEqual(len(outputs),5)
        self.assertEqual(receipt['trace_error'],'ConnectionRefusedError')

    def test_pooled_scorer(self):
        scorer=SupportScorer()
        rows=[scorer.score(r['label'],self.predictor(r['text'],r['id'])) for r in self.rows]
        result=scorer.summarize(rows)
        self.assertEqual(result['macro_f1'],1)
        self.assertEqual(result['output_contract_validity'],1)

    def test_invalid_is_not_unknown(self):
        result=SupportScorer().score('unknown',{'prediction':'__invalid__','latency_seconds':0.1})
        self.assertFalse(result['correct']);self.assertFalse(result['valid'])

    def test_model_identity_versions_checkpoint_and_decoder(self):
        model=SupportModel(**IDENTITY);other=SupportModel(**{**IDENTITY,'checkpoint_artifact':'wandb/nanochat-demo/model:v1'})
        self.assertNotEqual(model.model_dump(),other.model_dump())
        self.assertNotIn('_predictor',model.model_dump())

    def test_standard_evaluation_runs_actual_model_without_gold_argument(self):
        model=SupportModel(**IDENTITY);seen=[]
        def predict(text,example_id):
            seen.append(example_id)
            return self.predictor(text,example_id)
        model._predictor=predict
        evaluation=weave.Evaluation(dataset=[{'text':r['text'],'example_id':r['id'],'label':r['label']} for r in self.rows],scorers=[SupportScorer()])
        # With no initialized client the real Evaluation code still executes locally.
        result=asyncio.run(evaluation.evaluate(model))
        self.assertEqual(result['SupportScorer']['macro_f1'],1)
        self.assertCountEqual(seen,[r['id'] for r in self.rows])


if __name__=='__main__':unittest.main()