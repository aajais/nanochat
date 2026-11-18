"""Structured, serial Weave evaluations around real prediction execution.

No weights, tokenizer objects, or secrets are serialized. Checkpoint identity is
part of the published Model, not a mutable module-level model name.
"""
import asyncio
import hashlib
import warnings
from pydantic import PrivateAttr
import weave
from eval_core import LABELS, PROMPT, score


class SupportModel(weave.Model):
    checkpoint_artifact: str
    model_weight_sha256: str
    tokenizer_artifact: str
    method: str
    prompt_template: str = PROMPT
    max_new_tokens: int = 12
    backend: str = 'nanochat'
    _predictor: object = PrivateAttr(default=None)

    @weave.op()
    def predict(self, text: str, example_id: str) -> dict:
        # Only request fields reach the model. The expected label is scorer-only.
        return self._predictor(text, example_id)


class SupportScorer(weave.Scorer):
    @weave.op()
    def score(self, label: str, output: dict) -> dict:
        prediction = output['prediction']
        return {'expected': label, 'prediction': prediction,
                'correct': prediction == label,
                'valid': prediction in LABELS,
                'latency_seconds': output['latency_seconds']}

    @weave.op()
    def summarize(self, score_rows: list[dict]) -> dict:
        # Compute F1 from the pooled confusion matrix, not mean row-wise F1.
        result = score([{'label': x['expected']} for x in score_rows],
                       [x['prediction'] for x in score_rows],
                       [x['latency_seconds'] for x in score_rows])
        result['output_contract_validity'] = 1 - result['invalid_rate']
        return result


def execute(rows, predictor, identity, *, dataset_artifact, split,
            init_fn=None, trace_enabled=True):
    """Run once. Tracing-init failure preserves inference; never replay a model.

    Missing backend evidence is *not* promotion eligibility: the external gate
    independently verifies the completed evaluation and every row/model child.
    """
    trace_error = None
    client = None
    if trace_enabled:
        try:
            client = (init_fn or weave.init)('wandb/nanochat-demo')
        except Exception as exc:
            trace_error = type(exc).__name__
            warnings.warn('Weave unavailable; inference continues, promotion blocked: ' + trace_error)
    else:
        trace_error = 'tracing_unconfigured'
    outputs = {}

    def once(text, example_id):
        if example_id in outputs:
            raise ValueError('Duplicate model execution; no automatic prediction retries')
        result = predictor(text, example_id)
        outputs[example_id] = result
        return result

    receipt = {'backend_verified': False, 'trace_error': trace_error,
               'row_count': len(rows), 'split': split}
    if client is None:
        for row in rows:
            once(row['text'], row['id'])
    else:
        model = SupportModel(**identity)
        model._predictor = once
        dataset = weave.Dataset(
            name='support-intent-' + split + '-rows',
            rows=[{'example_id': x['id'], 'text': x['text'], 'label': x['label']}
                  for x in rows])
        evaluation = weave.Evaluation(
            name='support-intent-' + split + '-' + identity['method'],
            dataset=dataset, scorers=[SupportScorer()], trials=1,
            metadata={'dataset_artifact': dataset_artifact, 'split': split,
                      'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                      'inference_mode': 'live', 'serial_model_execution': True,
                      'training_performed': False})

        async def run():
            return await evaluation.evaluate.call(evaluation, model)

        summary, call = asyncio.run(run())
        receipt.update(evaluation_call_id=call.id, trace_id=call.trace_id,
                       model_ref=str(model.ref),
                       dataset_ref=str(dataset.ref), summary=summary,
                       trace_url=f'https://wandb.ai/wandb/nanochat-demo/weave/calls/{call.id}')
        # Flush once, but deliberately do NOT claim backend verification here.
        try:
            weave.finish()
        except Exception as exc:
            receipt['trace_error'] = type(exc).__name__
    if set(outputs) != {x['id'] for x in rows}:
        raise ValueError('Incomplete evaluation predictions; no success/promotion')
    return [outputs[row['id']] for row in rows], receipt