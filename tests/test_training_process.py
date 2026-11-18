"""CPU regressions for transfer staging and accumulation telemetry; no service calls."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from nanochat import dataloader


class Tokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, docs, prepend, **kwargs):
        return [[prepend] + doc for doc in docs]


def documents(*args, **kwargs):
    i = 1
    while True:
        yield [[i, i + 1]], (0, 0, 1)
        i += 2


class TestTransferStaging(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'requires a real CUDA device')
    def test_real_cuda_tokens_match_cpu(self):
        outputs = []
        with patch.object(dataloader, '_document_batches', documents):
            for device in (torch.device('cpu'), torch.device('cuda')):
                loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
                    Tokenizer(), 2, 3, 'train', device=device, buffer_size=2)
                outputs.append([(x.clone(), y.clone()) for x, y, _ in (next(loader) for _ in range(12))])
        torch.cuda.synchronize()
        for cpu, cuda in zip(*outputs):
            self.assertTrue(torch.equal(cpu[0], cuda[0].cpu()))
            self.assertTrue(torch.equal(cpu[1], cuda[1].cpu()))

    def test_cpu_device_forms_preserve_tokens(self):
        results = []
        with patch.object(dataloader, '_document_batches', documents), patch.object(torch.cuda, 'Event') as event:
            for device in ('cpu', torch.device('cpu')):
                loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
                    Tokenizer(), 2, 3, 'train', device=device, buffer_size=2)
                results.append([(x.clone(), y.clone()) for x, y, _ in (next(loader) for _ in range(3))])
            event.assert_not_called()
        for actual, expected in zip(*results):
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_cuda_forms_pin_and_wait_before_reusing_staging(self):
        # Model an outstanding async H2D read without requiring a CUDA driver.
        # synchronize() checks that its CPU source was not overwritten early.
        real_empty = torch.empty
        for device in ('cuda', torch.device('cuda'), torch.device('cuda:0')):
            with self.subTest(device=device):
                state = {'pinned': [], 'waits': 0, 'records': 0}

                class DeviceBuffer:
                    def __init__(self, *args, **kwargs):
                        self.tensor = real_empty(*args, device='cpu', dtype=kwargs['dtype'])

                    def __getitem__(self, key):
                        return self.tensor[key]

                    def copy_(self, source, non_blocking=False):
                        self_outer.assertTrue(non_blocking)
                        state['source'], state['snapshot'] = source, source.clone()
                        self.tensor.copy_(source)

                class CopyEvent:
                    def record(self, stream):
                        state['records'] += 1

                    def synchronize(self):
                        self_outer.assertTrue(torch.equal(state['source'], state['snapshot']),
                                              'CPU staging was mutated before H2D completed')
                        state['waits'] += 1

                def empty(*args, **kwargs):
                    if torch.device(kwargs.get('device', 'cpu')).type == 'cuda':
                        return DeviceBuffer(*args, **kwargs)
                    state['pinned'].append(kwargs.pop('pin_memory', False))
                    return real_empty(*args, **kwargs)

                self_outer = self
                with patch.object(dataloader, '_document_batches', documents), \
                     patch.object(torch, 'empty', empty), patch.object(torch.cuda, 'Event', CopyEvent), \
                     patch.object(torch.cuda, 'current_stream', return_value=object()):
                    loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
                        Tokenizer(), 1, 3, 'train', device=device, buffer_size=2)
                    batches = []
                    for _ in range(3):
                        x, y, _ = next(loader)
                        batches.append((x.clone(), y.clone()))
                self.assertIn(True, state['pinned'])
                self.assertEqual(state['records'], 3)
                self.assertEqual(state['waits'], 2)
                self.assertFalse(torch.equal(batches[0][0], batches[1][0]))


class TestAccumulationTelemetry(unittest.TestCase):
    @staticmethod
    def run_accumulation(scaler=None):
        # Execute the actual micro-step AST, not a second implementation. Importing
        # base_train would run training and initialize devices at module scope.
        source = Path(__file__).parents[1] / 'scripts' / 'base_train.py'
        tree = ast.parse(source.read_text())
        loop = next(n for n in tree.body if isinstance(n, ast.While))
        start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 't0' for t in n.targets)) + 1
        end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'lrm' for t in n.targets))
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()

        def loss_model(x, y):
            return torch.nn.functional.mse_loss(model(x), y)

        inputs = torch.ones(1, 1)
        targets = [torch.tensor([[1.]]), torch.tensor([[3.]])]
        env = {'torch': torch, 'device': torch.device('cpu'), 'model': loss_model,
               'x': inputs, 'y': targets[0], 'grad_accum_steps': 2, 'scaler': scaler,
               'train_loader': iter([(inputs, targets[1], {}), (inputs, targets[0], {})])}
        exec(compile(ast.Module(body=loop.body[start:end], type_ignores=[]), str(source), 'exec'), env)
        return env['train_loss'].item(), model.weight.grad.item()

    def test_mean_loss_and_unchanged_gradient(self):
        loss, gradient = self.run_accumulation()
        self.assertAlmostEqual(loss, 5.)  # mean of losses1 and9, not last loss9
        self.assertAlmostEqual(gradient, -4.)

    def test_scaler_does_not_scale_reported_loss(self):
        class Scaler:
            def scale(self, loss):
                return loss * 8
        loss, gradient = self.run_accumulation(Scaler())
        self.assertAlmostEqual(loss, 5.)
        self.assertAlmostEqual(gradient / 8, -4.)


if __name__ == '__main__':
    unittest.main()