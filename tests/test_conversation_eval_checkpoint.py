"""CPU-only loader tests. Tiny random tensors are NOT nanochat quality evidence."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from nanochat import conversation_eval_checkpoint as checkpoint


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'weights.pt').write_bytes(b'fabricated weights, never deserialized')
        (self.root/'meta.json').write_text(json.dumps({'model_config': checkpoint.MODEL_CONFIG}))
        (self.root/'tokenizer.pkl').write_bytes(b'fabricated tokenizer, never unpickled')
        self.weights = checkpoint.sha256(self.root/'weights.pt')
        self.metadata = checkpoint.sha256(self.root/'meta.json')
        self.tokenizer = checkpoint.sha256(self.root/'tokenizer.pkl')
        self.sources = {name: checkpoint.sha256(Path(checkpoint.__file__).parent/name)
                        for name in checkpoint.SOURCE_NAMES}
        self.spec = dict(schema='nanochat-eval-checkpoint-v1', kind='base',
                         model_config=checkpoint.MODEL_CONFIG, base_weights_sha256=self.weights,
                         checkpoint_sha256=self.weights, weights_file='weights.pt',
                         metadata_file='meta.json', tokenizer_sha256=self.tokenizer,
                         source_sha256=self.sources)

    def verify(self):
        manifest = self.root/'manifest.json'
        manifest.write_text(json.dumps(self.spec))
        with patch.multiple(checkpoint, BASE_WEIGHTS=self.weights, BASE_METADATA=self.metadata,
                            TOKENIZER=self.tokenizer):
            return checkpoint.verify_manifest(self.root, manifest, checkpoint.sha256(manifest), self.root)

    def test_base_preflight(self):
        self.assertEqual(self.verify()[1], self.root/'weights.pt')

    def test_wrong_manifest_hash(self):
        manifest = self.root/'manifest.json'
        manifest.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            checkpoint.verify_manifest(self.root, manifest, '0'*64, self.root)

    def test_wrong_model(self):
        self.spec['model_config'] = dict(checkpoint.MODEL_CONFIG, n_layer=12)
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_tokenizer(self):
        (self.root/'tokenizer.pkl').write_bytes(b'tampered')
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_source(self):
        self.spec['source_sha256'] = dict(self.sources, **{'gpt.py': '0'*64})
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_weights(self):
        (self.root/'weights.pt').write_bytes(b'tampered')
        with self.assertRaises(ValueError): self.verify()

    def test_path_escape(self):
        with self.assertRaisesRegex(ValueError, 'escapes root'):
            checkpoint.checked_file(self.root, '../outside.pt', '0'*64)

    def test_gpu_refused_before_loading(self):
        with self.assertRaises(PermissionError):
            checkpoint.load_verified(self.root, self.root/'missing', trusted_manifest_sha256='0'*64,
                                     tokenizer_dir=self.root, device='cuda')
    def test_base_manifest(self):
        self.assertEqual(self.verify()[1], self.root/'weights.pt')

    def test_untrusted_manifest(self):
        self.verify()
        with self.assertRaises(ValueError):
            checkpoint.verify_manifest(self.root, self.root/'manifest.json', '0'*64, self.root)

    def test_wrong_weights(self):
        (self.root/'weights.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_metadata(self):
        (self.root/'meta.json').write_text('{}')
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_tokenizer(self):
        (self.root/'tokenizer.pkl').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_source(self):
        self.spec['source_sha256']['gpt.py'] = '0'*64
        with self.assertRaises(ValueError): self.verify()

    def test_missing_source(self):
        del self.spec['source_sha256']['common.py']
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_architecture(self):
        self.spec['model_config'] = dict(checkpoint.MODEL_CONFIG, n_layer=12)
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_base_lineage(self):
        self.spec['base_weights_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.verify()

    def test_path_escape(self):
        self.spec['weights_file'] = '/absolute.pt'
        with self.assertRaises(ValueError): self.verify()
        self.spec['weights_file'] = '../outside.pt'
        with self.assertRaises(ValueError): self.verify()

    def test_gpu_refused_before_load(self):
        with self.assertRaises(PermissionError):
            checkpoint.load_verified(self.root, 'does-not-exist', trusted_manifest_sha256='0'*64,
                                     tokenizer_dir=self.root, device='cuda')

    def make_sft(self):
        self.spec['kind'] = 'sft'
        self.spec['weights_file'] = 'model.pt'
        (self.root/'model.pt').write_bytes((self.root/'weights.pt').read_bytes())
        loop = {'world_size': 2, 'identity': {'base_weights_sha256': self.weights,
                                           'tokenizer_sha256': self.tokenizer,
                                           'base_metadata_sha256': self.metadata,
                                           'code': {'nanochat/' + name: value
                                                    for name, value in self.sources.items()}}}
        (self.root/'loop.json').write_text(json.dumps(loop))
        for rank in range(2):
            (self.root/f'rank-{rank:04d}.pt').write_bytes(b'optimizer shards must not be unpickled')
        names = ['model.pt', 'loop.json', 'rank-0000.pt', 'rank-0001.pt']
        complete = {name: {'sha256': checkpoint.sha256(self.root/name),
                           'bytes': (self.root/name).stat().st_size} for name in names}
        (self.root/'COMPLETE.json').write_text(json.dumps(complete))
        self.spec.update(kind='sft', complete_file='COMPLETE.json',
                         complete_sha256=checkpoint.sha256(self.root/'COMPLETE.json'))

    def test_complete_sft_without_optimizer_deserialization(self):
        self.make_sft()
        with patch.object(torch, 'load', side_effect=AssertionError('No deserialization in preflight')):
            self.assertEqual(self.verify()[1], self.root/'model.pt')

    def test_missing_sft_shard(self):
        self.make_sft()
        (self.root/'rank-0001.pt').rename(self.root/'hidden-rank.pt')
        with self.assertRaises(ValueError): self.verify()

    def test_corrupt_sft_shard(self):
        self.make_sft()
        (self.root/'rank-0000.pt').write_bytes(b'bad')
        with self.assertRaises(ValueError): self.verify()

    def test_incomplete_sft_manifest(self):
        self.make_sft()
        complete = json.loads((self.root/'COMPLETE.json').read_text())
        del complete['rank-0001.pt']
        (self.root/'COMPLETE.json').write_text(json.dumps(complete))
        self.spec['complete_sha256'] = checkpoint.sha256(self.root/'COMPLETE.json')
        with self.assertRaises(ValueError): self.verify()


class NativeTensorTests(unittest.TestCase):
    def test_pinned_digest_shapes(self):
        for value in (checkpoint.BASE_WEIGHTS, checkpoint.BASE_METADATA, checkpoint.TOKENIZER):
            self.assertRegex(value, r'^[0-9a-f]{64}$')

    def test_verified_identity_cannot_be_replaced_accidentally(self):
        from types import SimpleNamespace
        evaluator = checkpoint.VerifiedEvaluator(
            SimpleNamespace(identity={'checkpoint_sha256': 'a'*64}), None,
            {'checkpoint_sha256': 'b'*64})
        with self.assertRaisesRegex(ValueError, 'identity was changed'):
            evaluator.run_development('unused', output_path='unused', ledger_path='unused')

    @classmethod
    def setUpClass(cls):
        from nanochat.gpt import GPT, GPTConfig
        torch.set_num_threads(2)
        cls.config = dict(sequence_len=16, vocab_size=64, n_layer=2,
                          n_head=1, n_kv_head=1, n_embd=32, window_pattern='L')
        torch.manual_seed(123)
        cls.original = GPT(GPTConfig(**cls.config))
        cls.original.init_weights()
        cls.original.eval()

    def test_native_forward_equivalence(self):
        model = checkpoint.strict_model(self.original.state_dict(), self.config)
        ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(ids), self.original(ids)))
        self.assertEqual(model.cos.device.type, 'cpu')
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_native_backend_equivalence(self):
        from nanochat.conversation_eval import NativeGreedyBackend
        model = checkpoint.strict_model(self.original.state_dict(), self.config)
        actual = list(NativeGreedyBackend(model, synthetic=True).stream([1, 2], 3))
        expected, prefix = [], [1, 2]
        with torch.inference_mode():
            for _ in range(3):
                token = int(self.original(torch.tensor([prefix]))[0, -1].argmax())
                expected.append(token)
                prefix.append(token)
        self.assertEqual(actual, expected)

    def test_key_mismatch(self):
        state = dict(self.original.state_dict())
        state.pop('smear_lambda')
        with self.assertRaises(ValueError): checkpoint.strict_model(state, self.config)

    def test_shape_mismatch(self):
        state = dict(self.original.state_dict(), smear_lambda=torch.zeros(2))
        with self.assertRaises(ValueError): checkpoint.strict_model(state, self.config)

    def test_nonfinite(self):
        state = dict(self.original.state_dict(), smear_lambda=torch.tensor([float('nan')]))
        with self.assertRaises(ValueError): checkpoint.strict_model(state, self.config)


if __name__ == '__main__':
    unittest.main()