"""CPU-only plumbing tests. These are not model-quality experiments."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from nanochat.sft_runtime import (PackedLoader, save_sft, restore_sft,
                                 validate_checkpoint, lr_multiplier, Deadline)
from scripts.prepare_conversation_sft import NearIndex, validate_messages, fingerprint, normalize

ROOT = Path(__file__).resolve().parents[1]


def records():
    return [{'ids':[0,1,2,3,4,5], 'mask':[0,0,0,1,1,1], 'category':'general'},
            {'ids':[0,1,6,7,8,5], 'mask':[0,0,0,1,1,1], 'category':'general'}]


def make_components():
    model=torch.nn.Sequential(torch.nn.Embedding(16,8),torch.nn.Dropout(.2),torch.nn.Linear(8,16))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    # The production checkpoint verifier records optimizer group kinds. This
    # synthetic replay fixture uses the AdamW branch, not a Muon shard proof.
    for group in optimizer.param_groups:
        group['kind'] = 'adamw'
    loader=PackedLoader(records(),12,1,0,{'general':1},seed=42)
    return model,optimizer,loader


def update(model,optimizer,loader):
    x,y=loader.next('cpu')
    optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=-1)
    loss.backward()
    optimizer.step()
    return float(loss.detach()),x,y


class PreparationTests(unittest.TestCase):
    def test_optimized_simhash_matches_reference(self):
        for text in ['', 'short text', ' '.join(str(i) for i in range(250)),
                     'One repeated phrase. '*100, 'Mixed CAPITALS, punctuation and Unicode café.']:
            words=normalize(text).split()
            shingles=set(' '.join(words[i:i+5]) for i in range(max(1,len(words)-4)))
            bits=[0]*64
            for shingle in shingles:
                h=int.from_bytes(hashlib.blake2b(shingle.encode(),digest_size=8).digest(),'big')
                for i in range(64):bits[i]+=1 if h&(1<<i) else -1
            expected=sum(1<<i for i,value in enumerate(bits) if value>=0)
            self.assertEqual(fingerprint(text),expected)

    def test_missing_explicit_approval_blocks_before_compute_or_wandb(self):
        from scripts import chat_sft_resumable as trainer
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            Path(folder,'manifest.json').write_text(json.dumps({'release_gate':{'training_approved':False}}))
            args=SimpleNamespace(max_updates=30,calibration_steps=12,overhead_reserve_fraction=.25,
                                 save_reserve_seconds=10,deadline_unix=time.time()+1000,
                                 data_dir=folder,tokenizer_dir=folder,cpu_fixture=False,
                                 approval_file=str(Path(folder)/'missing-approval.json'))
            with patch.object(trainer,'parser') as parser, patch.object(trainer,'compute_init') as compute, patch.object(trainer.wandb,'init') as initialize:
                parser.return_value.parse_args.return_value=args
                with self.assertRaises(FileNotFoundError):
                    trainer.main()
                compute.assert_not_called()
                initialize.assert_not_called()

    def test_mask_does_not_corrupt_input(self):
        loader=PackedLoader(records(),12,1,0,{'general':1})
        x,y=loader.next('cpu')
        self.assertTrue(bool((x>=0).all()))
        self.assertTrue(bool((y==-1).any()))
        self.assertTrue(bool((y==5).any()))  # assistant EOT remains supervised

    def test_exact_cpu_resume(self):
        torch.manual_seed(7); random.seed(7); np.random.seed(7)
        model,optimizer,loader=make_components()
        for _ in range(3): update(model,optimizer,loader)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            checkpoint=save_sft(tmp,3,model,optimizer,loader,{'step':3,'horizon':10,'warmup_steps':2},{'test':'v1'})
            expected=[update(model,optimizer,loader)[0] for _ in range(3)]
            expected_weights=copy.deepcopy(model.state_dict())
            model2,optimizer2,loader2=make_components()
            restored=restore_sft(checkpoint,model2,optimizer2,loader2,{'test':'v1'},'cpu')
            self.assertEqual(restored['loop']['step'],3)
            actual=[update(model2,optimizer2,loader2)[0] for _ in range(3)]
            self.assertEqual(expected,actual)
            for key in expected_weights:
                self.assertTrue(torch.equal(expected_weights[key],model2.state_dict()[key]))

    def test_reject_corruption_wrong_world_and_identity(self):
        model,optimizer,loader=make_components()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            folder=save_sft(tmp,0,model,optimizer,loader,{'step':0},{'test':'v1'})
            with self.assertRaises(ValueError): validate_checkpoint(folder,{'test':'v1'},2)
            with self.assertRaises(ValueError): validate_checkpoint(folder,{'test':'v2'},1)
            with open(folder/'rank-0000.pt','ab') as f: f.write(b'bad')
            with self.assertRaises(ValueError): validate_checkpoint(folder,{'test':'v1'},1)

    def test_incomplete_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaises(FileNotFoundError): validate_checkpoint(tmp,{},1)

    def test_schedule_and_deadline(self):
        rates=[lr_multiplier(i,100,3) for i in range(100)]
        self.assertEqual(rates[2],1)
        self.assertEqual(rates[-1],0)
        self.assertTrue(all(a>=b for a,b in zip(rates[50:],rates[51:])))
        deadline=Deadline(time.time()+100,10)
        self.assertFalse(deadline.should_stop('cpu'))
        deadline._signal()
        self.assertTrue(deadline.should_stop('cpu'))

    def test_message_and_dedup_guards(self):
        self.assertIsNone(validate_messages([{'role':'user','content':'Hi'}, {'role':'assistant','content':'Hello'}]))
        self.assertIsNotNone(validate_messages([{'role':'user','content':'Hi'}]))
        self.assertIsNotNone(validate_messages([{'role':'user','content':'Hi'}, {'role':'assistant','content':[]}]))
        index=NearIndex(); index.add('Some simple text')
        self.assertTrue(index.contains('SOME  simple text!'))
        self.assertFalse(index.contains('A different prompt'))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)