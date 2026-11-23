"""CPU structural tests, NOT eight-GPU execution evidence."""
import os
os.environ.update(NANOCHAT_DTYPE='float32',TORCHDYNAMO_DISABLE='1')
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
from nanochat.distributed_readiness import token_plan,verify_hardware,optimizer_partition
from nanochat.sft_runtime import atomic_json,atomic_torch,sha256,validate_checkpoint
from nanochat.workflow_policy import Allocation
import workflow

def proof():
    return dict(world_size=8,same_world_size_verified=True,first_result=dict(measurements=[
        dict(world_size=8,global_input_tokens=16384,assistant_tokens=8000,seconds=2) for _ in range(12)]))

class Contract(unittest.TestCase):
    def test_reserve_aware_measured_token_plan(self):
        p=token_plan(proof(),1000,5173)
        self.assertEqual(p['expected_main_updates'],1339)
        self.assertEqual(p['expected_input_tokens'],1339*16384)
        self.assertEqual(p['expected_supervised_assistant_tokens'],1339*8000)
        self.assertFalse(p['quality_guarantee'])

    def test_one_rank_or_missing_measurements_rejected(self):
        p=proof();p['world_size']=1
        with self.assertRaises(ValueError):token_plan(p,0,4173)
        p=proof();p['first_result']['measurements']=p['first_result']['measurements'][:5]
        with self.assertRaises(ValueError):token_plan(p,0,4173)

    def test_hardware_mismatch_rejected(self):
        for n,name,world in [(1,'NVIDIA L40',8),(8,'NVIDIA H100',8),(8,'NVIDIA L40',1)]:
            with patch.object(torch.cuda,'device_count',return_value=n),patch.object(torch.cuda,'get_device_name',return_value=name):
                with self.assertRaises(ValueError):verify_hardware(world)

    def test_main_and_readiness_same_eight_rank_torchrun(self):
        for stage in ('readiness','main'):
            cmd=workflow.training_command(Path('work'),Path('assets'),Path('approval'),Allocation(time.time()),stage,gpus=8)
            self.assertIn('--nproc_per_node=8',cmd);self.assertIn('torch.distributed.run',cmd)
            self.assertIn('scripts.chat_sft_resumable',cmd)

    def test_sigterm_targets_all_workers_not_launcher(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            atomic_json(Path(d)/'progress.json',dict(worker_pids=list(range(100,108))))
            with patch.object(workflow.os,'kill') as kill,patch.object(workflow.os,'killpg') as group:
                workflow.terminate_workers(type('P',(),{'pid':99})(),d)
                self.assertEqual(kill.call_count,8);group.assert_not_called()

    def test_missing_eighth_checkpoint_shard_rejected(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            root=Path(d);identity={'code':'synthetic-test-only'}
            atomic_torch(root/'model.pt',{});atomic_json(root/'loop.json',dict(world_size=8,identity=identity))
            for rank in range(7):atomic_torch(root/f'rank-{rank:04d}.pt',{})
            atomic_json(root/'COMPLETE.json',{p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in root.iterdir()})
            with self.assertRaises(ValueError):validate_checkpoint(root,identity,8)

    def test_adamw_partition_shape(self):
        p=torch.nn.Parameter(torch.zeros(16,128));o=type('FixtureOptimizer',(),{})()
        o.param_groups=[dict(kind='adamw',params=[p])];o.state={p:dict(exp_avg=torch.zeros(2,128),exp_avg_sq=torch.zeros(2,128))}
        for rank in range(8):self.assertTrue(optimizer_partition(o,rank,8)['validated'])
        o.state[p]['exp_avg']=torch.zeros(16,128)
        with self.assertRaises(ValueError):optimizer_partition(o,0,8)

    def test_muon_ownership_with_padding(self):
        ps=[torch.nn.Parameter(torch.zeros(16,8)) for _ in range(10)];o=type('FixtureOptimizer',(),{})()
        o.param_groups=[dict(kind='muon',params=ps)]
        o.state={ps[0]:dict(momentum_buffer=torch.zeros(2,16,8),second_momentum_buffer=torch.zeros(2,16,1))}
        counts=[optimizer_partition(o,r,8)['groups'][0]['entries'][0]['owned_count'] for r in range(8)]
        self.assertEqual(counts,[2,2,2,2,2,0,0,0])

    def test_one_rank_optimizer_evidence_cannot_pass_eight_rank_gate(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            for rank in range(8):atomic_torch(Path(d)/f'rank-{rank:04d}.pt',{'optimizer_partition':dict(rank=rank,world_size=1,validated=True,optimizer_class='MuonAdamW')})
            with self.assertRaises(ValueError):workflow.verify_partitions(d,8)

    def test_token_plan_saved_logged_before_handoff(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            root=Path(d);path=root/'proof.json';atomic_json(path,proof())
            run=type('Run',(),{'log':lambda self,record:setattr(self,'record',record)})()
            with patch('builtins.print') as printed:
                workflow.prepare_training_plan(run,root,path,Allocation(time.time()),'sha')
            plan=json.loads((root/'training-token-plan.json').read_text())
            self.assertTrue(plan['reported_before_handoff']);self.assertEqual(plan['readiness_sha256'],sha256(path))
            self.assertIn('plan/expected_input_tokens',run.record);printed.assert_called_once()

if __name__=='__main__':unittest.main(verbosity=2)