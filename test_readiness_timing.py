"""CPU-only overhead planning and diagnostic regressions; no GPU evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from nanochat.sft_runtime import PhaseTimer
from nanochat.workflow_policy import Allocation,readiness_projection,READINESS,CAP
import workflow

def measured():
    return {'uploads':[{'checkpoint_elapsed_seconds':305.}, {'checkpoint_elapsed_seconds':305.}]}

class Timing(unittest.TestCase):
    def test_flushed_events_and_return_value(self):
        with tempfile.TemporaryDirectory() as folder:
            timer=PhaseTimer(Path(folder)/'timings.jsonl',rank=1)
            self.assertEqual(timer.call('add',lambda x:x+1,4),5)
            rows=[json.loads(line) for line in timer.path.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],['start','end'])
            self.assertGreaterEqual(rows[1]['seconds'],0)
            self.assertNotIn('arguments',rows[0])

    def test_error_recorded_and_reraised(self):
        with tempfile.TemporaryDirectory() as folder:
            timer=PhaseTimer(Path(folder)/'timings.jsonl',rank=1)
            with self.assertRaises(ZeroDivisionError):timer.call('failure',lambda:1/0)
            rows=[json.loads(line) for line in timer.path.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'],'error')
            self.assertEqual(rows[-1]['error_type'],'ZeroDivisionError')

    def test_old_budget_rejected_from_observed_cost(self):
        estimate=readiness_projection(measured(),625,140)
        self.assertFalse(estimate['fits'])
        self.assertEqual(estimate['remaining_publications'],3)
        self.assertEqual(estimate['remaining_worker_launches'],4)

    def test_proposed_budget_headroom_is_not_gpu_proof(self):
        estimate=readiness_projection(measured(),625,1700)
        self.assertTrue(estimate['fits'])
        self.assertTrue(estimate['estimate_only'])
        self.assertTrue(estimate['required_probes_unchanged'])

    def test_invalid_and_inconsistent_measurements_rejected(self):
        for cost in (None,float('nan'),float('inf'),-1,0,True):
            with self.assertRaises(ValueError):readiness_projection({'uploads':[{'checkpoint_elapsed_seconds':cost}]*2},625,1700)
        with self.assertRaises(ValueError):readiness_projection(measured(),10,1700)
        with self.assertRaises(ValueError):readiness_projection(measured(),625,-1)

    def test_total_and_downstream_reserves_unchanged(self):
        budget=Allocation(1000,now=lambda:1001)
        self.assertEqual((CAP,READINESS,budget.end),(7113,2400,8113))
        self.assertEqual(budget.end_for('main'),5953)
        self.assertEqual(budget.end_for('evaluation'),7453)
        self.assertEqual(budget.end_for('packaging'),8053)
        self.assertEqual(budget.spec()['main_cap_if_full_readiness_seconds'],2553)

    def test_headroom_failure_blocks_restored_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            budget=type('Budget',(),{'started':1000,'end_for':lambda self,stage:1100})()
            result=measured();result['identity']={}
            with patch.object(workflow,'child') as child, \
                 patch.object(workflow,'verify_result',return_value=(result,work/'checkpoint')), \
                 patch.object(workflow,'validate_checkpoint'),patch.object(workflow,'verify_partitions'), \
                 patch.object(workflow.time,'time',return_value=1000), \
                 patch.object(workflow.time,'perf_counter',side_effect=[0,625]):
                with self.assertRaisesRegex(TimeoutError,'cannot fit'):
                    workflow.readiness(work,work,work/'approval',budget,'hash',8)
                self.assertEqual(child.call_count,1)
            report=json.loads((work/'readiness-budget-projection.json').read_text())
            self.assertFalse(report['fits'])
            self.assertFalse((work/'readiness.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)