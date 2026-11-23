import unittest
from final_workflow_budget import *

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.t = 0
        self.b = Budget(10000., 0., monotonic=lambda:self.t)
    def proof(self):
        return dict(passed=True,completed_elapsed_seconds=self.t,**{k:True for k in REQUIRED_PROOFS})
    def test_exact_shared_limits(self):
        s=specification()
        self.assertEqual(s['remaining_allocation_hard_cap_seconds'],7113)
        self.assertEqual(s['main_seconds_if_readiness_uses_full_cap'],6333)
        self.assertLessEqual(87+780,900)
        self.assertEqual(self.b.stage_limits('readiness')['work_seconds_before_reserve'],615)
    def test_startup_consumes_readiness_budget(self):
        self.t=90
        self.assertEqual(self.b.stage_limits('readiness')['work_seconds_before_reserve'],525)
    def test_main_does_not_reset_deadline(self):
        self.t=700
        r=self.b.authorize_main(self.proof())
        self.assertEqual(r['remaining_hard_seconds'],6413)
        self.assertEqual(r['absolute_allocation_deadline_unix'],17113)
    def test_readiness_failure_skips_main(self):
        calls=[]
        with self.assertRaises(RuntimeError):
            gated_stages(self.b,lambda _:dict(passed=False),lambda _:calls.append('main'))
        self.assertFalse(calls)
    def test_missing_each_proof_blocks(self):
        for key in REQUIRED_PROOFS:
            p=self.proof();p.pop(key)
            with self.assertRaises(RuntimeError):self.b.authorize_main(p)
    def test_exception_skips_main(self):
        calls=[]
        def worker(_):raise ChildProcessError('nonzero child')
        with self.assertRaises(ChildProcessError):gated_stages(self.b,worker,lambda _:calls.append('main'))
        self.assertFalse(calls)
    def test_overrun_blocks_main(self):
        self.t=781
        with self.assertRaises(TimeoutError):self.b.authorize_main(self.proof())
    def test_reported_overrun_blocks(self):
        p=self.proof();p['completed_elapsed_seconds']=781
        with self.assertRaises(TimeoutError):self.b.authorize_main(p)
    def test_no_budget_for_save_blocks(self):
        self.t=620
        with self.assertRaises(TimeoutError):self.b.stage_limits('readiness')
        self.t=6500
        with self.assertRaises(TimeoutError):self.b.stage_limits('main')
    def test_negative_clock_blocks(self):
        self.t=-1
        with self.assertRaises(RuntimeError):self.b.remaining()
    def test_single_successful_handoff(self):
        calls=[]
        def ready(_):
            self.t=500
            calls.append('readiness')
            return self.proof()
        def main(limits):
            calls.append('main')
            return limits
        result=gated_stages(self.b,ready,main)
        self.assertEqual(calls,['readiness','main'])
        self.assertEqual(result['remaining_hard_seconds'],6613)

if __name__ == '__main__': unittest.main()