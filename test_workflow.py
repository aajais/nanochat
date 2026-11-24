"""CPU production-path tests. Tiny native GPT/MuonAdamW, never GPU evidence."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.update(WANDB_MODE='disabled',NANOCHAT_DTYPE='float32',TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1')
import torch
from nanochat.gpt import GPT,GPTConfig
from nanochat.sft_runtime import atomic_json,sha256,validate_checkpoint
from nanochat.workflow_policy import Allocation,CAP,READINESS,authorize
from workflow import child,compare_replay

def fixture(root):
    for name in ('data','tokenizer','base'):(root/name).mkdir()
    (root/'tokenizer/tokenizer.pkl').write_bytes(b'synthetic fixture; never unpickle')
    rows=[dict(ids=[0,1,2+i,3,4,5+i,6],mask=[0,0,0,0,1,1,1],category='general') for i in range(8)]
    for name in ('train.jsonl','dev.jsonl'):(root/'data'/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=dict(context=16,tokenizer_sha256=sha256(root/'tokenizer/tokenizer.pkl'),effective_shares={'general':1},
                  files={name:sha256(root/'data'/name) for name in ('train.jsonl','dev.jsonl')})
    atomic_json(root/'data/manifest.json',manifest)
    cfg=dict(sequence_len=16,vocab_size=256,n_layer=2,n_head=1,n_kv_head=1,n_embd=64,window_pattern='L')
    torch.manual_seed(93);model=GPT(GPTConfig(**cfg));model.init_weights()
    torch.save(model.state_dict(),root/'base/model_133632.pt');atomic_json(root/'base/meta_133632.json',{'model_config':cfg})

class BudgetAuthorization(unittest.TestCase):
    def test_reserves(self):
        b=Allocation(1000,now=lambda:1001)
        self.assertEqual(b.end,8113)
        self.assertEqual(b.end_for('main'),5953)
        self.assertEqual(b.end_for('readiness'),3400)
        self.assertEqual(b.spec()['main_cap_if_full_readiness_seconds'],2553)
        self.assertEqual(b.end_for('packaging')-b.end_for('evaluation'),600)
        self.assertEqual(b.end_for('evaluation')-b.end_for('main'),1500)

    def test_no_clock_reset(self):
        for start in (float('nan'),float('inf'),2000,-10000):
            with self.assertRaises(TimeoutError):Allocation(start,now=lambda:1000)
        with self.assertRaises(TimeoutError):Allocation(1000,now=lambda:1000+CAP)

    def test_owner_is_not_approval(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'approval.json';atomic_json(path,dict(workflow_sha256='x',assets={},scope='internal demo only',owner='Aayush Jaiswal'))
            with self.assertRaises(PermissionError):authorize(path,'x',{})

    def test_mechanical_ranking_not_newest(self):
        from scripts.workflow_evaluate import rank_development
        score=dict(case_results=[dict(pass_=True)],response_results=[])
        score['case_results']=[{'case_id':'one','pass':True,'capability':'instruction'},{'case_id':'two','pass':None,'capability':'human_review'}]
        score['response_results']=[dict(case_id='one',check_pass=True,boundary_ok=False,loop_free=False)]
        self.assertEqual(rank_development(score),(1,0,0))

class Handoff(unittest.TestCase):
    def test_readiness_failure_cannot_start_main(self):
        import workflow
        with patch.object(workflow,'readiness',side_effect=ValueError('failed proof')),patch.object(workflow,'child') as launch:
            with self.assertRaises(ValueError):workflow.run_stages(None,Path('/workspace/mock'),None,Allocation(time.time()),'sha',1)
            launch.assert_not_called()

    def test_upload_failure_cannot_start_main(self):
        import workflow
        with patch.object(workflow,'readiness',return_value=Path('/workspace/proof.json')),\
             patch.object(workflow,'prepare_training_plan'),\
             patch.object(workflow,'upload_evidence',side_effect=RuntimeError('upload failed')),patch.object(workflow,'child') as launch:
            with self.assertRaises(RuntimeError):workflow.run_stages(None,Path('/workspace/mock'),None,Allocation(time.time()),'sha',1)
            launch.assert_not_called()

    def test_main_failure_cannot_evaluate_or_package(self):
        import workflow
        with patch.object(workflow,'readiness',return_value=Path('/workspace/proof.json')),\
             patch.object(workflow,'prepare_training_plan'),\
             patch.object(workflow,'upload_evidence'),patch.object(workflow,'child',side_effect=RuntimeError('worker failed')) as launch:
            with self.assertRaises(RuntimeError):workflow.run_stages(None,Path('/workspace/mock'),Path('/workspace/approval'),Allocation(time.time()),'sha',1)
            self.assertEqual(launch.call_count,1)

    def test_success_is_single_sequential_handoff(self):
        import workflow
        with patch.object(workflow,'readiness',return_value=Path('/workspace/proof.json')),\
             patch.object(workflow,'prepare_training_plan'),\
             patch.object(workflow,'upload_evidence'),patch.object(workflow,'verify_result'),patch.object(workflow,'child') as launch:
            workflow.run_stages(None,Path('/workspace/mock'),Path('/workspace/approval'),Allocation(time.time()),'sha',1)
            self.assertEqual(launch.call_count,3)
            args=[call.args[0] for call in launch.call_args_list]
            self.assertIn('scripts.chat_sft_resumable',args[0]);self.assertIn('--readiness-evidence',args[0])
            self.assertIn('scripts.workflow_evaluate',args[1]);self.assertIn('scripts.workflow_package',args[2])

class ProductionWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='production-cpu-')
        cls.root=Path(cls.temp.name);fixture(cls.root);cls.end=time.time()+150
        cls.argv=[sys.executable,'-m','scripts.chat_sft_resumable','--data-dir',str(cls.root/'data'),
            '--dataset-artifact','synthetic/fixture/cache:v0','--tokenizer-dir',str(cls.root/'tokenizer'),
            '--base-dir',str(cls.root/'base'),'--deadline-unix',str(cls.end),'--save-reserve-seconds','3',
            '--max-updates','20','--calibration-steps','1','--context','16','--global-input-tokens','32',
            '--no-compile','--device-type','cpu','--cpu-fixture','--fixed-horizon','4','--stage','readiness',
            '--save-every-seconds','0','--eval-every','0']

    def launch(self,name,extra=(),signal_at_step=False,expect_failure=False,argv=None):
        folder=self.root/name
        command=list(argv or self.argv)+['--output-dir',str(folder)]+list(extra)
        result=child(command,self.end,folder/'worker.log',signal_at_step=signal_at_step,expect_failure=expect_failure)
        return folder,result

    def test_a_periodic_fresh_process_next_update_exact(self):
        a,_=self.launch('continuous',['--stop-after-step','2'])
        b,_=self.launch('resumed',['--resume',str(a/'step-00000001'),'--stop-after-step','2'])
        compare_replay(a/'step-00000002',b/'step-00000002')
        ar=json.loads((a/'result.json').read_text())
        self.assertTrue(ar['passed']);self.assertEqual(ar['loop']['step'],2)
        self.assertFalse(ar['uploads'][0]['server_verified'])
        cp=a/'step-00000001';meta=json.loads((cp/'loop.json').read_text())
        validate_checkpoint(cp,meta['identity'],1)
        self.__class__.complete_checkpoint=cp

    def test_b_real_sigterm_saves(self):
        argv=list(self.argv);argv[argv.index('--save-every-seconds')+1]='999999'
        folder,result=self.launch('signal',['--pause-after-step','1'],True,argv=argv)
        self.assertTrue(result['signal_sent'])
        self.assertEqual(json.loads((folder/'result.json').read_text())['loop']['stop_reason'],'signal')
        self.assertTrue((folder/'step-00000001/COMPLETE.json').is_file())

    def test_c_real_deadline_saves(self):
        argv=list(self.argv);argv[argv.index('--deadline-unix')+1]=str(time.time()+9)
        argv[argv.index('--save-every-seconds')+1]='999999'
        folder,_=self.launch('deadline',['--pause-after-step','1'],argv=argv)
        self.assertEqual(json.loads((folder/'result.json').read_text())['loop']['stop_reason'],'deadline')
        self.assertTrue((folder/'step-00000001/COMPLETE.json').is_file())

    def test_d_worker_failure_propagates(self):
        folder,result=self.launch('failure',['--inject-failure-step','1'],expect_failure=True)
        self.assertNotEqual(result['exit_code'],0)
        self.assertFalse(json.loads((folder/'failure.json').read_text())['passed'])
        self.assertFalse((folder/'result.json').exists())

    def test_e_corrupt_restore_refused(self):
        cp=self.root/'corrupt';shutil.copytree(self.complete_checkpoint,cp)
        with (cp/'rank-0000.pt').open('ab') as stream:stream.write(b'corrupt')
        folder,result=self.launch('bad-resume',['--resume',str(cp),'--stop-after-step','2'],expect_failure=True)
        self.assertNotEqual(result['exit_code'],0);self.assertFalse((folder/'result.json').exists())

    def test_f_supervisor_hard_timeout(self):
        log=self.root/'timeout/worker.log'
        with self.assertRaises(TimeoutError):child([sys.executable,'-c','import time; time.sleep(20)'],time.time()+.3,log)

    def test_g_upload_production_code_mocked_transport(self):
        # Execute actual production entrypoint and upload branch in-process;
        # replace only authorization/tokenizer/service boundaries for tiny CPU.
        from scripts import chat_sft_resumable as trainer
        class Tokenizer:
            def get_vocab_size(self):return 256
            def get_bos_token_id(self):return 0
        class FakeArtifact:
            def __init__(self,*args,**kwargs):self.name='mock:v0';self.version='v0';self.digest='mock-digest'
            def add_dir(self,path):
                self.path=Path(path)
                self.manifest=type('Manifest',(),{'entries':{p.name:None for p in self.path.iterdir()}})()
            def wait(self):return self
            def download(self,root):shutil.copytree(self.path,root);return root
        class Run:
            id='mockrun';summary={}
            def log(self,*a,**k):pass
            def use_artifact(self,*a,**k):pass
            def log_artifact(self,artifact,**kwargs):self.artifact=artifact;return artifact
            def finish(self,exit_code=0):self.exit_code=exit_code
        for corrupt in (False,True):
            run=Run();folder=self.root/('upload-fail' if corrupt else 'upload-ok')
            argv=[x for x in self.argv[3:] if x!='--cpu-fixture']+['--output-dir',str(folder),'--stop-after-step','1']
            def fetch(ref):
                art=run.artifact
                if corrupt:art.digest='different'
                return art
            # A separate published object prevents a mocked digest changing both sides.
            def artifact_fetch(ref):
                import copy
                art=__import__('copy').copy(run.artifact)
                if corrupt:art.digest='different'
                return art
            with patch.object(sys,'argv',['trainer']+argv),patch.object(trainer,'approved_worker'),\
                 patch.object(trainer,'verify_hardware'),\
                 patch.object(trainer.RustBPETokenizer,'from_directory',return_value=Tokenizer()),\
                 patch.object(trainer.wandb,'init',return_value=run),patch.object(trainer.wandb,'Artifact',FakeArtifact),\
                 patch.object(trainer.wandb,'Api') as api:
                api.return_value.artifact.side_effect=artifact_fetch
                if corrupt:
                    with self.assertRaises(ValueError):trainer.main()
                    self.assertEqual(run.exit_code,1);self.assertFalse((folder/'result.json').exists())
                else:
                    trainer.main();self.assertEqual(run.exit_code,0)
                    self.assertTrue(json.loads((folder/'result.json').read_text())['uploads'][0]['server_verified'])

if __name__=='__main__':unittest.main(verbosity=2)
