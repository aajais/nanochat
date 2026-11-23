"""Freeze an eight-L40 target and CPU evidence; NEVER enqueue GPU work."""
import json
from pathlib import Path
import sys
import time
import torch
import wandb
ROOT=Path('/workspace/nanochat_sft');P=ROOT/'final_workflow_v2'
sys.path.insert(0,str(P));sys.path.insert(0,'/workspace/skills/wbagent/scripts')
from nanochat.sft_runtime import atomic_json,sha256
from nanochat.gpt import GPT,GPTConfig
import launch_helpers as launch

def main():
    prior=json.loads((ROOT/'final_workflow_v1/WORKFLOW.json').read_text())
    api=wandb.Api()
    for name,pin in prior['assets'].items():
        if api.artifact(pin['ref']).digest!=pin['digest']:raise ValueError('Asset pin mismatch: '+name)
    for name in ('nanochat/conversation_eval.py','nanochat/conversation_eval_checkpoint.py','scripts/conversation_eval.py'):
        if sha256(P/name)!=prior['source_sha256'][name]:raise ValueError('Corrected evaluator source changed')
    r=api.run('wandb/nanochat-demo/b1384b1');queue=launch.get_queue('wandb','nanochat-demo-k8s')
    atomic_json(P/'decision/GPU_RECONCILIATION.json',dict(
        reason_for_prior_one_gpu='Incorrect adoption of queue one-GPU default and mistaken two-GPU-hour interpretation',
        corrected_target=dict(gpu_count=8,gpu_model='NVIDIA L40',world_size=8,nodes=1,budget_basis='shared_wall_clock'),
        historical_base_run='wandb/nanochat-demo/b1384b1',historical_config={k:r.config.get(k) for k in ('gpu_count','world_size','gpu_type','runtime_gpu')},
        queue_name='nanochat-demo-k8s',queue_default_limits=queue.get('limits'),queue_default_requests=queue.get('requests'),
        current_eight_l40_availability_verified=False,current_capacity_evidence_required=True,gpu_work_paused=True))
    cfg=GPTConfig(**json.loads((ROOT/'base_metadata/meta_133632.json').read_text())['model_config'])
    with torch.device('meta'):model=GPT(cfg);opt=model.setup_optimizer()
    groups=[]
    for i,g in enumerate(opt.param_groups):
        bad=[list(p.shape) for p in g['params'] if g['kind']=='adamw' and p.numel()>=1024 and p.shape[0]%8]
        if bad:raise ValueError('Indivisible model optimizer group')
        groups.append(dict(group=i,kind=g['kind'],parameters=len(g['params']),adamw_indivisible_shapes=bad))
    atomic_json(P/'decision/FULL_MODEL_SHAPE_AUDIT.json',dict(model_parameters=sum(p.numel() for p in model.parameters()),
        world_size=8,groups=groups,shape_compatible=True,gpu_execution_tested=False,
        limitation='Meta-device shape audit only; NOT distributed optimizer/NCCL execution proof'))
    resources=launch._make_resource_args('wandb-launch',gpus=8,cpu=8,memory='80Gi')
    resources['kubernetes']['spec'].update(activeDeadlineSeconds=7113,backoffLimit=0)
    resources['kubernetes']['spec']['template']['spec'].update(restartPolicy='Never',terminationGracePeriodSeconds=15)
    atomic_json(P/'KUBERNETES_RESOURCE_TEMPLATE.json',resources)
    for src,dst in [('final_workflow_v2-cpu-tests.log','production-cpu-tests.log'),
        ('final_workflow_v2-eight-rank-tests.log','eight-rank-contract-tests.log'),
        ('final_workflow_v2-evaluator-tests.log','corrected-evaluator-tests.log')]:
        (P/'decision'/dst).write_bytes((ROOT/src).read_bytes())
    source={str(p.relative_to(P)):sha256(p) for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
        and p.name not in ('WORKFLOW.json','APPROVAL_TEMPLATE.json') and 'wandb' not in p.parts}
    prior.update(schema='nanochat-consolidated-sft-eight-l40-v2',source_sha256=source,gpu_count=8,world_size=8,
        gpu_model='NVIDIA L40',nodes=1,budget_basis='shared_wall_clock',current_capacity_verified=False,
        gpu_readiness_verified=False,launch_authorized=False,release_approved=False,
        supersedes_job='wandb/nanochat-demo/nanochat-consolidated-conversation-sft:v0')
    atomic_json(P/'WORKFLOW.json',prior)
    approval=json.loads((ROOT/'final_workflow_v1/APPROVAL_TEMPLATE.json').read_text())
    approval['workflow_sha256']=sha256(P/'WORKFLOW.json')
    approval['decisions']['gpu_execution'].update(gpu_count=8,gpu_model='NVIDIA L40',world_size=8,
        budget_basis='shared_wall_clock',capacity_evidence=dict(eight_l40_one_node_confirmed=False,source='',observed_at_utc=''))
    assert all(d['status']=='pending' for d in approval['decisions'].values())
    atomic_json(P/'APPROVAL_TEMPLATE.json',approval)
    print(json.dumps(dict(workflow_sha256=sha256(P/'WORKFLOW.json'),source_files=len(source),gpu_launched=False)))

if __name__=='__main__':main()