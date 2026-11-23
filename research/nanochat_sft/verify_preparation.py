import json,sys
from pathlib import Path
import torch

root=Path('/workspace/nanochat_sft')
sys.path.insert(0,str(root/'prepared'))
from scripts.prepare_conversation_sft import prepare


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    else:
        assert a==b,(path,a,b)


results={}
for kind,ranks in [('native_single',1),('protocol_two',2)]:
    for rank in range(ranks):
        a=torch.load(root/f'checks/{kind}/save-rank-{rank}.pt',weights_only=False)
        b=torch.load(root/f'checks/{kind}/resume-rank-{rank}.pt',weights_only=False)
        equal(a,b)
    results[kind]={'status':'PASS','ranks':ranks,'exact_comparison':
                  ['model','optimizer','loader','loop','losses','python_numpy_torch_rng']}
results['native_two_cpu']={'status':'BLOCKED','reason':'Gloo Work::getFuture not implemented; NCCL path untested'}
fixture=root/'checks/data_fixture.jsonl'
with open(fixture,'w') as f:
    for i in range(240):
        marker=f'{i:08x}'
        row={'messages':[{'role':'user','content':'Return marker '+marker},
                         {'role':'assistant','content':marker}],
             'category':['general','context','reasoning'][i%3],
             'source':'unit-test-fixture-only','revision':'fixture-v1'}
        f.write(json.dumps(row)+'\n')
    # Exact duplicate, protected ARC question, malformed, empty, and tool-shaped examples.
    f.write(json.dumps(row)+'\n')
    q=json.loads((root/'arc_inputs/benchmark-inputs.json').read_text())['examples'][0]['query']
    f.write(json.dumps({**row,'messages':[{'role':'user','content':q},{'role':'assistant','content':'not for training'}]})+'\n')
    f.write(json.dumps({**row,'messages':[{'role':'user','content':'incomplete'}]})+'\n')
    f.write(json.dumps({**row,'messages':[{'role':'user','content':'empty'},{'role':'assistant','content':''}]})+'\n')
manifest=prepare([str(fixture)],root/'checks/prepared_fixture',root/'tokenizer',
                 [root/'arc_inputs/benchmark-inputs.json'])
assert manifest['filter_counts']['protected_overlap']==1
assert manifest['filter_counts']['duplicate_or_near_duplicate']>=1
results['data_preparation_fixture']={'status':'PASS','warning':'Synthetic plumbing fixture, not a training dataset',
                                     'filter_counts':manifest['filter_counts'],
                                     'statistics':manifest['statistics']}
(root/'preparation_tests.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))