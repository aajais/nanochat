import concurrent.futures,hashlib,json
from pathlib import Path
import requests
R=Path('/workspace/nanochat_sft/authoritative_notices')
def get(name,url):
 r=requests.get(url,timeout=25);r.raise_for_status();(R/(name+'.txt')).write_bytes(r.content)
 return dict(url=url,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content))
jobs={
 'metamath-readme-fixed':'https://raw.githubusercontent.com/meta-math/MetaMath/main/README.MD',
 'smollm-tree':'https://api.github.com/repos/huggingface/smollm/git/trees/main?recursive=1',
 'metamath-hf-tree':'https://huggingface.co/api/datasets/meta-math/MetaMathQA/tree/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18',
 'metamath-git-tree':'https://api.github.com/repos/meta-math/MetaMath/git/trees/main?recursive=1',
 'nanochat-pinned-license':'https://raw.githubusercontent.com/karpathy/nanochat/58d716e6255a5373f917ea9ca5e933d030db1779/LICENSE',
}
def task(item):
 n,u=item
 try:return n,get(n,u)
 except Exception as e:return n,{'url':u,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: results=dict(pool.map(task,jobs.items()))
(R/'followup-retrieval.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
tree=json.loads((R/'smollm-tree.txt').read_text())
print('SMOLLM_RELEVANT_PATHS',[(t['path'],t['sha']) for t in tree.get('tree',[]) if any(s in t['path'].lower() for s in ('summar','smoltalk','notice','license'))][:70])