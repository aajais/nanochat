import concurrent.futures,hashlib,json
from pathlib import Path
import requests
ROOT=Path('/workspace/nanochat_sft/authoritative_notices');ROOT.mkdir(exist_ok=True)
URLS={
 'metamath-license':'https://raw.githubusercontent.com/meta-math/MetaMath/main/LICENSE',
 'metamath-readme':'https://raw.githubusercontent.com/meta-math/MetaMath/main/README.md',
 'gsm8k-license':'https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE',
 'math-license':'https://raw.githubusercontent.com/hendrycks/math/main/LICENSE',
 'math-readme':'https://raw.githubusercontent.com/hendrycks/math/main/README.md',
 'llama31-license':'https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE',
 'llama31-use-policy':'https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/USE_POLICY.md',
 'nanochat-license':'https://raw.githubusercontent.com/karpathy/nanochat/master/LICENSE',
 'apache2':'https://www.apache.org/licenses/LICENSE-2.0.txt',
 'smoltalk-card':'https://huggingface.co/datasets/HuggingFaceTB/smoltalk/resolve/5feaf2fd3ffca7c237fc38d1861bc30365d48ffa/README.md',
 'metamath-card':'https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18/README.md',
 'everyday-card':'https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k/resolve/14f543216b9ba42b6b951dc5bd199460d193b162/README.md',
 'summarize-card':'https://huggingface.co/datasets/HuggingFaceTB/smol-summarize/resolve/main/README.md',
}
def fetch(item):
 name,url=item
 try:
  r=requests.get(url,timeout=25);r.raise_for_status();data=r.content
  (ROOT/(name+'.txt')).write_bytes(data)
  return name,dict(url=url,resolved_url=r.url,status=r.status_code,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
 except Exception as e:return name,dict(url=url,error=str(e))
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  results=dict(pool.map(fetch,URLS.items()))
 (ROOT/'retrieval.json').write_text(json.dumps(results,indent=2)+'\n')
 for k,v in results.items(): print(k, v.get('status'),v.get('bytes'),v.get('error',''))