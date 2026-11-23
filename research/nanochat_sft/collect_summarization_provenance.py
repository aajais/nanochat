import hashlib,json,concurrent.futures
from pathlib import Path
import requests
R=Path('/workspace/nanochat_sft/authoritative_notices')
tree=json.loads((R/'smollm-tree.txt').read_text());commit=tree['sha']
paths=['LICENSE','text/data/smoltalk/summarization/README.md','text/data/smoltalk/summarization/pipelines/cnn_daily_summaries.py','text/data/smoltalk/summarization/pipelines/email_summaries.py']
def fetch(path):
 url=f'https://raw.githubusercontent.com/huggingface/smollm/{commit}/{path}'
 r=requests.get(url,timeout=25);r.raise_for_status();name='smollm-'+Path(path).name
 (R/name).write_bytes(r.content)
 return name,dict(url=url,commit=commit,sha256=hashlib.sha256(r.content).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: data=dict(pool.map(fetch,paths))
(R/'summarization-retrieval.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))