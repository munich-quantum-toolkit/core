from pathlib import Path
import subprocess,hashlib,json
p=Path(__file__).parent;out={}
for kind in ['scalar','tensor']:
 samples=[]
 for i in range(24):
  r=subprocess.run([str(p/'probe-baseline'),'builder',kind,'16','1'],capture_output=True,text=True,check=True)
  text=r.stdout;h=hashlib.sha256(text.encode()).hexdigest();samples.append({'sha256':h,'source':text})
 out[kind]=samples
 print(kind,len({s['sha256'] for s in samples}),'distinct outputs from',len(samples),'processes',flush=True)
(p/'determinism.json').write_text(json.dumps(out,indent=2))
