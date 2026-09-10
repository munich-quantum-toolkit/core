from pathlib import Path
import subprocess,json,statistics,hashlib
p=Path(__file__).parent;rows=[]
for n in [128,256,512,1024]:
 expected=None
 for pair in range(5):
  for version in (['baseline','noreset'] if pair%2==0 else ['noreset','baseline']):
   output=p/f'used-{n}-{version}.out.mlir'
   r=subprocess.run(['taskset','-c','19',str(p/f'probe-{version}'),'canonicalize-qco','qasm',str(p/'inputs'/f'used-{n}.mlir'),'3',str(output)],capture_output=True,text=True,check=True,timeout=60)
   source=output.read_text();digest=hashlib.sha256(source.encode()).hexdigest()
   if expected is None:expected=digest
   assert digest==expected,(n,version,'output differs')
   assert source.count('qco.reset ')==n
   for sample in json.loads(r.stdout)['samples']:rows.append({'workload':'used-slot-reset','size':n,'variant':version,'pair':pair,'ms':sample['ms'][0],'ops':sample['ops'],'sha256':digest})
 print('reset',n,{v:round(statistics.median(x['ms'] for x in rows if x['size']==n and x['variant']==v),3) for v in ['baseline','noreset']},flush=True)
for n in [64,256,1024,4096]:
 for kind in ['empty','extracted']:
  for pair in range(5):
   r=subprocess.run(['taskset','-c','19',str(p/'probe-baseline'),'builder-prep',kind,str(n),'3'],capture_output=True,text=True,check=True,timeout=60)
   for sample in json.loads(r.stdout)['samples']:rows.append({'workload':'builder-preparation','size':n,'variant':kind,'pair':pair,'ms':sample['ms'][0],'ops':sample['ops']})
  print('builder',kind,n,round(statistics.median(x['ms'] for x in rows if x['workload']=='builder-preparation' and x['size']==n and x['variant']==kind),3),flush=True)
(p/'results.json').write_text(json.dumps(rows,indent=2))
