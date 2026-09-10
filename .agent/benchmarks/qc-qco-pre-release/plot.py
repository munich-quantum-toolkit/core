from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent;rows=json.loads((p/'results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,workload,variants,title in [(axes[0],'used-slot-reset',['baseline','noreset'],'Previously used slots: canonicalization'),(axes[1],'builder-preparation',['empty','extracted'],'Builder: preparing register arguments')]:
 for variant in variants:
  sizes=sorted({r['size'] for r in rows if r['workload']==workload})
  q=np.array([np.quantile([r['ms'] for r in rows if r['workload']==workload and r['variant']==variant and r['size']==n],[.25,.5,.75]) for n in sizes])
  label={'baseline':'Upstream main','noreset':'Unsuccessful fold disabled (diagnostic)','empty':'No extracted qubits (control)','extracted':'One extracted qubit per register'}[variant]
  ax.plot(sizes,q[:,1],'-o',label=label);ax.fill_between(sizes,q[:,0],q[:,2],alpha=.18)
 ax.set(xscale='log',yscale='log',xlabel='Slots / registers',ylabel='Time (ms)',title=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('QC/QCO pre-release audit · ad74680f1 · medians and interquartile ranges')
fig.savefig(p/'results.png',dpi=170)
