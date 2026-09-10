from fractions import Fraction
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from mqt.core.bench import qpe, repeat_until_success
from mqt.core.mlir import OutputFormat, compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device
root=Path(__file__).parent
device=open_device('mqt.ddsim.default')
rows=[]
for family,width in [('standard',8),('standard',16),('rus',4),('rus',16)]:
    b=(repeat_until_success.RepeatUntilSuccess(repeat_until_success.Options(data_qubits=width)) if family=='rus' else qpe.QPE(qpe.Options(precision=width,phase=Fraction(3,8))))
    for mode in ('device','direct-qir'):
        p=b.generate()
        c=compile_program(p,target=device) if mode=='device' else compile_program(p,output=OutputFormat.QIR_ADAPTIVE)
        payload=c.payload if mode=='device' else c.to_bitcode()
        times=[]
        for _ in range(3):
            start=perf_counter()
            job=device.submit_job(payload,ProgramFormat.QIR_ADAPTIVE_MODULE,num_shots=1024,custom1=17)
            job.wait()
            counts=job.get_counts()
            times.append(perf_counter()-start)
            if family!='rus': counts={s[::-1]:n for s,n in counts.items()}
            assert b.evaluate(counts).total_variation_distance < (0.03 if family=='rus' else 1e-12)
        row={'family':family,'width':width,'mode':mode,'shots':1024,'seconds':times,'median_seconds':median(times)}
        rows.append(row)
        print(json.dumps(row),flush=True)
        (root/'execution.json').write_text(json.dumps(rows,indent=2))
