from fractions import Fraction
import json
import statistics
from pathlib import Path
from time import perf_counter

from mqt.core.bench import qpe, repeat_until_success
from mqt.core.mlir import CompilerTarget, OutputFormat, compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

root = Path(__file__).parent
device = open_device('mqt.ddsim.default')
target = CompilerTarget.from_device(device)
print(json.dumps({'connectivity':str(target.connectivity_kind), 'native':str(target.native_operations_kind), 'sites':len(target.sites), 'site_restricted_operations':sum(bool(op.site_tuples) for op in target.operations)}), flush=True)
rows = []
for family, widths in [('iterative', [8, 32]), ('standard', [8, 16, 32, 64]), ('rus', [4, 16, 64])]:
    for width in widths:
        benchmark = (repeat_until_success.RepeatUntilSuccess(repeat_until_success.Options(data_qubits=width)) if family == 'rus' else qpe.QPE(qpe.Options(precision=width, phase=Fraction(3, 8), method=qpe.Method.ITERATIVE if family == 'iterative' else qpe.Method.STANDARD)))
        source = benchmark.generate()
        (root / f'{family}-{width}-input.mlir').write_text(source.ir)
        for mode in ('device', 'direct-qir'):
            row = {'family':family, 'width':width, 'mode':mode}
            durations = []
            try:
                for _ in range(3):
                    start = perf_counter()
                    if mode == 'device':
                        compiled = compile_program(source, target=device)
                        payload = compiled.payload
                    else:
                        compiled = compile_program(source, output=OutputFormat.QIR_ADAPTIVE)
                        payload = compiled.to_bitcode()
                    durations.append(perf_counter() - start)
                row |= {'seconds':durations, 'median_seconds':statistics.median(durations), 'bitcode_bytes':len(payload)}
                if mode == 'direct-qir':
                    (root / f'{family}-{width}-direct.ll').write_text(compiled.ir)
                if width <= 16:
                    job = device.submit_job(payload, ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=1024, custom1=17)
                    job.wait()
                    raw = job.get_counts()
                    counts = raw if family == 'rus' else {bits[::-1]:n for bits,n in raw.items()}
                    tvd = benchmark.evaluate(counts).total_variation_distance
                    assert tvd < (0.03 if family == 'rus' else 1e-12), (row,counts,tvd)
                    row |= {'counts':counts, 'tvd':tvd}
            except ValueError as error:
                row['error'] = str(error)
            rows.append(row)
            print(json.dumps(row),flush=True)
            (root/'baseline.json').write_text(json.dumps(rows,indent=2))
