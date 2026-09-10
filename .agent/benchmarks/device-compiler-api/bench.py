"""Run with the built package's Python; write raw timings as JSON."""
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from mqt.core import mlir
from mqt.core.qdmi.driver import open_device

source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c = measure q;'
device = open_device('mqt.ddsim.default')
compiled = mlir.compile_program(source, target=device)
results = {}

def measure(name, call):
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
        if hasattr(result, 'wait'):
            result.wait()
            counts = result.get_counts()
            assert sum(counts.values()) == 16 and set(counts) <= {'00', '11'}
    results[name] = {'seconds': samples, 'median': statistics.median(samples), 'min': min(samples), 'max': max(samples)}

measure('snapshot', lambda: mlir.CompilerTarget.from_device(device))
measure('compiled_submit', lambda: device.submit(compiled, num_shots=16))
measure('raw_submit', lambda: device.submit_job(compiled.payload, compiled.program_format, 16))
measure('source_submit', lambda: device.submit(source, num_shots=16))
for size in (256, 1024, 4096):
    connectivity = mlir.CompilerTarget.Connectivity([(i-1, i) for i in range(1, size)])
    measure(f'chain_construct_{size}', lambda: mlir.CompilerTarget(size, connectivity=connectivity, native_operations=mlir.CompilerTarget.NativeOperations.unrestricted()))
report = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'python': sys.version, 'platform': platform.platform(), 'module': mlir.__file__, 'module_sha256': hashlib.sha256(Path(mlir.__file__).read_bytes()).hexdigest(), 'results': results}
print(json.dumps(report, indent=2))
