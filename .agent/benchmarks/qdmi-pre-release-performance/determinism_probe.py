import hashlib
import json
import os

from mqt.core.mlir import compile_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

padding = [
    bytearray((i % 511) + 1) for i in range(int(os.environ["PYTHONHASHSEED"]) % 1000)
]
device = open_device("mqt.ddsim.default")
source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[3] c; h q[0]; cx q[0],q[2]; rx(0.3) q[1]; c = measure q;'
result = {}
for fmt in [
    ProgramFormat.QASM3,
    ProgramFormat.QIR_BASE_MODULE,
    ProgramFormat.QIR_ADAPTIVE_MODULE,
]:
    program = compile_program(source, target=device, program_format=fmt)
    data = program.payload
    if isinstance(data, str):
        data = data.encode()
    result[str(fmt)] = hashlib.sha256(data).hexdigest()
print(json.dumps(result, sort_keys=True))
