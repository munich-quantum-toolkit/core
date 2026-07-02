---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Python Compiler Entry Point

The {py:mod}`mqt.core.mlir` exposes a compact compiler entry point,
{py:func}`~mqt.core.mlir.compile_program`, that routes multiple frontend formats
into the MLIR-based compiler pipeline.

```{code-cell} ipython3
from mqt.core.mlir import compile_program
```

## OpenQASM Input

```{code-cell} ipython3
qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

result = compile_program(qasm)
print(result)
```

## File-Based Input (`.jeff` / `.mlir` / `.qasm`)

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    qasm_path = Path(directory) / "example.qasm"
    qasm_path.write_text(qasm, encoding="utf-8")
    result = compile_program(qasm_path)

print(result)
```

## `QuantumComputation` Input

```{code-cell} ipython3
from mqt.core.ir import QuantumComputation

qc = QuantumComputation(2, 2)
qc.h(0)
qc.cx(0, 1)

result = compile_program(qc)
print(result)
```

## Qiskit `QuantumCircuit` Input

```{code-cell} ipython3
from qiskit import QuantumCircuit

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

result = compile_program(circuit)
print(result)
```

## Lowering to QIR

```{code-cell} ipython3
result = compile_program(result, convert_to_qir_base=True)
print(result)
```
