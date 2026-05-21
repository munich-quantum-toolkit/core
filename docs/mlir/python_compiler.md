---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: "0.13"
    jupytext_version: "1.16.7"
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Python Compiler Entry Point

The Python module `mqt.core.mlir` exposes a compact compiler entry point,
`compile_program`, that routes multiple frontend formats into the MLIR-based
compiler pipeline.

```{code-cell}
from pathlib import Path
from tempfile import TemporaryDirectory

from mqt.core.ir import QuantumComputation
from mqt.core.mlir import compile_program
```

## OpenQASM Input

```{code-cell}
qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

compiled_qc = compile_program(qasm)
print("\n".join(compiled_qc.splitlines()[:12]))
```

## `QuantumComputation` Input

```{code-cell}
quantum_computation = QuantumComputation(2, 2)
quantum_computation.h(0)
quantum_computation.cx(0, 1)

compiled_qir = compile_program(quantum_computation, convert_to_qir=True)
print("\n".join(compiled_qir.splitlines()[:12]))
```

## File-Based Input (`.qasm` / `.mlir` / `.jeff`)

```{code-cell}
with TemporaryDirectory() as directory:
    qasm_path = Path(directory) / "example.qasm"
    qasm_path.write_text(qasm, encoding="utf-8")
    compiled_from_file = compile_program(qasm_path)

print("\n".join(compiled_from_file.splitlines()[:12]))
```

## Qiskit `QuantumCircuit` Input

```{code-cell}
from qiskit import QuantumCircuit

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

compiled_from_qiskit = compile_program(circuit)
print("\n".join(compiled_from_qiskit.splitlines()[:12]))
```
