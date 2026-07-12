---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Python Compiler Programs

The {py:mod}`mqt.core.mlir` module exposes the compiler's intermediate
representations as move-aware Python objects. A {py:class}`QCProgram`,
{py:class}`QCOProgram`, {py:class}`JeffProgram`, or {py:class}`QIRProgram` owns
its MLIR module and prints as textual MLIR. Dialect-changing methods consume a
program by default, which avoids copying a potentially large module. Pass
`copy=True` when a source program should remain available for another pipeline
branch.

```{code-cell} ipython3
from mqt.core.mlir import OutputFormat, QCProgram, compile_program
```

## Frontends and QC

`QCProgram` represents QC immediately after frontend translation. It can be
created from OpenQASM 3, MLIR, an MQT `QuantumComputation`, or a Qiskit
`QuantumCircuit`.

```{code-cell} ipython3
qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

qc = QCProgram.from_qasm_str(qasm)
print(qc.ir)
```

The high-level function accepts the same frontends, including `.qasm`, `.mlir`,
and `.jeff` paths. Use `QC_IMPORT` to stop immediately after a QC frontend, or
the default `QC` format for the fully optimized QCO round trip.

```{code-cell} ipython3
imported = compile_program(qasm, output=OutputFormat.QC_IMPORT)
optimized_qc = compile_program(qasm)
```

## Building a Custom Pipeline

Each cleanup and conversion is available as a small typed operation. This makes
pipeline construction explicit while transferring ownership rather than copying
the module. Here `qc` stays valid because the conversion was requested with
`copy=True`; `qco` is consumed by `to_qc()`.

```{code-cell} ipython3
qco = qc.to_qco(copy=True)
qco.cleanup()
qco.optimize(enable_hadamard_lifting=True)
final_qc = qco.to_qc()

assert qc.is_valid
assert not qco.is_valid
print(final_qc)
```

The methods correspond directly to the compiler stages:

- `QCProgram.cleanup()`, `QCProgram.to_qco()`, and `QCProgram.to_qir()`
- `QCOProgram.cleanup()`, `QCOProgram.optimize()`, `QCOProgram.to_qc()`, and
  `QCOProgram.to_jeff()`
- `JeffProgram.cleanup()` and `JeffProgram.to_qco()`
- `QIRProgram.cleanup()` and `QIRProgram.llvm_ir`

## Jeff Serialization

Jeff serialization is now a capability of {py:class}`JeffProgram`, rather than a
separate compiler entry point. The generic compiler returns a `JeffProgram`,
which can be stored as bytes or written to a file and later restored.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "example.jeff"
    jeff = compile_program(qasm, output=OutputFormat.JEFF)
    path.write_bytes(jeff.to_bytes())
    restored = compile_program(path, output=OutputFormat.QC)

print(restored.ir)
```

## QIR

Both QIR profiles are first-class pipeline targets. `QIRProgram` keeps the
lowered MLIR available through `ir` and provides LLVM IR with `llvm_ir`.

```{code-cell} ipython3
base_qir = compile_program(qasm, output=OutputFormat.QIR_BASE)
print(base_qir.llvm_ir)
```
