---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Exchange programs with jeff

[jeff](https://github.com/unitaryfoundation/jeff) is an interchange format for
quantum compilers. It represents quantum operations together with classical
computation and structured control flow, so a receiving compiler can continue
working on the program before choosing an execution format or device.

MQT Core imports and exports jeff through
[jeff-mlir](https://github.com/unitaryfoundation/jeff-mlir), the shared MLIR
dialect and serialization implementation. The Python wheels include this
integration; using the APIs below requires no separate jeff installation.

For a guided experiment with a compiler handoff, full-unitary comparison, and a
measurement-dependent loop, use the
{doc}`jeff tutorial <tutorials/jeff_exchange>`. This page describes the
interfaces and supported boundary.

## Where jeff fits

| Representation             | Role in an MQT Core workflow                                               |
| -------------------------- | -------------------------------------------------------------------------- |
| OpenQASM or Qiskit circuit | Provide a program through a source language or SDK.                        |
| QC and QCO                 | Translate and optimize within the MQT Compiler Collection.                 |
| jeff                       | Exchange a structured program with another compatible compiler or process. |
| QIR                        | Express a program for a compatible execution runtime.                      |
| QDMI job payload           | Submit the format accepted by a selected device.                           |

The serialized jeff format uses
[Cap'n Proto](https://github.com/unitaryfoundation/jeff/blob/main/impl/capnp/jeff.capnp).
MQT Core's `JeffProgram` holds the corresponding MLIR program in memory. Its
`ir` property prints that MLIR representation; `to_bytes()` and `write()`
produce the binary exchange format.

## Compile and serialize

Pass an accepted compiler input to `compile_program` with `OutputFormat.JEFF`.
The same call accepts a Qiskit `QuantumCircuit`, an OpenQASM source string, a
supported input file, or a typed compiler program:

```{code-cell} ipython3
from mqt.core.mlir import JeffProgram, OutputFormat, compile_program, sample

source = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
bit[2] result = measure q;
"""

program = compile_program(source, output=OutputFormat.JEFF)
payload = program.to_bytes()
assert isinstance(payload, bytes)
print(f"Serialized jeff program: {len(payload)} bytes")
```

Inspect `program.ir` when debugging the MLIR representation. To control each
stage explicitly, use `qco.to_jeff(copy=True)` on an existing
{py:class}`~mqt.core.mlir.QCOProgram`. Omitting `copy=True` consumes it.

## Load a buffer or file

Deserialize bytes with `JeffProgram.from_bytes`. The returned program can enter
the compiler pipeline again:

```{code-cell} ipython3
received = JeffProgram.from_bytes(payload)
qc = compile_program(received, output=OutputFormat.QC)
counts = sample(qc, shots=64, seed=17)
assert set(counts) <= {"00", "11"} and sum(counts.values()) == 64
counts
```

Use `write` and `from_file` for an explicit file handoff. Alternatively, pass a
`Path` ending in `.jeff` directly to `compile_program`:

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "bell.jeff"
    program.write(path)
    loaded = JeffProgram.from_file(path)
    optimized = compile_program(path, output=OutputFormat.QCO_OPTIMIZED)
    assert sample(loaded, shots=64, seed=17) == sample(optimized, shots=64, seed=17)
```

Raw byte buffers go through `from_bytes` before compilation. `compile_program`
uses strings for source text; a `Path` makes file input explicit.

## Continue to a device or another output format

A received jeff program is a compiler input. Select a device to obtain a
compatible payload, then submit it:

```{code-cell} ipython3
from mqt.core.mlir import submit_program
from mqt.core.qdmi.driver import open_device

device = open_device("mqt.ddsim.default")
compiled = compile_program(received, target=device)
job = submit_program(compiled, target=device, num_shots=64, custom1=17)
assert job.wait()
assert set(job.get_counts()) <= {"00", "11"}
print(compiled.program_format.name, job.get_counts())
```

Choose `OutputFormat.OPENQASM3`, `QIR_BASE`, or `QIR_ADAPTIVE` instead when an
explicit output representation is required. The
{doc}`target-compilation guide <mlir/target_compilation>` covers device
constraints; the {doc}`QIR guide <qir/index>` covers LLVM text and bitcode.

The command-line driver also reads and writes jeff:

```console
mqt-cc input.qasm --emit=jeff -o exchange.jeff
mqt-cc exchange.jeff --emit=openqasm3 -o output.qasm
mqt-cc exchange.jeff --emit=qir-adaptive -o output.bc
```

## Supported conversion boundary

Exchange requires both compilers to support the program's operations and format
version. The upstream format is extensible; MQT Core does not implement every
possible extension. The conversion currently supports:

- Quantum gates, modifiers in the supported normal form, measurements, and
  classical computation and structured control flow used in the tutorial.
- Integer expressions through 64 bits. Integer absolute value and power require
  jeff's native widths: 1, 8, 16, 32, or 64.
- Scalar branch results and loop state, including different loop input and
  result tuples. Straight-line array snapshots are preserved.
- Defined single-block functions and calls. Imported helper functions become
  private, while the designated entry point remains public.

The following limits affect interchange:

- Import rejects live old array values across mutating control flow and shared
  array updates inside switch or while regions.
- Quantum allocations and deallocations inside conditional regions, and mutable
  classical-register arguments in helpers, are unsupported.
- Export converts physical static qubits to allocations, so static site IDs do
  not survive the round trip. Exchange logical programs before device mapping.
- Only selected custom operations and Pauli-product rotations are supported.
- Unitary calls inside quantum modifiers need expansion before export. The
  coordinated compiler pipeline performs modifier preparation. A direct pass
  requires each remaining modifier to contain one unitary using all body
  arguments in order.

See the {doc}`conversion reference <mlir/Conversions>` for the pass contracts
and
[upstream jeff documentation](https://github.com/unitaryfoundation/jeff/tree/main/docs)
for the format. SSA names and textual IR can change during conversion; compare
program semantics when validating a handoff.
