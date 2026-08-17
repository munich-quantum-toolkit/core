# Compile for a QDMI device

An MLIR {code}`mlir::CompilerTarget` is an immutable snapshot of a circuit-model
device. It contains the device sites, topology, native operations, and available
calibration data. Compilation decomposes supported multi-qubit operations,
optimizes and maps the program, synthesizes native gates, and verifies that the
result conforms to the target.

The snapshot is independent of its originating QDMI session. It can therefore be
stored, copied cheaply, and reused for multiple compilations.

## Python

Open a configured QDMI device and snapshot it as a compiler target:

```python
from mqt.core.mlir import CompilerTarget, OutputFormat, compile_program

target = CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
compiled = compile_program(
    "bell.qasm",
    target=target,
    output=OutputFormat.QCO_OPTIMIZED,
)
```

Target compilation accepts optimized QCO, QC, or QIR output and uses the
canonical QCO pipeline; it cannot be combined with a custom `qco_pipeline`.

The target can also be constructed directly. Omitting `couplings` selects
all-to-all connectivity; omitting `operations` means that every operation is
native:

```python
target = CompilerTarget(3, couplings=[(0, 1), (1, 2)])
```

Use {py:meth}`~mqt.core.mlir.QCOProgram.compile_for_target` to apply target
compilation to an existing QCO program. For pass-level benchmarking, the C++ API
exposes separate factories for pre-routing optimization, mapping, native
synthesis, and conformance verification.

## Command line from a source build

List the stable IDs of configured QDMI devices:

```console
mqt-cc --qdmi-list-devices
```

Select a device when compiling:

```console
mqt-cc --qdmi-device=mqt.sc.iqm.garnet \
  --emit=qco-optimized input.qasm
```

An explicit registry file can be selected before device discovery:

```console
mqt-cc --qdmi-config=/path/to/qdmi.json \
  --qdmi-device=example.device input.qasm
```

Target compilation produces optimized QCO, QC, or QIR. It cannot be combined
with a custom `--passes` pipeline because the canonical target pipeline owns the
required pass ordering.

## C++ source-tree API

The source build provides a narrow, non-throwing QDMI bridge between a stable
device ID and the compiler-owned target:

```cpp
#include "mlir/Compiler/QDMIAdapter.h"
#include "mlir/Compiler/Programs.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

auto target = mlir::compilerTargetFromDeviceId("mqt.sc.iqm.garnet");
if (!target) {
  llvm::errs() << "Failed to create compiler target: "
               << llvm::toString(target.takeError()) << '\n';
  return 1;
}

auto qc = mlir::QCProgram::fromQASMFile("input.qasm");
if (!qc) {
  return 1;
}
auto qco = std::move(*qc).intoQCO();
if (!qco || !qco->compileForTarget(*target)) {
  return 1;
}
```

The adapter accepts circuit-model devices whose operations are available
throughout the topology in both operand orientations. Operand-symmetric gates,
such as CZ, may report each edge once. Neutral-atom zone models require a
different compilation model and are rejected with a diagnostic.

The bundled Garnet and Emerald snapshots contain available T1, T2, and fidelity
data. Operation durations are absent because they were unavailable. See
{doc}`../qdmi/sc_device` for their stable IDs and {doc}`../qdmi/configuration`
for registry configuration.

If the program should use fewer physical qubits, run the {code}`mqt-qubit-reuse`
pipeline before target compilation.

## Physical target extent

Mapping records the physical target address extent on the MLIR module as
{code}`mqt.target_qubit_extent`. The unsigned 64-bit value is one greater than
the largest target site identifier. It is an exclusive upper bound, not the
number of target sites. For example, sites 5, 9, and 17 have an extent of 18.

Cleanup can remove unused {code}`qco.static` and {code}`qco.sink` pairs without
losing this metadata. Conversion from QCO to QC retains the module attribute.
Native QC-to-Qiskit export uses the extent as the minimum circuit width. Thus, a
program that uses two sites of a five-site dense target exports as a five-qubit
Qiskit circuit without retaining three dead static-qubit operations.
