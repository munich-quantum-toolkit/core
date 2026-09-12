# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list
of changes including minor and patch releases, please refer to the
[changelog](CHANGELOG.md).

## [Unreleased]

OpenQASM import no longer emits runtime assertions, and QIR conversion no longer
lowers explicit `cf.assert` operations. Validate runtime inputs before
execution: classical indices must stay in bounds after negative-index wrapping,
range steps must be nonzero, integer arithmetic powers require nonnegative
exponents, and integer `pow` modifier exponents must be exactly representable as
`f64`. Static diagnostics remain. Runtime integer powers now wrap at their
machine width, matching other runtime integer arithmetic.

## [4.0.0]

### Migrating from MQT Core 3 to 4

MQT Core 4 makes the **MQT Compiler Collection**, built on **MLIR and LLVM**,
the foundation for quantum program representation, transformation, and
compilation. This is a major architectural and API migration from v3's classic
circuit representation. The compiler represents quantum operations together with
classical arithmetic, reusable functions, structured control flow, and
measurement feedback, then lowers supported programs for exchange or execution.

Python circuit users move from `mqt.core.ir` and `mqt.core.load` to
`mqt.core.mlir`. C++ circuit users adopt the compiler's source-tree interfaces
or stay on the v3 release series. The low-level DD and QDMI libraries remain
available; their migration requirements depend on whether they consume classic
circuits or operate directly on states, matrices, devices, and jobs.

The [v4 release overview](CHANGELOG.md#unreleased) describes the new compiler
capabilities. This guide explains how to adopt them in existing applications.

This section describes changes since **v3.10.0**. When upgrading from an older
version, also consult the intervening release sections for changes that affect
APIs you still use. In particular, Python 3.11, Apple silicon on macOS 13.3+,
nanobind 3 split-mode wheels, and the FoMaC, CircuitOptimizer, and neutral-atom
removals were already part of v3.10.0; they are not new v4 migrations.

If a downstream package still needs classic circuits, constrain its dependency
to `mqt-core>=3,<4`. Use a v3 tag or version constraint for C++ consumers too.
Use separate virtual environments or installation prefixes for the two versions.

### Choose a migration path

| Your current use                                                           | Start here                                                                                                                                    |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Load, construct, or convert Python circuits                                | [Python circuit loading and conversion](#python-circuit-loading-and-conversion)                                                               |
| Pass classic circuits to DD simulation or functionality helpers            | [Decision-diagram simulation and functionality](#decision-diagram-simulation-and-functionality)                                               |
| Link `MQT::CoreIR` or `MQT::CoreQASM`, or embed MQT Core in a native build | [C++ libraries and build requirements](#c-libraries-and-build-requirements)                                                                   |
| Discover QDMI devices, submit OpenQASM/QIR jobs, or query results          | [QDMI submission and QIR execution](#qdmi-submission-and-qir-execution)                                                                       |
| Use raw DD states and matrices without classic circuits                    | Keep the low-level DD API; check [native build requirements](#c-libraries-and-build-requirements) and [other API changes](#other-api-changes) |

Port one representative program through import, compilation, and execution
before migrating the rest of an application. Check its unitary, statevector, or
sampled outputs as appropriate, then update downstream dependencies that still
require the classic circuit types. The
[executable compiler tutorials](https://mqt.readthedocs.io/projects/core/en/latest/tutorials/index.html)
cover these workflows without requiring prior MLIR knowledge.

### Understand the new compiler model

In v3, circuit-facing APIs shared a `QuantumComputation` object and its mutable
operations. In v4, typed program objects own compiler representations:

- **QC** uses references to qubits and serves as the interoperability
  representation for frontends such as OpenQASM and Qiskit.
- **QCO** uses linear quantum values, with each value consumed exactly once. It
  is the main representation for quantum transformations. QTensor and CBit
  provide quantum and classical registers; MLIR supplies classical arithmetic,
  functions, and structured control flow.
- **Output programs** carry the requested representation: QC or QCO for further
  compiler work, OpenQASM source, jeff for structured exchange, or QIR that can
  be serialized as LLVM text or bitcode.

Use `compile_program` to run the shared pipeline and choose where it stops with
`OutputFormat`. Supplying a device target adds compilation for its operations,
topology, and payload contract. Submission is a separate step, so a compiled
payload can be reused while that contract still matches the destination. The
[compiler guide](https://mqt.readthedocs.io/projects/core/en/latest/mlir/mqt_compiler_collection.html)
also covers explicit passes and custom pipelines.

Code that rewrites operation lists must be adapted to program builders or
compiler passes that preserve the representations' type, control-flow, and
quantum-value invariants. Representation-changing methods can consume their
input, so ownership is also part of the migration. Each frontend, conversion,
and execution path has a supported subset; representing a program in MLIR does
not guarantee that every output format or device can execute it.

### Python circuit loading and conversion

MQT Core removes `mqt.core.ir`, `mqt.core.load`, and the classic `mqt_to_qiskit`
and `qiskit_to_mqt` converters. There are no compatibility aliases for
`QuantumComputation` or its operation and register types.

| Previous use                         | MQT Core 4 replacement                                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `mqt.core.load(path)` for OpenQASM   | `QCProgram.from_openqasm_file(path)` or `compile_program(Path(path), ...)`              |
| Import OpenQASM source text          | `QCProgram.from_openqasm_str(source)`                                                   |
| `qiskit_to_mqt(circuit)`             | `QCProgram.from_qiskit(circuit)` or `compile_program(circuit, ...)`                     |
| `mqt_to_qiskit(circuit)`             | `QCProgram.to_qiskit()` or `QCOProgram.to_qiskit()` on the corresponding program        |
| Construct or edit classic operations | QC/QCO program builders and compiler passes; there is no drop-in operation-list adapter |
| Dump classic operations as OpenQASM  | `compile_program(program, output=OutputFormat.OPENQASM3).source`                        |

Pass a `Path` for file input to `compile_program`; a plain string denotes source
text. This example loads, compiles, and inspects a circuit without a file:

```python
from mqt.core.mlir import OutputFormat, QCProgram, compile_program

source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0], q[1];'
qc = QCProgram.from_openqasm_str(source)
qco = compile_program(qc, output=OutputFormat.QCO_OPTIMIZED)
assert qc.is_valid and qco.is_valid
print(qco.ir)
```

Supported OpenQASM and Qiskit inputs are described in the
[OpenQASM guide](https://mqt.readthedocs.io/projects/core/en/latest/mlir/OpenQASM.html)
and
[Qiskit guide](https://mqt.readthedocs.io/projects/core/en/latest/mlir/qiskit.html).

`compile_program` preserves input program objects by default. Direct
representation-changing methods such as `qc.to_qco()` consume their source; pass
`copy=True` when retaining it. Do not use a consumed program or borrowed MLIR
handles from it. Converting to a Qiskit circuit preserves the source.

### Decision-diagram simulation and functionality

The classic circuit-taking functions in `mqt.core.dd`, including `simulate`,
`simulate_statevector`, `sample`, `build_unitary`, and `build_functionality`,
are removed. So are `DDPackage` methods that take classic operation objects. Use
the compiler-backed functions in `mqt.core.mlir`:

| Previous `mqt.core.dd` call        | Compiler-backed replacement                            |
| ---------------------------------- | ------------------------------------------------------ |
| `simulate_statevector(qc)`         | `mqt.core.mlir.simulate(source)`                       |
| `build_unitary(qc)`                | `mqt.core.mlir.build_functionality(source)`            |
| `sample(qc, shots, seed)`          | `mqt.core.mlir.sample(source, shots=shots, seed=seed)` |
| `simulate(qc, initial, package)`   | `qco.simulate(initial, package)` for a `QCOProgram`    |
| `build_functionality(qc, package)` | `qco.build_functionality(package)` for a `QCOProgram`  |

```python
import numpy as np

from mqt.core.mlir import build_functionality, sample, simulate

source = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; h q;'
state = simulate(source)
unitary = build_functionality(source)
counts = sample(source, shots=128, seed=17)
np.testing.assert_allclose(state, np.array([1, 1]) / np.sqrt(2), atol=1e-12)
np.testing.assert_allclose(unitary, np.array([[1, 1], [1, -1]]) / np.sqrt(2), atol=1e-12)
assert set(counts) <= {"0", "1"} and sum(counts.values()) == 128
```

The top-level `simulate` and `build_functionality` functions return dense NumPy
arrays; they do not return DD handles. Dense statevectors and matrices require
space exponential in the number of qubits. Statevector simulation starts a
closed program in the all-zero state and rejects measurement-dependent
computation and resets. Functionality construction requires a supported unitary
program; measurements and resets are not supported.

For DD results, compile to `QCOProgram` and use `build_functionality(package)`
or `simulate(initial_state, package, seed=...)`. The latter consumes one live
reference to the input DD, even when execution fails after input validation.
Keep the package alive and release returned vector or matrix references with
`dec_ref_vec` or `dec_ref_mat`. Raw vector/matrix constructors and DD operations
remain in `mqt.core.dd`.

`sample` returns declared classical outputs in count-string order: the last
returned register first, with each register most-significant-bit first. If the
program has no classical-bit result, it samples all final qubits. A seed of zero
selects nondeterministic seeding for these QCO helpers; nonzero seeds make the
same execution path reproducible. Do not expect the classic simulator's random
sequence to remain unchanged.

### C++ libraries and build requirements

Remove dependencies on `MQT::CoreIR` and `MQT::CoreQASM`, all `ir/` and `qasm3/`
public headers, and the `qc::QuantumComputation` hierarchy. The circuit-facing
`dd/Simulation.hpp` and `dd/FunctionalityConstruction.hpp` headers are also
removed. Use low-level matrix/vector APIs from `MQT::CoreDD` or the compiler's
QC/QCO interfaces for circuit work.

The compiler C++ interfaces are source-tree APIs. `mqt/Compiler/Programs.h` owns
program types; `mqt/Compiler/Pipeline.h` owns compilation;
`mqt/Compiler/QDMIAdapter.h` provides device compilation and submission. Link
the corresponding `MQTCompilerPrograms`, `MQTCompilerPipeline`, or
`MQTCompilerQDMIAdapter` target in an embedded source build. These are not
installed public-header replacements for `MQT::CoreIR`. See the
[C++ compiler example](https://mqt.readthedocs.io/projects/core/en/latest/mlir/target_compilation.html#c-source-tree-api).

Rebuild downstream native libraries against v4. The shared-library ABI version
changes from `3.10` to `4.0`; update any explicit library filenames or wheel
repair exclusions that contain the old suffix.

Source builds now require **CMake 3.28+** and build the compiler by default,
which requires **LLVM/MLIR 23.1+**. The Python build backend selects CMake
4.4.1+ itself. Wheels already contain the compiler and local simulator; wheel
users do not need to install LLVM separately.

For an LLVM-free C++ build, set `BUILD_MQT_CORE_MLIR=OFF`. This retains the DD
and QDMI libraries and the superconducting device model, but omits the compiler,
compiler-backed benchmark generation, `mqt-cc`, and the DDSIM device. DDSIM
requires MLIR for OpenQASM as well as QIR execution. Do not rely on an old DDSIM
build option to enable it without MLIR.

Prebuilt LLVM/MLIR distributions are available from
[setup-mlir](https://github.com/munich-quantum-software/setup-mlir).
Set `MLIR_DIR` to their `lib/cmake/mlir` directory. A repository-local `.env`
can supply `MLIR_DIR` when it is not set in CMake. The supplied macOS LLVM/MLIR
builds require Clang rather than GCC; AppleClang 17+ is required for MQT's MLIR
code.

#### CMake presets on Windows

All CMake presets now use Ninja. On Windows, remove `-windows` from preset names
when configuring, building, and testing:

| Previous preset           | Replacement       |
| ------------------------- | ----------------- |
| `debug-windows`           | `debug`           |
| `release-windows`         | `release`         |
| `debug-windows-no-mlir`   | `debug-no-mlir`   |
| `release-windows-no-mlir` | `release-no-mlir` |

Install Ninja and run CMake from a Visual Studio developer shell for the target
architecture. Use a new build directory if an existing directory uses the Visual
Studio generator.

### QDMI submission and QIR execution

Existing QDMI device discovery and job APIs remain in `mqt.core.qdmi`. DDSIM now
imports OpenQASM through QC and executes QCO. Inputs must satisfy the compiler's
supported OpenQASM subset. In particular, initialize classical output bits
before reading or returning them. The Qiskit backend preserves Qiskit's
zero-initialized classical-bit semantics during serialization.

Device-directed compilation is separate from submission. Reuse a compiled
payload while its target contract still matches the destination:

```python
from mqt.core.mlir import compile_program, submit_program
from mqt.core.qdmi.driver import open_device

device = open_device("mqt.ddsim.default")
source = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; x q; bit c = measure q;'
compiled = compile_program(source, target=device)
job = submit_program(compiled, target=device, num_shots=32, custom1=17)
assert job.wait()
assert job.get_counts() == {"1": 32}
```

The standalone QIR runner is removed. Its runtime and JIT are internal DDSIM
implementation details. Submit QIR through QDMI instead. Use `str` with
`QIR_BASE_STRING`/`QIR_ADAPTIVE_STRING` and `bytes` with
`QIR_BASE_MODULE`/`QIR_ADAPTIVE_MODULE` from `ProgramFormat`.

Regenerate QIR for the current QIR 2.1 runtime and QIS signatures; legacy
allocator and output overloads are rejected. Each job now owns its runtime,
random-number generator, and output sink. Jobs no longer write QIR records to
process stdout. To retrieve records, submit a positive-shot QIR job with
`custom2=True`, then call `job.get_custom_result(CustomProperty.CUSTOM1, str)`.
Import `CustomProperty` from `mqt.core.qdmi`. Job parameter `custom1` remains
the positive integer seed; it is separate from result `CUSTOM1`.

Statevector extraction supports **both Base and Adaptive QIR**. Zero shots
request state extraction; eligible terminal-sampling jobs also retain the
uncollapsed state. Base extraction requires the first measurement boundary to be
marked `irreversible` and a terminal irreversible region. Adaptive extraction
can execute classical control flow, dynamic allocations, and direct helpers
while deferring terminal Z measurements. It rejects measurement-dependent
computation, resets, and subsequent operations on measured wires. Capturing
output forces per-shot execution and does not retain an uncollapsed state;
OpenQASM and zero-shot jobs reject capture.

For the exact resource, result-use, and lifetime restrictions, see the
[QIR execution contracts](https://mqt.readthedocs.io/projects/core/en/latest/qir/index.html)
and
[DDSIM guide](https://mqt.readthedocs.io/projects/core/en/latest/qdmi/ddsim_device.html).

### Other API changes

`dd::toDot` and `dd::export2Dot` keep their signatures, but their node IDs now
follow traversal order. Direct users of `modernNode`, `classicNode`, and
`memoryNode` must pass a node ID between the edge and output-stream arguments.
Direct users of `bwEdge`, `coloredEdge`, and `memoryEdge` must replace the
source/destination edge pair with the destination edge and explicit
source/destination IDs. Prefer `toDot` to assign IDs for a complete graph.

The new typed benchmark API is available through `MQT::CoreBench` and
`mqt.core.bench`; generation also requires the compiler. `mqt-core-bench` is its
command-line interface. These are not drop-in replacements for the circuit
factories removed in v3.10.0. See the
[benchmark guide](https://mqt.readthedocs.io/projects/core/en/latest/benchmarks.html)
for family options, instance specifications, and result evaluation.

## [3.10.0]

### Shared-library ABI version

The shared-library ABI version (`SOVERSION`) changes from `3.9` to `3.10`.
Rebuild downstream C++ libraries against MQT Core 3.10.0. In `cibuildwheel`
configurations that exclude bundled MQT Core libraries from wheel repair,
replace each `libmqt-core-*.so.3.9` entry with the corresponding
`libmqt-core-*.so.3.10` entry.

### Removal of the `spdlog` dependency

MQT Core no longer discovers, downloads, builds, installs, or exports `spdlog`.
The installed CMake package no longer calls `find_dependency(spdlog)`, and the
Python wheels no longer contain the `spdlog` headers or shared library. QDMI
diagnostics continue to use standard error.

Downstream projects that use `spdlog` must declare and package the dependency
themselves. Stop passing `MQT_CORE_SPDLOG_INSTALL` or `SPDLOG_*` cache variables
when configuring MQT Core. Configure the downstream project's own `spdlog`
dependency instead.

### CircuitOptimizer removal

MQT Core no longer provides `qc::CircuitOptimizer`. Replace the two generic
transformations with `QuantumComputation` member calls:

- Replace `qc::CircuitOptimizer::flattenOperations(qc, customGatesOnly)` with
  `qc.flattenOperations(customGatesOnly)`.
- Replace `qc::CircuitOptimizer::removeFinalMeasurements(qc)` with
  `qc.removeFinalMeasurements()`.

Include `ir/QuantumComputation.hpp` and link `MQT::CoreIR`. MQT QCEC and MQT
QMAP each own their single-qubit gate-fusion implementation. MQT Core provides
no replacement for `singleQubitGateFusion` outside those packages.

MQT QCEC now owns the equivalence-checking transformations `swapReconstruction`,
`removeDiagonalGatesBeforeMeasure`, `eliminateResets`, `deferMeasurements`,
`backpropagateOutputPermutation`, and `elidePermutations`. Use MQT QCEC's
equivalence-checking flow for this behavior, or keep a package-specific
transformation with the consumer that needs it.

MQT QMAP now owns the mapping transformations `decomposeSWAP`, `cancelCNOTs`,
and `replaceMCXWithMCZ`. Replace calls to the corresponding
`qc::CircuitOptimizer` methods with `qmap::decomposeSWAP`, `qmap::cancelCNOTs`,
and `qmap::replaceMCXWithMCZ`, respectively. Include
`datastructures/CircuitOptimizations.hpp` and link `MQT::QMapDS`.

The public `constructDAG` function and the `DAG`, `DAGIterator`,
`DAGReverseIterator`, `DAGIterators`, and `DAGReverseIterators` aliases have no
Core replacement. Build the small traversal structure in the package that
consumes it. MQT QMAP and MQT QuSAT demonstrate this migration.

The public `removeIdentities`, `removeOperation`, `collectBlocks`, and
`collectCliffordBlocks` functions have no replacement. Erase operations through
`QuantumComputation` where needed.

The `MQT::CoreCircuitOptimizer` CMake target and the
`circuit_optimizer/CircuitOptimizer.hpp` header are removed. The
`circuit_optimizer/mqt_core_circuit_optimizer_export.h` header is no longer
generated or installed.

### Pruned DD construction helpers

MQT Core no longer provides `dd::GenerationWireStrategy`,
`dd::generateExponentialState`, or `dd::generateRandomState`. These APIs
generated decision diagrams with selected shapes for tests and have no direct
replacement.

MQT Core also removed `dd::buildFunctionalityRecursive`. The Python
`mqt.core.dd.build_unitary` and `mqt.core.dd.build_functionality` functions no
longer accept the `recursive` argument and always use sequential construction.
Use MQT DDSIM's unitary simulator when recursive pairwise construction is
required.

MQT Core also removed `dd/GateMatrixDefinitions.hpp`,
`dd::opToSingleQubitGateMatrix`, `dd::opToTwoQubitGateMatrix`,
`dd::opToThreeQubitGateMatrix`, `dd::getStandardOperationDD`,
`dd::MEAS_ZERO_MAT`, and `dd::MEAS_ONE_MAT`. Low-level consumers must pass raw
matrices to `dd::Package::makeGateDD`, `makeTwoQubitGateDD`,
`makeThreeQubitGateDD`, or `makeDDFromMatrix`. Circuit-facing DD functions
continue to translate CoreIR operations internally.

The zero, basis, GHZ, W, dense-vector, dense-matrix, raw gate-matrix, and
sequential circuit constructors remain available.

### macOS support

MQT Core no longer supports x86 macOS. Use Apple silicon with macOS 13.3 or
newer. The new deployment target enables `std::format` in libc++.

### Removal of CoreAlgorithms

MQT Core no longer installs `MQT::CoreAlgorithms` or the headers below
`algorithms/`. MQT Core provides no direct replacement for the removed circuit
factories. Move required implementations to the package that uses them. The
`BUILD_MQT_CORE_BENCHMARKS` option and its legacy DD evaluation target were also
removed.

### Removal of the FoMaC compatibility APIs

MQT Core no longer provides `mqt.core.fomac` or `mqt.core.qdmi.driver.Session`.
Import QDMI entities such as `Device`, `Job`, and `ProgramFormat` from
`mqt.core.qdmi`. Import `DeviceDefinition` and the module-level registry
functions from `mqt.core.qdmi.driver`. Use `registered_device_ids()` to discover
devices and `open_device()` to open a fresh device session. Pass provider
configuration overrides to `open_device()` when a device needs per-open
configuration.

MQT Core also removes the FoMaC name from its C++ API. Apply these replacements:

- `fomac::` becomes `qdmi::`.
- `fomac/FoMaC.hpp` becomes `qdmi/Client.hpp`.
- `fomac/Slurm.hpp` becomes `qdmi/Slurm.hpp`.
- `MQT::CoreFoMaC` becomes `MQT::CoreQDMI`.

The class and function names do not change. For example:

```cpp
#include "qdmi/Client.hpp"

auto device = qdmi::Session::openDevice("mqt.ddsim.default");
```

### Qiskit 2.1 minimum

The minimum Qiskit version increases from **1.1.0 to 2.1.0**, dropping support
for all Qiskit 1.x releases and Qiskit 2.0. Upgrade Qiskit to 2.1.0 or newer.

### Native QDMI Qiskit primitives

`QDMISampler` and `QDMIEstimator` are removed. Use Qiskit's `BackendSamplerV2`
and `BackendEstimatorV2`, or the backend factories with native options:

```python
sampler = backend.sampler(default_shots=2048)
estimator = backend.estimator(default_precision=0.01)
```

Estimator uses positive precision, not `default_shots`: its default is `1/64`
(4096 shots). Grouping, metadata, broadcasting, and standard errors now follow
Qiskit. Sampler requires genuine QDMI `SHOTS`, which DDSIM supports. Counts-only
devices remain usable for Estimator but cannot run Sampler. No shot
reconstruction is available.

`backend.run(memory=True)` preserves shot order. Results include classical
register boundaries; failed jobs and invalid results raise on collection, and
unsupported execution options are rejected. See the
[backend requirements](https://mqt.readthedocs.io/projects/core/en/stable/qdmi/qdmi_backend.html#backend-requirements).

### Private `nlohmann_json` dependency

MQT Core uses `nlohmann_json` only inside its implementation. It no longer
installs the library, exports it, or looks for it in its package configuration.
Depend on `nlohmann_json` directly if your project uses it.

No installed header includes a `nlohmann` header any more. The decision-diagram
statistics report through strings and streams instead. MQT Core removed the
following names:

- `dd::Statistics::json`, `dd::MemoryManagerStatistics::json`,
  `dd::TableStatistics::json`, and `dd::UniqueTableStatistics::json`. Use
  `toString`, the stream operator, or the individual counters.
- `dd::UniqueTable::getStatsJson`. Use `dd::getStatisticsString`.
- `dd::getStatistics` and `dd::getDataStructureStatistics`. Use
  `dd::getStatisticsString` and `dd::getDataStructureStatisticsString`, which
  return the same report as a JSON-formatted string.
- The `MQT_CORE_JSON_INSTALL` CMake option.

`dd::getStatisticsString` takes the `includeIndividualTables` flag that
`dd::getStatistics` used to take.

### Removal of the neutral-atom stack

MQT Core no longer contains neutral-atom functionality. The complete stack moved
to [MQT QMAP](https://github.com/munich-quantum-toolkit/qmap), which is now its
sole owner. Depend on MQT QMAP to keep using this functionality.

MQT Core removed the following names:

- The `MQT::CoreNA`, `MQT::CoreNAFoMaC`, `MQT::CoreQDMINaDevice`, and
  `MQT::CoreQDMINaDeviceConfig` CMake targets.
- The `BUILD_MQT_CORE_QDMI_NA_DEVICE` CMake option.
- The `na/NAComputation.hpp`, `na/entities/*.hpp`, `na/operations/*.hpp`,
  `na/fomac/Device.hpp`, `qdmi/devices/na/Configuration.hpp`, and
  `ir/operations/AodOperation.hpp` headers.
- The `na` C++ namespace.
- The `mqt.core.na` Python module and its `mqt.core.na.qdmi` and
  `mqt.core.na.fomac` submodules.
- The `Move`, `Bridge`, `AodActivate`, `AodDeactivate`, and `AodMove`
  `qc::OpType` values, together with `QuantumComputation::move`,
  `QuantumComputation::cmove`, `QuantumComputation::mcmove`,
  `QuantumComputation::bridge`, and the OpenQASM names `move`, `bridge`,
  `aod_activate`, `aod_deactivate`, and `aod_move`.
- The bundled `mqt.na.default` QDMI device.

### Removal of the ZX-calculus library

MQT Core no longer provides the `mqt-core-zx` library, the `MQT::CoreZX` CMake
target, the `mqt-core/zx` headers, or the global `zx` namespace. Remove these
from downstream includes and link dependencies. Equivalence-checking users
should use [MQT QCEC]; QCEC's ZX implementation is internal and is not a
replacement public API.

The build-tree `MQT::Multiprecision` alias, the installed `MQT::multiprecision`
target, the `USE_SYSTEM_BOOST`, `MQT_CORE_WITH_GMP`, and
`MQT_CORE_ZX_SYSTEM_BOOST` CMake options, and the `BOOST_MIN_VERSION` cache
variable have also been removed. MQT Core no longer discovers, fetches, or
exports configuration for Boost.Multiprecision or GMP.

### Removal of density matrix support from the DD package

MQT Core no longer provides density matrix decision diagrams or the related
deterministic and stochastic noise functionality. [MQT DDSIM] 2.5.0 and newer
provide this functionality in the `dd::ddsim` namespace. Downstream code that
used the MQT Core APIs must migrate to MQT DDSIM or provide the functionality
directly. The `ATrue`, `AFalse`, `MultiATrue`, and `MultiAFalse` operation types
have also been removed.

### Removal of DD approximation support

MQT Core no longer provides the decision-diagram approximation algorithm. The
algorithm had no production owner in the MQT ecosystem. Remove uses of the
`dd/Approximation.hpp` header, the `dd::ApproximationMetadata` type, and the
`dd::approximate` function. MQT Core does not provide a replacement.

### CoreIR API cleanup

The CoreIR API cleanup requires the following migrations:

- Replace `getNmeasuredQubits()` and `num_measured_qubits` with
  `getNoutputQubits()` and `num_output_qubits`, respectively.
- Replace permutation-aware `Operation::equals()` and `getUsedQubitsPermuted()`
  calls by applying the permutation to cloned operations before comparing them.
- Replace `getHighestLogicalQubitIndex()`, `printStatistics()`, and
  `printPermutation()` with `initialLayout.maxValue()`, the individual count
  accessors, and direct `Permutation` iteration, respectively.
- Construct output-permutation measurements explicitly instead of calling
  `appendMeasurementsAccordingToOutputPermutation()`.
- Replace direct `Operation::dumpOpenQASM2()`, `dumpOpenQASM3()`, or
  `dumpOpenQASM()` calls with `qasm3::Serializer`. The register-map aliases
  moved from `ir/Register.hpp` to `qasm3/Serializer.hpp`:

  ```cpp
  #include "qasm3/Serializer.hpp"

  qasm3::Serializer(stream, qc::Format::OpenQASM2)
      .serialize(operation, qubitMap, bitMap);
  ```

  Use `qc::Format::OpenQASM3` for OpenQASM 3 output. The relocated maps own
  their register metadata instead of retaining references to the registers used
  to construct them. Packages that define custom `Operation` subclasses must own
  serialization for their extended syntax; in particular, MQT QMAP owns
  neutral-atom OpenQASM serialization.

The register lookup helpers `getQubitRegister()`, `getPhysicalQubitIndex()`, and
`physicalQubitIsAncillary()` are now private implementation details.

### QuantumComputation random-number generator

`QuantumComputation` no longer stores a random-number generator or seed. Remove
the third `seed` argument from C++ and Python constructor calls. C++ callers
that used `QuantumComputation::getGenerator()` must create and own a
random-number generator instead. Randomized circuit generators continue to
accept a seed and now own a separate generator for each call.

### Removal of the `datastructures` (sub)library

MQT Core no longer provides the `datastructures` (`ds`) sublibrary. [MQT QMAP]
3.8.0 and newer provide the moved code under `datastructures/` through the
`MQT::QMapDS` CMake target. Downstream users must depend on MQT QMAP or provide
the required data structures directly.

### Python 3.11 and Stable ABI wheels

MQT Core now requires Python 3.11 or newer. Upgrade the Python environment
before installing this release.

MQT Core now publishes one `cp311-abi3` wheel for GIL-enabled CPython 3.11 and
newer. Free-threaded support starts with CPython 3.15 in a separate
`cp315-abi3t` wheel. MQT Core no longer publishes free-threaded CPython 3.13 or
3.14 wheels.

This release updates `nanobind` to 3.0.1, which changes the `nanobind` ABI.
Rebuild downstream native Python extensions that use MQT Core's `nanobind`-bound
C++ types. Pure Python consumers do not need to recompile anything.

The Python bindings depend on `nanobind-backend`, which supplies the
interpreter-specific `nanobind` runtime. This dependency does not change the C++
API or the Python import paths.

## [3.9.2]

### Optional QDMI shot counts

QDMI jobs whose repetition count is encoded in the program can now omit
`num_shots`. Existing C++ calls that pass a `size_t` keep the same ABI and
behavior; new C++ overloads omit the argument, while Python accepts `None` and
uses it by default.

## [3.9.1]

### Program serializers for the Qiskit backend

The Qiskit backend no longer decides in its own code how to turn a circuit into
a program. It takes every program format from a registered _program serializer_,
and MQT Core registers its own OpenQASM 2 and OpenQASM 3 serializers the same
way as everyone else.

A serializer takes the circuit and the backend. It returns `str` for a text
format and `bytes` for a binary format;
{py:func}`~mqt.core.qdmi.is_binary_program_format` states which kind a format
carries. Register one at run time:

```python
import io

from qiskit import qpy

from mqt.core.plugins.qiskit import register_program_serializer
from mqt.core.qdmi import ProgramFormat


def my_qpy_serializer(circuit, backend) -> bytes:
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return buffer.getvalue()


register_program_serializer(ProgramFormat.QPY, my_qpy_serializer)
```

A package that owns a device advertises its serializer through the
`mqt.core.qiskit.program_serializers` entry point group instead, so MQT Core
finds it without importing the package:

```toml
[project.entry-points."mqt.core.qiskit.program_serializers"]
IQM_JSON = "iqm.qdmi.serializers:qiskit_to_iqm_json"
```

`mqt.core.plugins.qiskit.serializers.PROGRAM_FORMAT_PREFERENCE` states which
format the backend picks when a device accepts several. Pass `replace=True` to
`register_program_serializer` to take over a format that already has a
serializer, including OpenQASM 2 and OpenQASM 3.

A backend subclass that must represent a device-native operation outside
Qiskit's standard gate library sets `_EXTRA_GATES`:

```python
class MyBackend(QDMIBackend):
    _EXTRA_GATES = {"move": MoveGate()}
```

### IQM JSON serialization moved to QDMI-on-IQM

MQT Core no longer provides `qiskit_to_iqm_json` or `MoveGate`.
[QDMI-on-IQM](https://github.com/iqm-finland/QDMI-on-IQM) owns both. Import them
from `iqm.qdmi` instead:

```python
from iqm.qdmi.serializers import qiskit_to_iqm_json
from iqm.qdmi.gates import MoveGate
```

Installing `iqm-qdmi` is enough to keep submitting IQM JSON. The package
advertises its serializer through the entry point group described above, so a
backend over an IQM device needs no code change.

## [3.9.0]

### Shared-library ABI version

The shared-library ABI version (`SOVERSION`) changes from `3.8` to `3.9`.
Rebuild downstream C++ libraries against MQT Core 3.9.0. In `cibuildwheel`
configurations that exclude bundled MQT Core libraries from wheel repair,
replace each `libmqt-core-*.so.3.8` entry with the corresponding
`libmqt-core-*.so.3.9` entry.

### `nanobind` updated to version 2.15.0

`nanobind` 2.15.0 changes the `nanobind` ABI. Rebuild downstream native Python
extensions that use MQT Core's `nanobind`-bound C++ types. Pure Python consumers
do not need to recompile anything.

### QDMI updated to version 1.3.3

The minimum supported QDMI version changes from 1.3.2 to 1.3.3. CMake builds
that use a system installation of QDMI must provide version 1.3.3 or newer.
Builds that let MQT Core fetch QDMI need no change.

### QDMI calibration runs and batch jobs

`Device::submitJob` used to reject `CALIBRATION` and `BATCH_JOB` together, which
left MQT Core reporting that a device needs calibration through
`needs_calibration()` without any way to trigger one. The two formats are
different cases and are now treated as such.

A calibration run has its own entry point. QDMI does not require a program for
one, so the payload is optional; when it is present, the device defines what it
means:

```python
device.submit_calibration_job()
device.submit_calibration_job("configuration")
```

In C++, use `Device::submitCalibrationJob`. A calibration run executes no
circuit, so neither form takes a shot count.

Batch jobs are explicitly unsupported. A batch job's program is a list of job
handles rather than a byte payload, which `submitJob` cannot express. Passing
`ProgramFormat.BATCH_JOB` to `submit_job` raises `ValueError` in Python and
`std::invalid_argument` in C++.

### Removal of QDMI configuration through `pyproject.toml`

MQT Core no longer reads QDMI device definitions from a `[tool.qdmi]` table in
`pyproject.toml`. Project discovery now looks only for `qdmi.json`. Move an
existing table into a `qdmi.json` file beside the `pyproject.toml`. For example,
replace this `pyproject.toml` table:

```toml
[tool.qdmi]
devices = [
  { id = "example.device", library = "libexample-device.so", prefix = "EXAMPLE" },
]
```

with this `qdmi.json`:

```json
{
  "schema-version": 1,
  "qdmi": {
    "devices": [
      {
        "id": "example.device",
        "library": "libexample-device.so",
        "prefix": "EXAMPLE"
      }
    ]
  }
}
```

The JSON document adds the `"schema-version": 1` key and nests the device array
under `qdmi`. Every other key keeps its name and meaning. Relative paths still
resolve against the file that declares them. `MQT_CORE_QDMI_CONFIG_FILE`,
`MQT_CORE_QDMI_CONFIG_JSON`, the system and user files, and the packaged
`*.qdmi.json` fragments do not change.

### QDMI Qiskit primitive options

`QDMISampler` and `QDMIEstimator` no longer accept the MQT-specific `options`
mapping. Pass shot and precision defaults directly, preferably through the
backend factories:

```python
sampler = backend.sampler(default_shots=2048)
estimator = backend.estimator(default_precision=0.01, default_shots=2048)
```

Replace `QDMIEstimator(..., options={"default_shots": shots})` with
`QDMIEstimator(..., default_shots=shots)`. The sampler ignored its former
`options` mapping, so remove that argument without replacement.

### Runtime-configurable SC QDMI device

The built-in superconducting QDMI provider now parses its device description
when each session is initialized. The `mqt-core-qdmi-sc-device-gen` target,
SC-specific generator executable, `sc::writeHeader`, `sc::writeJSONSchema`, and
generated `DeviceMemberInitializers.hpp` file have been removed. Replace
generator API use with `sc::Device` and the `sc::readJSON` functions declared in
`qdmi/devices/sc/Configuration.hpp`.

### Runtime-configurable neutral-atom QDMI device

The built-in neutral-atom QDMI provider now parses its device description when
each session is initialized. The `mqt-core-qdmi-na-device-gen` target,
`mqt-core-qdmi-na-device-generator` executable, `na::writeHeader`, and generated
`DeviceMemberInitializers.hpp` file have been removed. Replace generator API use
with the `na::Device` configuration type and the `na::readJSON` functions in
`qdmi/devices/na/Configuration.hpp`.

At runtime, use the registry `session.device-config` field or Python
`device_config` and `device_config_file` arguments. Direct low-level QDMI
clients pass inline JSON through CUSTOM1 or a file path through CUSTOM2.

### QDMI Python namespace

The native Python module has moved from `mqt.core.fomac` to `mqt.core.qdmi`.
QDMI entities such as `Device`, `Job`, and `ProgramFormat` are in
`mqt.core.qdmi`. Import functions and classes from `mqt.core.qdmi.driver` for
device discovery, registration, and opening:

```python
from mqt.core.qdmi.driver import open_device

device = open_device("mqt.ddsim.default")
```

`mqt.core.fomac` remains available in MQT Core v3 and re-exports the same
objects. Importing that module emits a `DeprecationWarning`. It will be removed
in MQT Core 4.0. The legacy `driver.Session` class also emits a
`DeprecationWarning` when constructed and will be removed in 4.0. Replace
session-based discovery with `registered_device_ids()` and `open_device()` from
`mqt.core.qdmi.driver`.

The neutral-atom specialization has moved from `mqt.core.na.fomac` to
`mqt.core.na.qdmi`. The former submodule remains a v3 compatibility alias and
will be removed in MQT Core 4.0.

The C++ FoMaC namespace, headers, library, and `MQT::CoreFoMaC` target do not
change.

### Python binding CMake helper

The `add_mqt_python_binding_nanobind` function is now called
`add_mqt_python_binding`. Rename the calls in downstream `CMakeLists.txt` files:

```cmake
add_mqt_python_binding(
  MYPACKAGE
  py_mypackage
  ${SOURCES}
  MODULE_NAME
  _core
  INSTALL_DIR
  .
  LINK_LIBS
  MQT::Core)
```

The old `add_mqt_python_binding` function built modules with `pybind11` and has
been removed. MQT Core now uses that name for its `nanobind` helper. The
arguments to the renamed helper do not change.

## [3.8.0]

The shared library ABI version (`SOVERSION`) is increased from `3.7` to `3.8`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### QDMI updated to version 1.3.2

MQT Core already bundled QDMI 1.3.2 in the previous release, but now also
requires at least that version when using a system-provided QDMI installation.

### Bundled QDMI devices in embedded builds

The bundled QDMI devices now have individual CMake options:
`BUILD_MQT_CORE_QDMI_DDSIM_DEVICE`, `BUILD_MQT_CORE_QDMI_NA_DEVICE`, and
`BUILD_MQT_CORE_QDMI_SC_DEVICE`. All three remain enabled by default in a
standalone MQT Core build. They default to disabled when MQT Core is consumed
through CMake's `FetchContent` or `add_subdirectory`; embedded consumers can
enable only the devices they need before making MQT Core available. The QDMI
driver and FoMaC libraries remain available independently.

### QDMI runtime device registration

The unstable runtime-loading helpers have been replaced with registration by a
stable device ID followed by an explicit open. In Python, replace
`add_dynamic_device_library(library_path, prefix, ...)` with:

```python
from mqt.core.fomac import DeviceDefinition, open_device, register_device

definition = DeviceDefinition("my.device", library_path, prefix, base_url="https://device.example")
register_device(definition)
device = open_device("my.device")
```

Per-backend session values can be passed directly to
`open_device("my.device", base_url=..., token=...)`. Every call creates a fresh
device session without registering another device ID. Repeated integration setup
can use `register_device_if_absent(definition)` instead of suppressing
duplicate-ID errors; invalid definitions are still rejected, and a device
disabled by higher-precedence configuration remains reserved.

The equivalent C++ flow is:

```cpp
qdmi::DeviceDefinition definition{.id = "my.device",
                                  .library = libraryPath,
                                  .prefix = prefix};
auto& driver = qdmi::Driver::get();
driver.registerDevice(definition);
auto device = fomac::Session::openDevice("my.device");
```

Registration validates and stores metadata without loading native code. Opening
an unknown or disabled ID fails. `fomac::Session::openDevice` creates a fresh
owned session on every call. `qdmi::Driver::open(id)` retains its cached-device
behavior for client callers.

See the {doc}`QDMI device configuration guide <qdmi/configuration>` for the
versioned JSON and TOML formats, configuration precedence, and relocatable
device manifests.

### FoMaC program payload handling

FoMaC now distinguishes textual programs from exact binary payloads. In C++, use
`Device::submitJob(const std::string&, ...)` for text formats and
`Device::submitJob(std::span<const std::byte>, ...)` for binary formats. In
Python, pass `str` for text and `bytes` for binary payloads. In particular, QIR
`*_STRING` formats are text, while QIR `*_MODULE` formats are LLVM bitcode and
must be submitted as bytes.

`Job::getProgram()` and Python's `Job.program` remain the textual accessors and
now reject binary or non-null-terminated payloads. Use `Job::getProgramBytes()`
or `Job.program_bytes` to retrieve the exact submitted bytes. Calibration and
batch-job formats cannot be submitted through these generic program APIs because
their QDMI payloads have specialized representations.

### QDMI child devices

The QDMI driver now translates device-library-specific `QDMI_Child_Device`
handles into client-facing `QDMI_Device` handles backed by dedicated child
sessions. Direct child devices can be queried through
`fomac::Device::getChildDevices()` in C++ and `Device.child_devices()` in
Python. Devices without child-device support continue to behave unchanged.

## [3.7.0]

The shared library ABI version (`SOVERSION`) is increased from `3.6` to `3.7`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### `nanobind` updated to version 2.13.0

This release updates the `nanobind` dependency to version 2.13.0, which includes
an ABI bump. Any existing code that uses the `mqt-core` Python bindings will
need to be recompiled with the new `nanobind` version.

### QDMI updated to version 1.3.2

While not a breaking change, this release updates the QDMI dependency to version
1.3.2

### CMake presets

[CMake presets] have been added to provide a standardized and reproducible way
to configure builds across different platforms. These presets are also used in
our CI. They assume that `MLIR_DIR` is defined in your environment and pointing
to an MLIR installation.

On Unix systems, the `debug`, `release`, and `coverage` presets can be used to
configure, build, and test MQT Core.

```console
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Additionally, the `lint` preset can be used to configure and build MQT Core in
preparation for a `clang-tidy` run.

If you are on Windows, use the `debug-windows` and `release-windows` presets.

## [3.6.0]

The shared library ABI version (`SOVERSION`) is increased from `3.5` to `3.6`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### Changes to builtin QDMI devices

The builtin QDMI devices (with prefixes `MQT_SC`, `MQT_NA`, and `MQT_DDSIM`) are
now all built as shared libraries by default. In turn, the shared library
wrappers (with prefixes `MQT_SC_DYN` and `MQT_NA_DYN`) have been removed
entirely. MQT Core's QDMI driver will automatically load the shared libraries of
the builtin devices if they are available in the library search path. If you
were previously using the statically builtin devices, no changes should be
necessary as the shared libraries are now the default. If you were previously
using the shared library wrappers, you should switch to using the builtin
devices instead, which are now shared libraries by default.

### Broader operation support in QDMI Qiskit converter

The QDMI Qiskit converter now supports a broader range of operations, including
multi-controlled gates such as `mcx`, `mcz`, `mcrx`, and more. As a consequence,
these operations can now be directly used without requiring decomposition, for
example, with the builtin `DDSIM` QDMI device.

### Minimum supported Qiskit version

From this release onwards, MQT Core requires Qiskit version 1.1.0 or higher.
This is due to the fact that we are relying on some fixes to Qiskit primitives
that were introduced in that version. If you are using MQT Core with Qiskit,
please ensure that you have updated to Qiskit 1.1.0 or higher to avoid any
compatibility issues.

## [3.5.1]

No breaking changes.

### Component-based CMake installs

Fixed exported `nlohmann_json` CMake metadata so `find_package(mqt-core CONFIG)`
no longer propagates an invalid `.../COMPONENT` include directory in
component-based installations. Anyone relying on an installed version of
`mqt-core` should update from 3.5.0 to 3.5.1.

## [3.5.0]

The shared library ABI version (`SOVERSION`) is increased from `3.4` to `3.5`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### `nanobind` updated to version 2.12.0

This release updates the `nanobind` dependency to version 2.12.0, which includes
an ABI bump. Any existing code that uses the `mqt-core` Python bindings will
need to be recompiled with the new `nanobind` version.

## [3.4.0]

The shared library ABI version (`SOVERSION`) is increased from `3.3` to `3.4`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### Python wheels

This release contains two changes to the distributed wheels.

First, we have removed all wheels for Python 3.13t. Free-threading Python was
introduced as an experimental feature in Python 3.13. It became stable in Python
3.14.

Second, for Python 3.12+, we are now providing Stable ABI wheels instead of
separate version-specific wheels. This was enabled by migrating our Python
bindings from `pybind11` to `nanobind`.

Both of these changes were made in the interest of conserving PyPI space and
reducing CI/CD build times. The full list of wheels now reads:

- 3.10
- 3.11
- 3.12+ Stable ABI
- 3.14t

### QDMI-Qiskit integration

This release introduces a Qiskit `BackendV2`-compatible interface to QDMI
devices. The `mqt.core.plugins.qiskit` module has been extended with
`QDMIProvider`, `QDMIBackend`, and `QDMIJob` classes that allow running Qiskit
circuits on QDMI-compliant devices.

Users can now execute Qiskit circuits directly on QDMI devices:

```python
from mqt.core.plugins.qiskit import QDMIProvider

provider = QDMIProvider()
backend = provider.get_backend("MQT Core DDSIM QDMI Device")
job = backend.run(circuit, shots=1024)
result = job.result()
```

The backend automatically converts circuits to QASM, introspects device
capabilities, validates circuits, and formats results. The existing FoMaC
interface (`mqt.core.fomac`) remains fully supported for direct, low-level
access to QDMI devices.

Install with Qiskit support: `uv pip install "mqt-core[qiskit]"`

See the
[Qiskit Backend documentation](https://mqt.readthedocs.io/projects/core/en/stable/qdmi/qdmi_backend.html)
for details.

### Argument name changes in `QuantumComputation` and `CompoundOperation` dunder methods

Since we enabled `ty` for type checking, it revealed that some of the dunder
methods of `QuantumComputation` and `CompoundOperation` had incorrect argument
names, which would prevent these classes from properly implementing the
`MutableSequence` protocol. This release fixes these issues by renaming the
arguments of the following methods:

- `QuantumComputation.__getitem__`
- `QuantumComputation.__setitem__`
- `QuantumComputation.__delitem__`
- `QuantumComputation.insert`
- `QuantumComputation.append`
- `CompoundOperation.__getitem__`
- `CompoundOperation.__setitem__`
- `CompoundOperation.__delitem__`
- `CompoundOperation.insert`
- `CompoundOperation.append`

All index arguments are now named `index` instead of `idx` (or `i` or `slice`)
and all values are now named `value` instead of `val` (or `op` or `ops`).

### DD Package evaluation

This release moves the DD Package evaluation functionality from within the
`mqt.core` package to a dedicated script in the `eval` directory. In the
process, the `mqt-core-dd-compare` entry point as well as the `evaluation` extra
have been removed. The `eval/dd_evaluation.py` script acts as a drop-in
replacement for the previous CLI entry point. Since the `eval` directory is not
part of the Python package, this functionality is only available via source
installations or by cloning the repository.

## [3.3.0]

The shared library ABI version (`SOVERSION`) is increased from `3.2` to `3.3`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### IfElseOperation

This release introduces an `IfElseOperation` to the C++ library and the Python
package to support Qiskit's `IfElseOp`. The new operation replaces the
`ClassicControlledOperation`.

An `IfElseOperation` can be added to a `QuantumComputation` using `if_else()`.

```python
qc.if_else(
    then_operation=StandardOperation(target=0, op_type=OpType.x),
    else_operation=StandardOperation(target=0, op_type=OpType.y),
    control_bit=0,
)
```

If no else operation is needed, the `if_()` method can be used.

```python
qc.if_(op_type=OpType.x, target=0, control_bit=0)
```

### End of support for Python 3.9

Starting with this release, MQT Core no longer supports Python 3.9. This is in
line with the scheduled end of life of the version. As a result, MQT Core is no
longer tested under Python 3.9 and no longer ships Python 3.9 wheels.

## [3.2.0]

The shared library ABI version (`SOVERSION`) is increased from `3.1` to `3.2`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

With this release, the minimum required C++ version has been raised from C++17
to C++20. The default compilers of our test systems support all relevant
features of the standard. Some frameworks we plan to integrate with even require
C++20 by now.

The `dd.BasisStates`, `ir.operations.ComparisonKind`,
`ir.operations.Control.Type`, and `ir.operations.OpType` enums are now exposed
via `pybind11`'s new `py::native_enum`, which makes them compatible with
Python's `enum.Enum` class (PEP 435). As a result, the enums can no longer be
initialized using a string. Instead of `OpType("x")`, use `OpType.x`.

## [3.1.0]

The shared library ABI version (`SOVERSION`) is increased from `3.0` to `3.1`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

Even though this is not a breaking change, it is worth mentioning to developers
of MQT Core that all Python code (except tests) has been moved to the top-level
`python` directory. Furthermore, the C++ code for the Python bindings has been
moved to the top-level `bindings` directory.

### DD Package

The `makeZeroState`, `makeBasisState`, `makeGHZState`, `makeWState`, and
`makeStateFromVector` methods have been refactored to functions taking the DD
package as an argument. These functions reside in the `StateGeneration` header.
Any existing code that uses these methods must replace the respective calls with
their function counterpart.

## [3.0.0]

This major release introduces several breaking changes, including the removal of
deprecated features and the introduction of new APIs. In preparation for this
release, most direct dependents of MQT Core have been updated to use the new
APIs. The following sections describe the most important changes and how to
adapt your code accordingly. We intend to provide a more comprehensive migration
guide for future releases.

### Intermediate Representation (IR)

The OpenQASM parser has been encapsulated in its own library, which is now a
dedicated target in the CMake build system. Any use of
`qc::QuantumComputation::import...` needs to be replaced with the respective
`qasm3::Importer::load...` function.

Several parsers have been removed, including the `.real`, `.qc`, `.tfc`, and
`GRCS` parsers. The `.real` parser lives on as part of the [MQT SyReC] project.
All others have been removed without replacement.

The `Teleportation` gate has been removed from the IR. This was a placeholder
gate and was only used in a single method (in [MQT QMAP]), which is bound to be
removed as part of [MQT QMAP] `v3.0.0`.

[MQT QCEC], [MQT QMAP], and [MQT DDSIM] have been updated to use the new API,
which will be released in [MQT QCEC] `v3.0.0`, [MQT QMAP] `v3.0.0` and
[MQT DDSIM] `v2.0.0`.

### DD Package

The DD package has undergone some initial refactoring to streamline the
implementation and prepare it for future extensions. The `Config` template has
been removed in favor of a constructor that takes the configuration as a
parameter. Any existing code using `dd::Package<...>` needs to be updated to use
`dd::Package` or `dd::Package(numQubits, ...)` instead. The `MemoryManager` and
adjacent classes have been refactored to remove the template parameters. This
should not have user-visible effects, but it is a breaking change nonetheless.
Depending libraries may now also use the `mqt-core` Python package to interact
with the DD package.

[MQT QCEC] and [MQT DDSIM] have been updated to use the new API, which will be
released in [MQT QCEC] `v3.0.0` and [MQT DDSIM] `v2.0.0`.

### Neutral Atom Quantum Computing

The `NAComputation` class hierarchy has been refactored to use an MLIR-inspired
design. This will act as a foundation for future extensions and improvements.

[MQT QMAP] has been updated to use the new API, which will be released in
[MQT QMAP] `v3.0.0`.

### General

MQT Core has moved to the
[munich-quantum-toolkit](https://github.com/munich-quantum-toolkit) GitHub
organization under <https://github.com/munich-quantum-toolkit/core>. While most
links should be automatically redirected, please update any links in your code
to point to the new location. All links in the documentation have been updated
accordingly.

MQT Core now ships all its C++ libraries as shared libraries with the `mqt-core`
Python package. Depending packages can now solely rely on the Python package for
obtaining the C++ libraries. This is demonstrated in [MQT QCEC] `v3.0.0`,
[MQT QMAP] `v3.0.0` and [MQT DDSIM] `v2.0.0`, which will be released in the near
future.

MQT Core now requires CMake 3.24 or higher. Most modern operating systems should
have this version available in their package manager. Alternatively, CMake can
be conveniently installed from PyPI using the
[`cmake`](https://pypi.org/project/cmake/) package.

It also requires the `uv` library version 0.5.20 or higher.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/core/compare/v4.0.0...HEAD
[4.0.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.10.0...v4.0.0
[3.10.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.9.2...v3.10.0
[3.9.2]: https://github.com/munich-quantum-toolkit/core/compare/v3.9.1...v3.9.2
[3.9.1]: https://github.com/munich-quantum-toolkit/core/compare/v3.9.0...v3.9.1
[3.9.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.8.0...v3.9.0
[3.8.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.7.0...v3.8.0
[3.7.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.6.0...v3.7.0
[3.6.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.5.1...v3.6.0
[3.5.1]: https://github.com/munich-quantum-toolkit/core/compare/v3.5.0...v3.5.1
[3.5.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.3.0...v3.4.0
[3.3.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/munich-quantum-toolkit/core/compare/v2.7.0...v3.0.0

<!-- Other links -->

[MQT DDSIM]: https://github.com/munich-quantum-toolkit/ddsim
[MQT QMAP]: https://github.com/munich-quantum-toolkit/qmap
[MQT QCEC]: https://github.com/munich-quantum-toolkit/qcec
[MQT SyReC]: https://github.com/munich-quantum-toolkit/syrec
[CMake presets]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
