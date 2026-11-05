# Preserve target information in CLI QPY export

Status: complete; validated locally.

## Scope and decisions

The CLI compiles directly with `compile_program(output=OutputFormat.QPY)`. The
shared Python compiler owns optimization, target compilation, and QPY
serialization. No subprocess or intermediate QCO file is needed. Qiskit stays
optional, and serialization finishes before the destination is opened.

The QPY argument adapter retains input formats, standard input for OpenQASM and
MLIR, explicit targets and payloads, QDMI discovery/configuration, custom
pipelines, decomposition, seeds, and mapping controls. Native-only MLIR
debugging options are rejected. Version, help, and device-list requests still
use the native command.

Generic MLIR import selects QC or QCO from parsed operations and uses the typed
program validators. `Program.from_mlir_file` and the compiler convenience API
share that importer. Target and payload attribute factories use MLIR parsing and
the existing model validators; the CLI does not parse attribute syntax.

QPY export also reads an embedded target environment when no explicit target is
supplied. Reoptimizing mapped MLIR still removes its layout, as the native
compiler did previously. Export retains the target's aliases, address width, and
physical circuit action without reattaching stale layout metadata.

QPY reconstructs renamed standard-gate instances with their standard names.
Custom aliases therefore use a `Gate` with an explicit definition containing the
original standard operation. Parameter order, qubit order, and exact phase
survive serialization. Existing standard aliases keep their native
representation. Definitions also preserve semantics when an alias collides with
another standard name, such as an RX operation named `ry`.

## Validation

Built merged main at `ed8fe3a6f`, including #2925, and reproduced loss of target
layout and failures for `--emit=qpy` combined with version or hidden help. The
custom-alias tests also failed before the exporter fix.

The direct implementation passes 1,041 Python tests with each of Qiskit 2.5.0
and 2.6.1, and five native program-import tests. Stubs were regenerated.
Repository lint passes. Full-file C++ lint checked the changed implementation,
binding, adapter, and test files; the binding file was rechecked after
correcting its sole finding. Tests use local devices only.

Three QCO source/file import cases failed before the generic importer fix, while
the corresponding QC cases passed. Four embedded-target export cases failed
before the shared exporter recovered target information. Tests cover input
formats, standard input, custom pipelines, decomposition, numeric and boolean
option spelling, target/device/configuration selection, unsupported debug
options, and output preservation on failure.

## Timing evidence

The public command compiled the existing 16- and 64-qubit QFT inputs for the
same all-to-all target with a custom single-qubit gate alias. Three runs per
input on macOS ARM64 with Python 3.14.9 and Qiskit 2.6.1 gave median times of
1.894 s and 2.187 s. Earlier subprocess-path medians were 1.970 s and 2.268 s.
These historical measurements were not interleaved and do not establish a
statistical speedup. Imports still dominate; QPY serialization took about 5 ms
and 50 ms in the profiles. No memory claim was measured.

Both outputs match the previous operation names, parameters, and physical
operands, retain layout, and have the same sizes: 140,537 and 1,218,206 bytes.
The retained change removes a subprocess and intermediate IR round trip; no
broader performance rewrite is justified by these measurements.
