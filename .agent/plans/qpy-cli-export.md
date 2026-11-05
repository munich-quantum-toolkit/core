# Preserve target information in CLI QPY export

Status: complete.

## Scope and decisions

The CLI exports compiled QCO through a private binding that reuses the Python
compiler and QDMI QPY serializer. It recovers the canonical `mqt.target_env`
with `TargetEnvironmentAnalysis`, retaining native gate aliases and layouts for
sparse device sites. The binding consumes its input, including on failure.

The native compiler remains the owner of compilation options. Its temporary QCO
output already carries the complete environment, so no new transport or public
API is needed. Qiskit stays optional. Serialization remains buffered before
opening the destination. Version, help, and device-list requests pass through to
the native command.

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

The final build passes all 637 tests in `test_commands.py`, `test_cli.py`,
`test_mlir_qiskit_translation.py`, and `test_mlir_qiskit_target.py` with both
Qiskit 2.5.0 and 2.6.1. These cover sparse addresses, custom and standard
aliases, fixed and symbolic parameters, controlled gates, exact matrices, QPY
17, binary stdout, and preservation of existing files on failure.

Regenerated stubs with `uvx nox -s stubs --force-python 3.14`. Repository lint
and full-file C++ lint pass, including `QiskitAdapter.h` and
`register_mlir.cpp`. Validation used macOS ARM64 and LLVM/MLIR 23.1.0. No remote
hardware execution was needed.
