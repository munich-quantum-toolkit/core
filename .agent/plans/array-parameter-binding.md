# Array parameter binding

Status: complete. Shared array binding is implemented and validated. Benchmark
adoption belongs to the dependent QAOA change.

## Goal and scope

Extend QC/QCO parameter binding to named, statically sized, one-dimensional
`f64` tensors alongside scalar `f64` inputs. A binding map may contain both
scalar values and arrays. Binding replaces entire inputs; unknown names, wrong
value kinds, length mismatches, and non-finite values fail without changing the
program. Partial binding preserves other inputs and their metadata.

The compiler owns validation and replacement in
`mlir/lib/Compiler/ParameterBinding.cpp`; Python forwards values through the
existing binding API. Preserve source compatibility for scalar C++ maps and
brace-initialized calls. Bind arrays to dense constants using existing MLIR
facilities. Do not add arbitrary-rank, dynamic-shape, or elementwise binding.

QC/QCO and binary jeff already support fixed-length array inputs. Unbound
OpenQASM array export remains unsupported; QIR requires all inputs bound.
Document this boundary rather than extending exporter scope.

## Validation

All 4,031 native and 1,974 Python tests passed, including 301 compiler tests.
The binding tests cover mixed and partial binding, atomic failures, scalar C++
calls, metadata preservation, and bound OpenQASM/QIR export. The 16 Python
binding cases passed with NumPy 2.3.2 and 2.5.4, including sampling and NumPy
scalar conversion. Independent matrix comparisons checked signed array elements
and jeff/OpenQASM round trips; loop inputs also reached both QIR profiles after
explicit unrolling.

Python stubs were regenerated. Whole-file C++ lint, repository lint, executable
documentation, and local documentation links passed. The binding guide was
inspected in a browser. Large arrays were not allocated or executed; supplying
their values still requires proportional memory.
