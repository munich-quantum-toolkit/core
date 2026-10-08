# Concrete DD operations

Status: complete.

## Outcome and scope

Node-specific DD operations are concrete overloads in `dd/Edge.hpp` and
`dd/CachedEdge.hpp`. C++ callers use `dd::getVector(edge)`,
`dd::getMatrix(edge, numQubits)`, and the corresponding free operations. Generic
`Edge<Node>` and `CachedEdge<Node>` handles retain their layouts and
node-independent operations. Python methods and array ownership stay the same.
The migration covers Package dispatch, DD and MLIR tests, DDSIM, bindings, and
`docs/cpp_api.md`.

## Decisions

Constrained exported template members have different symbol names across
supported GCC and Clang versions. Concrete overloads remove that constraint
mangling boundary without compatibility wrappers or new traits. Numerical
algorithms and error handling stay unchanged. File-local traversal helpers pass
the callback by reference; the public matrix traversal owns one callback copy.
Cached-edge class instantiations are unnecessary because all remaining members
are defined in the header. Out-of-line edge and hash instantiations remain.

## Validation

- Native release build and `ctest --preset release --parallel 4`: 3,950 tests
  passed; the superconducting device's unsupported job-ID property caused one
  expected skip. This includes the DD, MLIR, QIR, and DDSIM consumers. The build
  used GCC 13 and the assertion-free LLVM/MLIR 23.1.2 SDK without IPO.
- `uv run --no-sync pytest test/python -n4`: all 1,920 tests passed with both
  bundled devices enabled. `uvx nox -s stubs` produced no tracked stub changes.
- `uvx nox -s lint` and `uvx nox -s cpp-lint`: passed. C++ lint inspected every
  changed source file, including the full contents outside changed lines.
- Doxygen 1.17 built the native reference with the repository's Doxyfile and
  QDMI inventory configuration from `docs/_ext/cpp_api.py`. The C++ guide's GHZ
  example compiled and produced the expected amplitudes.
- Unoptimized GCC 13 and Clang 23 local consumers compiled and ran. Compile-time
  probes accepted the matching vector, matrix, and cached-edge overloads and
  rejected mismatched edge kinds. Both public edge headers compiled alone.

## Limits

Installed cross-compiler and LTO qualification belongs to the release-build
validation that accompanies this change. The complete Sphinx documentation build
and external link check were not run for this migration.
