# Concrete DD operations

Status: in progress; implementation and validation remain.

## Goal and scope

Expose node-specific DD operations as concrete overloads in `dd/Edge.hpp` and
`dd/CachedEdge.hpp`. C++ callers use `dd::getVector(edge)`,
`dd::getMatrix(edge, numQubits)`, and the corresponding free operations. Generic
`Edge<Node>` and `CachedEdge<Node>` handles retain their layouts and
node-independent operations. Python method names and array ownership stay the
same. The change reaches Package dispatch, DD and MLIR tests, DDSIM, bindings,
and `docs/cpp_api.md`.

## Decisions

Constrained exported template members have different symbol names across
supported GCC and Clang versions. Concrete overloads remove that constraint
mangling boundary without compatibility wrappers or new traits. Keep numerical
algorithms and error handling unchanged. File-local traversal helpers pass the
callback by reference; the public matrix traversal takes one callback by value.

## Work remaining

- [ ] Move concrete operations and migrate all callers.
- [ ] Regenerate binding stubs and verify DD, affected MLIR/Python tests, and
      lint.
- [ ] Review the final contracts and record validation limits.

## Validation

Build the native test targets and run DD tests plus the MLIR DD functionality,
QIR runtime/JIT, and Shor generation consumers. Run Python DD and compiler
tests, `uvx nox -s stubs`, `uvx nox -s lint`, and `uvx nox -s cpp-lint`.
Cross-compiler installed-consumer qualification belongs to the release-build
validation that accompanies this change.
