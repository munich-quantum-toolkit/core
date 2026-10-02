# Canonicalization performance audit

Status: findings implemented and re-audited for
[issue 2253](https://github.com/munich-quantum-toolkit/core/issues/2253).
Base: `1b9d08911cdfce55e801e5b0791b4437c2bd2e4a`. Scope: CBit, MQT, QC, QCO, and
QTensor canonicalization, parameter validation, modifier and gate-matrix
helpers, and their compiler, export, cleanup, and mapping consumers.

## Result

All four reported cases are addressed. No additional release-blocking defect or
observable nondeterminism was found in this scope. This does not complete the
repository-wide issue.

## Resolved findings

### Adjacent CBit reads

`findKnownLoadValue` in `mlir/lib/Dialect/CBit/IR/CBitOps.cpp` reuses the
immediately preceding load only if register and index SSA values match. No
intervening write is possible. Both QC and QCO builders produce these loads, and
cleanup runs canonicalization before CSE. The shortcut avoids repeated backward
scans for adjacent identical reads while preserving existing store,
initialization, region, and unknown-user handling.

Regression coverage checks dynamic-index identity, distinct registers/indices,
and intervening stores. Distinct-index and non-adjacent scans retain their
existing worst-case quadratic cost. A wider cache or backward-search policy is
not introduced.

### Direct gate parameters

`mqt::verifyFiniteConstantParameters` uses the standard MLIR constant matcher
and no longer allocates per-gate lookup tables. Both QC and QCO use it.
Parameter expressions and block arguments require no traversal. The allocation
argument follows from the source; allocator calls were not counted.

### QTensor commuting

`CommuteInsertExtractChains` in
`mlir/lib/Dialect/QTensor/IR/Operations/InsertOp.cpp` groups provably commuting
accesses in one rewrite. It finds the commuting prefix even when the greedy walk
starts near the tail, then preserves extraction and insertion order. Inserts
move after the last collected extract. Trailing inserts stay in place so their
operands need not dominate an earlier position.

Unknown operations, dynamic indices, same-slot dependencies, and block or
source-order boundaries stop batching. Mapping's `discoverComputation` in
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` still receives the
required extract-before-insert form. The old pair rewrite takes N(N-1)/2
commutations on a flat distinct-slot chain. Tests check one-rewrite
normalization from the last pair, slot identity, dominance, linearity, and index
barriers. The rewrite does not guarantee linear behavior for arbitrary graphs.

### Parameter-expression scans

Finite gate parameters are a precondition of valid QC/QCO programs, including
values produced at runtime. Operation verifiers retain a direct-constant sanity
check. They do not traverse or fold expression graphs to establish finiteness.

There is no program-wide finite-parameter validator, cache, or extra validation
around pipelines and exports. Ordinary MLIR structural verification remains in
place. Existing point-of-use checks reject values that numerical algorithms or
external formats cannot handle. Guards against overflow or phase loss from valid
finite inputs remain necessary.

Tests that require discovering hidden non-finite values, checking unselected
operands, or rechecking after arbitrary mutations are outside this contract.
Direct NaN/infinity sanity checks and numerical correctness tests remain.

## Final review and validation

- Producer/consumer review confirms CBit memory barriers, tensor slot identity
  and dominance, and mapping's extraction-before-insertion requirement.
- Hash tables are used for lookup, not observable iteration. New traversal
  follows IR and operand order. Existing conditional result forwarding keeps
  result order; QTensor canonicalization sorts accessed indices; modifier sets
  test membership. Whole-compiler cross-process determinism was not measured.
- Gate-matrix helpers retain fixed-size storage, phase-sensitive identities,
  numerical limits, and bounded integer powering. No speculative global cache or
  container conversion is added.
- Native `release-clang-ipo` build and 1467 focused C++ tests pass: 14 CBit, 42
  MQT utilities, 44 QTensor, 367 QC, 579 QCO, 209 QC translation, 212 compiler.
- Both mqt-cc CTest cases and 445 Python MLIR/Qiskit tests pass on the rebuilt
  extension. QDMI cases use the native build's device configuration. Stub
  generation passes without tracked stub changes.
- `uvx nox -s lint` and full-file `uvx nox -s cpp-lint` pass.
- Full repository tests, application profiles, and hosted CI were not run as
  part of this audit.
