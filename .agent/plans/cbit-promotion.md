# Classical register promotion

Status: complete; the supported paths and local validation are recorded below.

## Scope and ownership

The registered `promote-classical-memory` pass connects CBit to MLIR's SROA and
mem2reg interfaces. It normalizes indices before preparing wide comparisons.
CBit owns decomposition and comparison preparation; QC/QCO cleanup, Base QIR
preparation, and payload loop unrolling reuse the pass.

Private registers of at most 256 bits support constant-index accesses,
whole-register writes, packed reads up to 64 bits, and wider reads used only in
constant comparisons. Returned registers, arguments, dynamic accesses, and
opaque users retain storage. Output widths are independent of the promotion
bound. Each allocation owns distinct storage; register arguments may alias.

QCO declares native promotion interfaces. Typed builders preserve operation
properties and place new classical results before positional quantum results.
MLIR owns SSA construction and structured loop handling. Upstream helpers that
append results cannot preserve this ordering, so the two QCO branch operations
share one replacement helper.

Reading an undefined bit before its first write remains undefined behavior;
promotion may choose zero. Allocation-based folding owns zero initialization.
Local forwarding preserves known snapshots without tracking later writes or
assuming arguments cannot alias. The block-level register tracker is removed;
scoped branch-condition propagation and bounded canonicalization remain.

Qiskit uses exact-width integer expressions and scalar snapshots. Its redundant
initialization tracker is removed; region-local zero allocations still reset on
every entry. Switch selectors remain limited to 64 bits. jeff's limit on general
packed integer arithmetic is unchanged.

## Outputs and QIR contracts

DD sampling and Adaptive QIR record explicit Boolean results alongside CBit
registers, preserving order, repetitions, and snapshots. DD resolves concrete
and deferred scalar bits through the same representation as register elements.
Other classical result types diagnose instead of silently selecting quantum-wire
output. Statevector extraction is unchanged.

Adaptive QIR preserves Result arrays when measurement stores fuse safely.
Otherwise it uses Boolean storage, including runtime indices and returned
measurements whose slots are overwritten. All Boolean output loads precede
output recording. Computed scalar outputs use function results, without
temporary registers. Unsupported register calls and mixed Result/Boolean storage
merges still diagnose. Base QIR requires measured outputs because static Results
have no specified initial value; Adaptive QIR preserves initialized zeros.

The QIR review uses specification tag `2.1`, revision
`a177eee5885d99e965f5cb919c8d70b0ab7a6a15`. Boolean initialization uses a native
LLVM array store. Fixed contiguous classical copies use aggregate loads and
stores with descriptor offsets and conservative alignment. These use standard
LLVM semantics consistent with the specified native-array operations; the
specification does not name a general array-copy operation. Array element types
contribute their required classical capabilities. Other ranked layouts use
loops; unranked copies diagnose. Dynamic indexing alone does not imply backward
branching; actual CFG cycles determine that capability.

LLVM math intrinsics remain backend extensions: the QIR floating-point
capability guarantees basic arithmetic and comparisons, not `sqrt` or
transcendental functions. Legacy QIR-Runner math names do not establish a
portable 2.1 ABI. Reusing integer-expansion patterns is also not automatically
portable: the current SWAR popcount expansion relies on multiplication overflow,
for which the QIR profile permits undefined behavior.

## Validation

The integration base is `6fffd4a68f1b24fae713bff5b49560fdd608245f`. Local checks
use LLVM/MLIR 23.1, Python 3.14, and Qiskit 2.6.0. The native suite passes 4,062
tests in release and debug coverage builds. The nine affected Python suites pass
1,159 tests; all 77 QIR compilation and runner tests also pass after the final
metadata adjustment. Repository lint, whole-file C++ lint, executable
documentation, and internal links pass. Rendered QIR guidance was inspected in
Chrome. These are local results, separate from hosted CI and hardware execution.

Run the native suite with `ctest --preset release`. Run affected Python tests:

```sh
uv run --no-sync pytest \
  test/python/test_mlir.py test/python/test_mlir_loops.py \
  test/python/test_mlir_integer_interchange.py \
  test/python/test_mlir_qiskit_translation.py test/python/test_qirrunner.py \
  test/python/test_qco_dd.py test/python/qdmi/test_compilation.py \
  test/python/test_mlir_parameter_binding.py \
  test/python/test_mlir_qiskit_target.py
```

- CBit and classical-simplification tests cover eligibility, aliases, snapshots,
  initialization, branch result positions, and zero-trip and nested loops. The
  wide-comparison test checks 480 results before and after promotion against
  APInt, including both operand orders, signed boundaries, and interior bits.
- Integer-interchange tests compare all predicates with Python integers through
  DD, OpenQASM, jeff, and Qiskit. A runtime-indexed 150-bit register retains its
  packed read; a branch-folded static index promotes.
- QDMI compilation tests execute 50-, 100-, and 150-bit programs as Base and
  Adaptive QIR and check every output bit. Array-copy tests compare DD and QIR
  against explicit expected results for Boolean and floating-point subviews.
- Native QIR tests cover capability flags and output-phase ordering. Moving an
  output load after recording can preserve simulator results while violating
  that ordering contract, so these structural checks remain necessary.
- Independent local probes exhaustively check 436,880 small-width comparisons.
  Deliberately reversing bit traversal, skipping an interior bit, or omitting
  sign bias fails the semantic reference. Delayed array copies and incorrect
  offsets also fail the snapshot reference.

## Performance evidence

The comparison implementation removes duplicate equality tracking and builds
ordering with one AND/OR per bit. It also avoids constructing a slot map merely
to check promotion eligibility and reserves maps of known size.

Paired measurements against `d107a14b56395e4f3f95128ae448c41fe80cac01` use the
same Clang 23.1 Release/ThinLTO build on a shared DGX Spark host. Each input
contains eight functions with independent Boolean inputs, a retained register
snapshot, an overwrite, and all ten integer predicates against an alternating
constant. After two warmups, 21 alternating runs measure process startup,
parsing, promotion, and textual output together:

| Register width | Before (median ms) | After (median ms) | Arithmetic ops before | After  |
| -------------- | ------------------ | ----------------- | --------------------- | ------ |
| 65             | 63.20              | 34.69             | 15,448                | 7,256  |
| 150            | 202.71             | 107.60            | 35,728                | 16,784 |
| 256            | 510.77             | 250.49            | 61,168                | 28,656 |

This demonstrates less generated IR and compiler work for wide comparisons, not
an end-to-end or hardware speedup. The 256-bit cap still bounds per-bit IR
growth. Aggregate copies add no allocation and require no loop for static
contiguous shapes.

## Remaining boundaries

Qiskit rejects scalar results and dynamic CBit loads; both exporters reject
repeated register results; OpenQASM rejects general arithmetic wider than 64
bits. Materializing Boolean results in fresh output registers permits Qiskit
export, with scratch bits still visible. Snapshots, zero-trip and nested loops,
and 150-bit bitwise operations export correctly.

Related work has distinct scope: #2591 addresses OpenQASM classical array
storage, #2806 frontend width limits, and #2626 complete QDMI outputs.
Integrating

## 2626 must preserve Boolean scalar output support. These changes do not define a

host/device ABI, general classical offloading, or wider numeric sampling.
