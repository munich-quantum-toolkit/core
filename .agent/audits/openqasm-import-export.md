# Contract audit: OpenQASM import and export

Status: complete; accepted findings implemented and locally validated. Original
baseline: `33dbc843e589d9e9166308084e825c9f8b2ff89d`. Simplification baseline:
`2a927489805c93e9931df65d6c6cf3e49e971324`. Date: 2026-09-08. Measurements used
native ARM64 and LLVM/MLIR 23.1.0.

## Scope and disposition

The scope covers `mlir/lib/Target/OpenQASM`, QC OpenQASM translation,
`bindings/mlir/qiskit`, their corresponding unit tests, and
`test/python/test_mlir_qiskit_translation.py`. The supported language subset is
in `docs/mlir/OpenQASM.md`; the Qiskit boundary is in
`docs/mlir/python_compiler_collection.md`. This audit does not establish full
OpenQASM conformance.

### Bounded analysis and emission

Affine reconstruction caches within one proof context and bounds its work.
Optional facts may be forgotten; required quantum-index proofs must succeed.
Intermediate overflow, induction domains, scope invalidation, and uncertain
alias rejection remain checked. Static operand distinctness uses a set before
pairwise affine proofs.

The original affine reproducer is `OPENQASM 3.1; int a = 1;` followed by N
copies of `a = a + a;`. With 20 assignments, analysis took 17.8 s before and
0.00072 s after under host contention. The static-distinctness reproducer is
`qubit[N] q; ctrl(N-1) @ x q[0], ..., q[N-1];`. At N = 64,000, analysis fell
from 7.21 s to 0.145 s. These are local scaling observations, not timing
promises.

An `OpBuilder` listener counts inserted operations; cancellation bounds further
construction. This replaces a predictive model that charged register-wide work
for native dynamic stores. The reproducer declares `bit[99999] c; output int i;`
with `i = 0;` and 40 copies of `c[i] = false;`. It must emit 40 stores within
the ordinary budget. The explicit output isolates this check from
initialization. Budget-boundary tests use the public importer and count actual
insertions.

### One syntax representation and explicit policies

The parser builds persistent expression IDs directly. Operands, bit references,
and modifiers have one shared representation. Gate-call arrays remain borrowed
while parsing and owned in stored syntax. Parsing remains separate from analysis
so source lifetimes, diagnostics, and reanalysis under different policies
survive. `GatePolicy` is passed directly to analysis; `OpenQASMImportOptions`
owns a flat policy and emission limit.

For 100,000 distinct declarations, the original syntax-ID change reduced peak
RSS from about 137,600 KiB to 88,000 KiB. Three-run median parse time fell from
0.66 s to 0.45 s. Removing duplicate leaf types has no separate speedup claim.

### Ordered outputs and scalar snapshots

The exporter preserves result order and integer zero as data. Absent outputs are
void; repeated register results are rejected rather than silently merged. The
original reproducer declares `output int a; output bit b;` and assigns
`a = 1; b = false;`: reimport must retain this result-type order. A lone
`output int answer; answer = 0;` must retain its result. Returning the same
`!cbit.reg` twice exercises alias rejection.

Entry-function scalar snapshots are materialized at their definition. Tests
check updates across loop iterations and simultaneous SCF assignments. Gate
bodies retain bounded expressions because OpenQASM gates cannot declare local
classical storage. Dynamic reads require full-register initialization; static
initialization facts and scalar generations remain supported.

Unused measurements emit `measure q;` without introducing an implicit classical
output. Used results retain storage. Zero-initialized registers use exact-width
bit strings, including widths above 64. Register comparisons use the existing
bit-vector representation while preserving out-of-range literal comparisons.

### Source and matrix correctness

Source-name validation shares the frontend keyword and builtin-constant policy.
Names such as `void`, `im`, `pragma`, and `pi` are renamed deterministically;
ordinary identifiers remain unchanged. Exporter-generated prefixes and gate
collisions remain exporter-owned. Include lookup retains its documented working
directory and explicit include-directory policy.

Floating `!=` round-trips through `arith.cmpf UNE`, whose NaN behavior differs
from `ONE`. The source reproducer is
`float f = 1.0; output bool b; b = f != 2.0;`. Float casts and exact-width bit
strings use the existing expression model.

Strict-policy lowering tests the emitted catalog bodies themselves.
Compatibility mode may replace a recognized body with its native operation, so
that mode alone cannot prove a helper's matrix. Phase-sensitive matrices exposed
native `u` and `u2` mismatches with OpenQASM `U`. A shared helper applies
`gphase(-theta/2)` before `U`; controlled tests preserve relative phase as well
as amplitudes.

### Qiskit normalization and parameter identity

Private gate formals are positional. Export generates local names directly;
private spelling is not a round-trip promise. Public inputs retain name and
parameter-vector validation. Calls bind the explicit formal list by position, so
local renaming does not reorder actual arguments.

Ordinary canonicalization does not require weakening the two float-castable
symbol tests. On LLVM/MLIR 23.1, `theta - theta` remains an `arith.subf`, and
the Qiskit-imported representation of `(theta - theta) + 2` retains `theta * 0`.
Without fast-math assumptions these are not universally constant floating-point
expressions. Canonicalization does fold literal arithmetic and integer-to-float
casts, allowing removal of the exporter's special cast recognizer.

Native folding is enabled in the existing integer-expansion rewrite driver,
which runs on a clone. No separate pass or custom cast recognizer is needed.
Identity tests cover explicit cleanup and several numeric bindings. Constant
measurement-index and low-bit tests permit folding; rejection tests use dynamic
operands and a non-foldable expression DAG.

A full canonicalizer was tested and rejected for implicit export normalization:
it removes unused standalone qubits and forwards CBit loads from measurements,
which can require extra scratch bits and stores in Qiskit. Those transformations
need separate resource and snapshot decisions. Native folding preserves the
existing measurement, layout, and control-flow tests. Explicit `cleanup()`
remains available to callers.

## Retained limits and rejected deletions

- Static-only quantum indexing was rejected; affine loops remain supported.
- Resource bounds, fixed-angle quantization, integer-power overflow checks, and
  uncertain-alias rejection protect distinct contracts and remain tested.
- Snapshot and initialization checks can be removed only when another
  representation enforces their guarantees.
- General arrays, runtime angles, and register subroutines remain separate work.
  Relevant original overlaps were issues #2413, #2414, and #2427–#2429.
- Equal emitted strings or overlapping coverage do not justify deleting
  independent matrix or semantic regression tests.

## Validation

The release build and `ctest --preset release` pass: 3,201 tests passed and one
optional QDMI job-ID test skipped. `uv run --no-sync pytest test/python -q`
passes all 1,156 tests, including 328 Qiskit translation tests. The 493
OpenQASM-focused CTest cases also pass. Repository lint passes and regenerated
stubs are unchanged. Full-file C++ lint of the exact PR diff passes with zero
findings. Hosted CI is separate.
