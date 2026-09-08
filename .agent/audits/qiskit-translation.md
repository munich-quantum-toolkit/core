# Qiskit import, export, and C API audit

Status: findings implemented. Audit baseline:
`33dbc843e589d9e9166308084e825c9f8b2ff89d`, initially clean. Integration base:
`3be5ee96f`, including independent measurement scheduling fixes. Environment:
native ARM64, Python 3.14, Qiskit 2.5.2, LLVM/MLIR 23.1.

## Outcome and contracts

The audit covered `bindings/mlir/qiskit/`, the version registry and vendored
headers, `scripts/qiskit_c_api_adopt.py`, dependency declarations,
documentation, and translation, loop, integer-interchange, and adoption tests.
It traced standard-gate descriptors, reusable function bindings, CBit storage,
and resource mapping where they constrain translation.

Import still completes validation before creating its unpublished MLIR module.
Export preflights the normalized circuit before constructing Python objects.
Both routes preserve source immutability and return only complete results. The
generic model remains Python-object-free; Qiskit-specific operations stay in
`Qiskit2_5.cpp`. See
[the implementation record](../plans/qiskit-translation.md).

## Findings and disposition

### Reject names before native string conversion

Qiskit accepted NUL-containing quantum-register, classical-register,
custom-gate, and bound loop-parameter names, but native name access aborted the
process. Four subprocess probes reproduced the failure on the audit baseline.

`pythonStringAttribute` now rejects NUL characters before native access.
Register and loop-parameter names are read through Python. Regression tests
cover all four paths; existing empty-name and parameter-vector checks remain.

### Use one output owner

`NativeCircuitWriter` now owns a private Python circuit throughout emission.
`createBlock` inherits exact parent bits, registers, parameters, and available
variables. Captures propagate through enclosing blocks. Numeric instructions use
a freshly borrowed native view; symbolic standard instructions use
`CircuitInstruction.from_standard` and append to the owned `CircuitData`.

This removes placeholder barriers, deferred indices, circuit rebasing, parameter
unification, vector restoration, and the native symbolic parameter allocator.
Controlled unitaries replace their just-appended operation immediately. The
private output never enters a builder scope or receives cached duration data.
Existing custom-Gate, symbol, vector, capture, modifier, matrix, and ownership
tests constrain this boundary. Vector elements outside their vector's current
size remain supported.

### Make ABI comparison reflect current usage

The old extractor missed raw `_Qk_API_Circuit[38]` access and `QkComplex64`.
Mutating the parameterized-gate slot produced an unchanged surface. Comparing
saved JSON with a current extraction of identical headers also reported 20 added
functions, 22 removed functions, and five removed types: adapter drift, not an
upstream ABI change.

Both snapshots are now extracted using the same current implementation. The
extractor resolves raw capsule accesses or fails closed, reads typedefs from all
headers, and follows types used by function signatures and other used typedefs.
The writer no longer needs the raw parameterized-gate workaround. Tests cover
raw slots, complex matrix types, and stale saved metadata. Exact header
provenance and behavioral tests remain necessary; the extractor is a review aid,
not an ABI proof.

### Derive test selection from the registry

Translation, loop, and integer tests had independent 2.5.x guards. A shipping
adoption of another minor skipped the translation module and exited pytest with
code 5. The common test helper now reads `SupportedVersions.inc`, and adoption
runs all three suites. A registry-mutation test checks newly registered minors
without claiming compatibility with an untested Qiskit release.

### Index writes and materialize snapshots by need

Each snapshot consumer previously scanned every operation back to its source
read. An unchanged source read shared by 16,000 stores took about 3.5–3.8
seconds to export, versus about 0.15 seconds with adjacent reads.

Writes are now indexed once in block order, including nested effects. Snapshot
checks use binary search over relevant writes. Supported runtime scalars are
materialized for writes that precede their consumers, region crossings,
control-flow edges, and deep expression chains. The policy no longer depends on
unrelated SCF results. A measured bit saved before a second measurement
therefore survives `cleanup()` and subsequent export.

Former scalar-rejection tests now assert observable saved-value behavior. Wide
reads, expression-size limits, definite initialization, and unsafe measurement
fusion remain guarded. The existing main-branch measurement scheduling contract
is retained: unitary operations and resets may separate a measurement from its
store, while intervening classical accesses or control flow remain unsupported.

### Bound generated dispatch

A one-level 70-element list loop with a bound parameter and `continue` imported
but could not export because recursive dispatch exceeded the nesting limit. List
loops with jumps now use balanced dispatch. Arithmetic-progression lists without
jumps use range lowering, with checked arithmetic at integer limits.
Variable-bearing switches also use balanced dispatch, and expansion accounting
charges repeated labels for duplicated bodies.

The 64-level budget applies to generated SCF as well as source control flow.
Tests cover regular and irregular 70-element lists, large switches selecting
first, last, and default cases, and source nesting within the budget whose
generated nesting exceeds it. Irregular lists remain supported within these
explicit resource limits.

### Reduce repeated import and numeric export work

Expansion counting and loop-jump inspection classify native primitives without
normalizing their parameters. Full preflight remains, so primitive parameters
are decoded at most twice rather than three times. A parameter-read-count test
protects that bound without requiring an instruction cache proportional to the
input size.

All-number gate parameter lists now use `qk_circuit_gate` and retain
finite-value checks. Expressions carrying symbols remain symbolic even when
float-castable.

### Remove redundant implementation state

Removed dead root-pointer plumbing, parameter-factory forwarding helpers, the
unreachable scalar decoder for non-gate instructions, and register-membership
bitmaps. Canonical register validation uses ordered contiguous ranges. Matrix
inversion conjugate-transposes in place, and version components use
`std::from_chars`. Existing malformed-layout, matrix/modifier, and version tests
remain the relevant checks.

### Clarify public support

The support table distinguishes reusable custom Gates from expanded generic
Instructions. Installation guidance separates broad SDK integration from direct
compiler translation, which requires a registered minor. No dependency extra or
unreviewed native ABI support was added.

## Retained boundaries

- Per-minor native isolation remains necessary for Qiskit's experimental C API.
- Parameter replay parsing preserves tracked symbols; `float()` cannot replace
  it.
- Structural definition interning remains necessary because Qiskit copies Gates.
  No measured workload justified a new hashing implementation.
- Register packing, parallel edge assignments, initialization checks, lexical
  captures, and matrix endianness preserve semantics and remain in place.
- Alias and transpiler-layout preservation remain separate work tracked by
  issues #2069 and #2070.

## Validation

Baseline: 477 tests passed; one adoption test that creates an unsigned fixture
commit was deselected. It remains excluded under the checkout's signing rule.

The implementation passed 506 focused tests against the integration base. See
[the decision record](../plans/qiskit-translation.md) for the validation
command, required checks, and measured tradeoffs. Hosted CI is separate from
local checks.
