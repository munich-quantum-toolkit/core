# Split target-independent gate fusion from target-native synthesis

Status: historical implementation record.

## Goal and scope

Two-qubit gate fusion and hardware lowering must answer different questions.
Before routing, target-independent two-qubit gate fusion should rewrite a
sequence only when doing so strictly reduces its two-qubit gate count, without
choosing a hardware basis. After routing, the compiler should lower operations
that the real `mlir::CompilerTarget` does not support and remove routing SWAPs
on targets that do not declare SWAP native. A final, independently runnable pass
rejects unsupported operation types, arities, and parameter counts as well as
dynamic allocations and unknown static target sites.

After this change, C++ pipeline code can construct these three stages
independently. Focused tests demonstrate a profitable CX cancellation, preserve
isolated and runtime-parameterized gates before routing, lower SWAP to a
target-selected basis after routing, and reject operation, arity, parameter,
allocation, and site mismatches.

## Constraints

- The old two-qubit pass performed four jobs: single-qubit target-basis
  lowering, two-qubit gate fusion, isolated two-qubit lowering, and residual
  menu checking. Evidence: its `hasNonNativeGate` condition could rewrite a
  pre-routing window without reducing the entangler count.

- CT-01 already recognizes all fifteen gates understood by the deleted menu,
  including gate aliases and the same entangler preference. The old
  `NativeGateset` duplicated the enum, parser switch, basis resolution, and
  operation classifier.

- Mapping emits `qco.static` with hardware identifiers, while `CompilerTarget`
  permits sparse target IDs. Conformance only needs to validate those
  declarations once; operation capabilities are homogeneous and do not require
  operand-by-operand site tracing.

- Applying an MLIR greedy rewrite driver can reorder an unrelated constant even
  when the quantum pattern does not match. The target-specific transforms now
  precompute their work and use `IRRewriter` directly, so a no-op fusion
  preserves the module byte-for-byte.

- Pull request #1969 established the useful progressive ordering of
  decomposition, optional routing, and late native synthesis, with a test that
  routed SWAPs disappear. Its `targetNative` Python duck typing and coupling CLI
  are intentionally excluded.

- Current targets expose homogeneous gate sets. Ordered
  `Operation::siteTuples()` retain calibration data, while
  `CompilerTarget::supports()` depends only on canonical operation name, arity,
  and parameter count.

- `qco.pow`, like `qco.ctrl` and `qco.inv`, is a target-visible unitary shell
  with a region body. Synthesis and conformance must classify the shell and skip
  its implementation body.

- A generic walk rewrite driver cannot safely anchor this multi-operation fusion
  at the run head because the rewrite erases operations the driver has not
  visited. Precollecting non-overlapping run heads and using `IRRewriter`
  directly is both safer and lighter.

- The earlier per-operation target-site tracer was unnecessary once support
  became homogeneous. Removing it eliminates both its quadratic worst case and
  its structured-control-flow special cases. Profitable windows also reuse the
  same prepared Weyl decomposition for counting and emission.

## Decisions

- Remove the native-menu pass and all high-level menu APIs in this slice instead
  of retaining a compatibility adapter. Rationale: the series intentionally
  moves callers to a typed `CompilerTarget`; a synthetic menu target would
  preserve the parallel configuration model this change removes.

- Use `CompilerTarget::SingleQubitBasis` directly throughout target, Euler, and
  Weyl code and delete `NativeSynthesisBasis`. Rationale: one enum and one
  synthesis-basis value remove a conversion switch, adapter files, and an
  otherwise redundant coverage test while preserving dependency direction.

- Measure pre-routing profitability by a strict reduction in two-qubit basis
  uses and materialize a canonical U/CZ sequence only after the comparison
  succeeds. Rationale: two-qubit operations drive routing cost, and selecting a
  hardware basis before routing would conflate optimization with target
  legality; CZ avoids introducing an arbitrary control direction.

- Let `CompilerTarget::supports` decide whether an ordinary `qco.swap` requires
  post-routing lowering. Rationale: the target is the sole capability authority;
  routing SWAPs still lower on ordinary targets that do not report SWAP, while a
  target-native SWAP must remain legal even when the target has no global
  synthesis basis.

- Treat operation capabilities as homogeneous across a target. Ordered site
  tuples retain calibration only; synthesis and conformance query canonical
  name, arity, and parameter count without directional fallback. Rationale:
  current target gate sets are uniform and bidirectional, so site tracing and
  reverse probes add code without changing compilation behavior.

- Preflight all target-lowering needs and apply planned rewrites directly with
  `IRRewriter`. Rationale: failure remains atomic and generic greedy/fixpoint
  work is unnecessary.

- Do not require `CompilerTarget::synthesisBasis()` at pass construction or pass
  entry. Rationale: absent operations mean all operations are native, and an
  incomplete explicit target can still describe a conforming program.
  Missing-basis failure matters only after an unsupported operation actually
  needs lowering.

- Keep gate fusion, target-native synthesis, and conformance as separate manual
  factories rather than textual passes. Rationale: `CompilerTarget` is an
  immutable typed C++ value that cannot be faithfully represented by generic
  pass options, and separate factories make each stage independently testable
  and benchmarkable.

## Outcome and validation

The implementation now has one homogeneous capability authority and three
separately observable transform stages. The latest cleanup removes the
decomposition basis adapter, general greedy rewrite machinery, directional
capability fallback, and per-operation target-site tracer. It retains explicit
CZ emission and failure-atomic lowering while adding direct static-site and
dynamic-allocation and quantum-function-input conformance coverage. Final
release builds pass 21 target-synthesis, 215 compiler, 33 dialect-utils, 27
mapping, and 199 decomposition tests. The SC device suite passes 41 tests with
one expected job-ID skip. Both affected interface-header targets build, all
repository hooks pass, focused LLVM 22.1.8 `clang-tidy` reports no new
diagnostics, and an independent review approves the exact working tree.

The main design lesson is that target support and fusion profitability must not
share a configuration surface. Canonical U/CZ gate fusion is useful without
hardware knowledge, while post-routing lowering and conformance require the
target's homogeneous operation set and declared static sites.

## Code and ownership

`mlir/include/mlir/Compiler/Target.h` defines the immutable target. Its
`supports(Operation*)` query recognizes QCO operation semantics and checks
canonical name, arity, and parameter count. Ordered operation site tuples retain
calibration only. Its `synthesisBasis()` query returns one usable single-qubit
basis and entangler only when both exist.

`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp` contains
the two-qubit gate-fusion scanner, three pass implementations, static-site
validation, and diagnostics. Public factory declarations live in
`mlir/include/mlir/Dialect/QCO/Transforms/Passes.h`.

`mlir/include/mlir/Dialect/QCO/Transforms/Decomposition/Euler.h` aliases the
target-owned `SingleQubitBasis`; no decomposition-layer basis DTO remains.
`mlir/lib/Dialect/QCO/Transforms/Decomposition/Weyl.cpp` caches the selected
entangler decomposer, returns a prepared decomposition, and emits its
single-qubit factors without recomputing it.

QCO qubits use linear static single assignment: each operation consumes a qubit
value and returns its successor. After mapping, `qco.static` operations declare
the assigned target sites. Since operation capabilities are homogeneous,
conformance validates these declarations once and does not trace each operand's
SSA lineage.

Focused tests are in
`mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/test_target_synthesis.cpp`.
Typed basis coverage remains in
`mlir/unittests/Dialect/QCO/Transforms/Decomposition/test_weyl_decomposition.cpp`.

## Acceptance

Acceptance requires the following observable behavior:

- Two adjacent constant CX operations fuse away with equivalent unitary
  behavior, while a three-CX SWAP form, an isolated SWAP, and a runtime RXX run
  remain quantum-structurally unchanged before routing.
- Target-native synthesis removes unsupported ordinary SWAP and produces only
  operations accepted by the selected target basis, preserving the complete
  unitary. A target-native SWAP remains unchanged without requiring a synthesis
  basis.
- Homogeneous operation capabilities apply in both operand orientations while
  ordered site tuples retain calibration data only. Synthesis preserves complete
  unitary behavior without an operand-reversal option, and a native `qco.pow`
  shell is checked without separately rejecting its implementation body.
- An absent operation set succeeds without synthesis. An explicit incomplete
  target succeeds for supported operations and reports “no usable synthesis
  basis” only when an unsupported operation actually needs lowering.
- A supported runtime-parameterized gate remains unchanged. An unsupported
  runtime gate reports that its unitary matrix is unavailable at compile time
  without partially rewriting an earlier constant gate.
- Conformance accepts sparse target IDs, rejects operation-type, arity,
  parameter-count, unknown-site, measurement, and dynamic-allocation mismatches,
  and does not reconstruct per-operation site provenance.
- The duplicate enum/parser and all native-menu text, CLI, C++ program, Python,
  and generated pass surfaces are absent.
- Focused tests, decomposition tests, compiler tests, changed-file checks,
  repository lint, and `git diff --check` pass.
- Every commit is signed and carries the required `Assisted-by` trailer. The
  implementation commit additionally carries
  `Co-authored-by: Simon Hofmann <simon.t.hofmann@tum.de>`.

## Interfaces

`CompilerTarget` remains dependency-light and does not depend on transform
libraries. `MLIRQCOTransforms` publicly links `MQTCompilerTarget` because its
public decomposition headers use compiler-target basis types. The public,
independently constructible factories are:

    std::unique_ptr<Pass> createFuseTwoQubitGates();
    std::unique_ptr<Pass>
    createTargetNativeSynthesis(const CompilerTarget& target);
    std::unique_ptr<Pass>
    createVerifyTargetConformance(const CompilerTarget& target);

There is no textual target-native pass, synthetic target, native-gate menu, or
parallel capability model.
