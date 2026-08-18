# Compile and export runtime classical control

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core accepts programs that measure qubits and use the results in structured
classical control, but target compilation previously had no way to state which
control forms a device could run. Mapping could also break the order between a
measurement store and the condition that reads it. Finally, direct Qiskit export
rejected structured control even though direct Qiskit import supported it.

After this work, a `CompilerTarget` declares its runtime classical-control
capabilities explicitly. Target compilation rejects unsupported programs before
it changes them, preserves quantum and classical ordering while it routes
supported conditionals, and exports supported `if`, `for`, `while`, and `switch`
operations directly to Qiskit 2.5. A narrow OpenQASM 3 compatibility path
retains arbitrary-width OpenQASM 2 register equality for the Benchpress
workloads whose registers exceed Qiskit's 64-bit classical-expression limit. The
user-visible acceptance test is that all 41 formerly skipped Benchpress
target/topology cases execute and pass their basis and coupling checks.

## Progress

- [x] (2026-08-18 17:25Z) Audited issues #2131 and #2071 and inventoried the 41
      Benchpress cases. The cases contain 14 unique OpenQASM 2 programs and use
      register-equality conditionals.
- [x] (2026-08-18 17:37Z) Added immutable, canonical classical-control
      capabilities to `CompilerTarget` in C++ and Python, including generated
      stubs and API tests.
- [x] (2026-08-18 17:37Z) Added a fail-closed target preflight as the first
      target-compilation pass. It checks reachable structured control, dynamic
      qubit indices, unsupported region control, and qubit-tensor state before
      any target-mutating pass.
- [x] (2026-08-18 22:59Z) Changed `qco.if` cleanup to thread only the accessed
      elements of a supported static rank-one QTensor. Mapping now treats unary
      structured control as a boundary and adds a full physical-wire workspace
      only when a cold routing preview proves that nested swaps need it. Added a
      stable topological reorder that preserves classical memory-access order
      while it restores SSA dominance after routing.
- [x] (2026-08-18 18:20Z) Added the narrow OpenQASM 3 compatibility form for
  complete register equality, arbitrary-width constants, eligible direct
  measurement stores, and stale-snapshot rejection.
- [x] (2026-08-18 19:35Z) Implemented recursive Qiskit structured-control
      collection and a Qiskit 2.5 writer that constructs validated control-flow
      operations through public Python classes after finalizing native C-API
      block circuits.
- [x] (2026-08-18 19:58Z) Added fail-closed validation for a conditional or
      delayed measurement destination store. The source-preservation regression
      and the complete 136-test post-rebase Qiskit translation file pass.
- [x] (2026-08-18 23:08Z) Canonicalized repeated pure OpenQASM register
      conditions within one unchanged classical-memory snapshot. The 301-bit
      workload now contains 602 condition nodes instead of 181,503 and target
      compilation takes 2.81 seconds in the focused all-to-all measurement.
- [x] (2026-08-18 19:50Z) Updated the public compiler, target, OpenQASM, and
      upgrade documentation for the final fail-closed capability and export
      design.
- [x] (2026-08-18 19:58Z) Validated the direct structured-control adapter with
      Qiskit 2.5.0 and 2.5.2. The focused semantic and round-trip suite passes.
- [x] (2026-08-19 00:20Z) Ran the exact 41 formerly skipped Benchpress
      target/topology cases against the final stable post-rebase module. All 41
      passed in 171.03 seconds of pytest time; all four `cc_n301` topologies
      executed normally and compiled in 1.89 to 3.64 seconds each.
- [x] (2026-08-19 00:20Z) Rebased the Core and Benchpress branches onto current
      `main`. The post-rebase tree passes 4,311 CTests, 136 Qiskit translation
      tests, the warnings-as-errors documentation build, the repository hooks,
      the 40-test Benchpress integration file, and the exact 41-case run.
- [x] (2026-08-19 06:59Z) Created Core pull request #2162 and added its real
      reference to the Core changelog without using a placeholder.

## Surprises & Discoveries

- Observation: the 41 Benchpress gaps are 36 QASMBench cases and five Feynman
  cases, but only 14 unique OpenQASM 2 inputs. Every input uses
  `if (creg == integer)`. Eight cases compare registers wider than 64 bits.
  Evidence: the inputs include `cc_n64`, `cc_n151`, and `cc_n301`, with the
  largest condition using 301 register bits.
- Observation: target capability declaration alone cannot compile these
  programs. QC-to-QCO conversion carries a fixed QTensor through `qco.if`.
  Routing must see each qubit as a scalar value, but the complete tensor also
  acts as a sequencing barrier for all its qubits.
- Observation: scalarizing only the tensor elements touched inside a branch
  exposed a mapper boundary defect. The mapper advanced an untouched wire past a
  unary conditional and then tried to extend the conditional with a value from
  the wrong side of the boundary. The final mapper recognizes unary structured
  control as a boundary and uses a cold routing preview to decide whether nested
  routing needs a larger physical-wire workspace.
- Observation: the former mapping reorder used only SSA dependencies. A
  classical register store and a later load are ordered through memory effects,
  not an SSA edge, so the reorder could move a register condition before the
  measurement store that initialized it. A three-bit sparse conditional
  regression reproduces this behavior.
- Observation: Qiskit 2.5 provides C functions that inspect structured control
  and classical expressions, but provides no C constructors for them. The public
  Python API does provide `IfElseOp`, `ForLoopOp`, `WhileLoopOp`,
  `SwitchCaseOp`, classical-expression constructors, and `CircuitInstruction`. A
  complete direct exporter therefore needs version-specific Python
  post-processing unless Qiskit adds C constructors.
- Observation: Qiskit's direct classical-expression representation supports
  `Uint` values up to 64 bits. It cannot represent the 151- and 301-bit
  Benchpress comparisons directly. Qiskit's OpenQASM 3 importer does accept
  direct equality against registers of those widths, so the OpenQASM
  compatibility path remains necessary even after direct structured export.
- Observation: Qiskit rejects generic output-qualified declarations and some
  expanded classical forms emitted by Core. The compatibility exporter must
  reconstruct one direct register comparison and fuse eligible measurement
  storage instead of emitting a recursive Boolean expansion.
- Observation: a classical-register load can become stale before a condition
  uses its computed value. The equality matcher must scan from every captured
  load to the consumer and reject an intervening write, free, or ambiguous
  memory effect on the same storage. Intervening quantum operations and barriers
  are safe when their operation trees do not reference that storage.
- Observation: a measurement whose result is stored only inside a later
  conditional cannot become one Qiskit measurement instruction without changing
  its classical-write semantics. The exporter now requires the static
  destination store in the same block immediately after the measurement, apart
  from constants, and fails before returning a circuit otherwise.
- Observation: a preliminary exact Benchpress run passed 25 cases and failed 16
  cases. The failures were the 15 sparse `cc_n12`, `cc_n32`, `cc_n64`,
  `cc_n151`, and `cc_n301` combinations plus Feynman's `qec`. This result led to
  the structured-control boundary and memory-order fixes; it is not final
  acceptance.
- Observation: the unshared 301-bit condition expanded 603 source conditions
  into 181,503 `scf.if` operations and the same number of loads. CSE and dead
  value removal dominated a 230-second target compile. Snapshot-safe root
  condition sharing reduces the program to 602 condition nodes, a 1.17 MB QCO
  module, a 2.81-second target compile, and a 508 MB measured peak RSS.
- Observation: the first optimized exact run stalled on `shor_n5` although its
  five-qubit input was small. An inactive tensor-carrier iterator was still
  considered by the mapping composite-frontier check, so a ready unary
  conditional was deferred forever. Ignoring inactive iterators closes the
  no-progress loop; the exact reproducer then spends 0.5 milliseconds in mapping
  and 13 milliseconds in target compilation.

## Decision Log

- Decision: default every target to no runtime classical-control support.
  Rationale: omission must fail closed, and a gate list does not imply a
  device's runtime control capabilities. Date/Author: 2026-08-18, Codex.
- Decision: model four independent capabilities: conditional, counted iteration,
  conditional loop, and multiway branch. Rationale: devices commonly support
  measurement feedback without general loops, and each diagnostic must name the
  exact missing capability. Date/Author: 2026-08-18, Codex.
- Decision: expose a canonical enum sequence instead of a public bitmask.
  Rationale: C++ and Python callers can pass ordinary sequences, duplicate
  values normalize deterministically, and an empty sequence has one meaning.
  Date/Author: 2026-08-18, Codex.
- Decision: run capability verification before cleanup, mapping, synthesis, or
  conformance. Rationale: rejection must not partially mutate the input or rely
  on a later pass to erase an unsupported construct. Date/Author: 2026-08-18,
  Codex.
- Decision: inspect only the reachable branch of a constant `if` or
  `index_switch`. Rationale: compile-time control does not require a runtime
  target capability, but all operations in the selected region still require
  validation. Date/Author: 2026-08-18, Codex.
- Decision: scalarize only the constant tensor elements accessed by either
  branch of a supported static rank-one QTensor `qco.if`. Treat every structured
  operation as a mapping boundary, including unary operations. Before routing a
  composite, run a cold preview: if the region needs no swaps, keep the original
  sparse wire set; otherwise add the complete target workspace. Rationale: this
  preserves sequencing and transient routing needs without creating a
  qubit-count product in every conditional. Date/Author: 2026-08-18, Codex.
- Decision: restore SSA dominance with a stable topological order containing
  both use-def edges and order edges between consecutive operations that access
  memref state. Rationale: the mapper may move newly inserted quantum
  operations, but a condition must not cross the measurement store that
  populated its register. A dependency cycle is diagnosed instead of silently
  changing semantics. Date/Author: 2026-08-18, Codex.
- Decision: directly construct Qiskit structured control through public Python
  classes in the Qiskit 2.5 adapter after native block finalization. Rationale:
  the user explicitly requested this complete solution to close #2071, while
  Qiskit 2.5 has no corresponding C constructors. The generic collector and
  writer interface stay frontend-neutral. The version-specific adapter creates
  all public Python control-flow objects only after native circuit and block
  finalization. Tests cover Qiskit 2.5.0 and 2.5.2. This decision supersedes the
  earlier plan to retain a native rejection and wait for new Qiskit C APIs.
  Date/Author: 2026-08-18, Codex.
- Decision: record pending control-flow insertion points and finalize nested
  writers before mutating the returned top-level circuit data. Rationale: this
  preserves instruction order, supports recursion, and prevents exposure of a
  partially returned circuit on failure. Date/Author: 2026-08-18, Codex.
- Decision: preserve one Qiskit `Parameter` identity for a live lexical
  `scf.for` induction value across all nested blocks. Rationale: equal display
  names do not guarantee Qiskit parameter identity, and independent block
  finalization creates distinct objects without canonicalization. Date/Author:
  2026-08-18, Codex.
- Decision: reject a measurement destination store unless it follows the
  measurement in the same block, with only constants between them. Rationale:
  moving a conditional or delayed write into Qiskit's unconditional measurement
  instruction changes the program. Fail-closed validation preserves the source
  and exposes no partial output. Date/Author: 2026-08-18, Codex.
- Decision: retain a narrow OpenQASM compatibility path for complete
  frontend-produced bit-register equality. Rationale: Qiskit expressions stop at
  64 bits, while Benchpress needs direct 151- and 301-bit comparisons. The
  matcher rejects partial, duplicate, mixed, dynamic, modified, or stale
  expressions rather than claiming generic arbitrary-width expression support.
  Date/Author: 2026-08-18, Codex.
- Decision: share only pure literal, static-bit, not, and/or root conditions
  within one classical-memory snapshot. Clear the cache at region boundaries and
  after every classical mutation. Rationale: the wide Benchpress programs repeat
  an identical register condition hundreds of times; sharing removes quadratic
  IR growth without reusing a stale value or a value defined in a nested region.
  Date/Author: 2026-08-18, Codex.

## Outcomes & Retrospective

The implementation now has the three required Core layers. `CompilerTarget`
declares runtime classical control and a first-pass verifier enforces it.
Supported `qco.if` tensor programs expose the accessed scalar wires without
losing their global quantum sequence, and routing preserves classical register
access order. Direct Qiskit export recursively collects structured control and
constructs it through the Qiskit 2.5 public API. The OpenQASM exporter retains a
checked arbitrary-width route for the Benchpress cases outside Qiskit's 64-bit
limit.

The earlier 14-source all-to-all corpus proved the narrow OpenQASM form. The 136
Qiskit translation tests and 40 focused Benchpress integration tests pass. The
Qiskit adapter passes with Qiskit 2.5.0 and 2.5.2. The exact 41-case
target/topology run now passes in full, including all four 301-qubit cases. The
final rebased Core tree passes all 4,311 discovered CTests, documentation,
stubs, and repository-configured lint. The implementation closes the Core and
Benchpress acceptance surfaces for #2071 and #2131.

## Context and Orientation

`CompilerTarget` is the immutable target description passed to target
compilation. Its public C++ API is in `mlir/include/mlir/Compiler/Target.h`, its
validated storage is in `mlir/lib/Compiler/Target.cpp`, and its Python binding
is in `bindings/mlir/register_mlir.cpp`. The generated Python declarations are
in `python/mqt/core/mlir.pyi`.

`populateTargetCompilationPipeline` in `mlir/lib/Compiler/TargetCompilation.cpp`
assembles the target pass sequence. A preflight is a read-only verifier at the
start of that sequence. Mapping assigns logical qubits to physical target sites.
Native synthesis rewrites gates to the target operation set, and conformance
verifies the final result.

QC is Core's reference-based circuit dialect. QCO is the value-based dialect
used for optimization and mapping. A `qco.if` carries quantum SSA values through
two regions. A QTensor is a tensor whose elements are linear qubit values. The
OpenQASM 2 frontend represents a register conditional with classical `scf.if`
operations that compute the comparison and a quantum `qco.if` that performs the
conditional gates.

`mlir/lib/Dialect/QCO/IR/SCF/IfOp.cpp` owns `qco.if` canonicalization. The
tensor pattern verifies a narrow complete constant-index extract/insert
lifecycle, extracts the union of accessed elements before the conditional,
threads them through both regions, forwards the untouched tensor, and inserts
the scalar results afterwards.
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` owns placement, routing,
and the final stable topological reorder. The reorder follows nested SSA uses
and preserves the order of top-level operations that contain memref accesses.

Qiskit translation is split across `bindings/mlir/qiskit/QiskitImport.cpp`,
`bindings/mlir/qiskit/QiskitExport.cpp`, the frontend-neutral types in
`bindings/mlir/qiskit/QiskitTranslation.h`, and the supported minor-version
adapter in `bindings/mlir/qiskit/Qiskit2_5.cpp`. The generic exporter collects
the full structure before construction. Flat operations use Qiskit's C API. The
version adapter records nested control flow, finalizes each block, rebases the
block onto its captured root bits, constructs the public Python operation, and
inserts a `CircuitInstruction` at the original position.

The OpenQASM compatibility logic lives in
`mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp`. It recognizes the
complete comparison tree created by the OpenQASM 2 frontend and emits one
register equality with an arbitrary-precision constant. The OpenQASM frontend in
`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` accepts that form. This path is
needed when Benchpress falls back from direct Qiskit export for a register wider
than 64 bits.

## Plan of Work

First, keep the target model fail-closed. Validate and canonicalize the enum
sequence in `Target.cpp`, expose the same names in Python, and place the
read-only verifier first in `TargetCompilation.cpp`. Test each capability,
constant reachability, nested operations, dynamic qubit indices, unknown region
control, and unsupported qubit-tensor state.

Second, preserve conditional sequencing before routing. In `IfOp.cpp`, analyze
each static rank-one tensor's branch lifecycle. If both branches use only
distinct constant extracts followed by matching inserts and yield the rebuilt
tensor, replace the accessed union with scalar elements and forward the
untouched tensor. In `Mapping.cpp`, treat unary structured control as a boundary
and add the complete target workspace only when a cold routing preview inserts
swaps. Restore dominance with the stable dependency order described above and
fail if memory-order and SSA edges form a cycle. Test both a small three-bit
register condition and a real sparse OpenQASM-derived shape.

Third, complete direct Qiskit structured export. In `QiskitExport.cpp`, collect
nested result-free `scf.if` and `scf.index_switch`, state-free `scf.while`, and
constant-range `scf.for` without loop-carried values. Normalize bit, register,
Boolean, unsigned-integer, and floating-point conditions. Reject stale memory
snapshots, invalid widths, signed interpretations, unsupported SSA results,
resource allocation inside blocks, and nesting beyond 64 levels. Preserve the
existing rejection of free symbolic parameters and runtime inputs.

In `Qiskit2_5.cpp`, keep C construction for flat gates and resources. Store
pending structured operations with their insertion index, block writers,
captures, target, loop metadata, and switch labels. On `finish()`, finalize
blocks recursively, canonicalize same-identity loop parameters, construct the
public Qiskit operations, and insert complete instructions in source order. Keep
all Python-specific work inside this version adapter.

Reject measurement export when its classical destination store is in another
block or is delayed by a non-constant operation. This validation must happen
before a partial output can escape and must leave the QC source unchanged.

Fourth, preserve the arbitrary-width Benchpress route. Match only an unchanged
complete register equality. Fuse a measurement result directly into the output
register when no intervening access changes its meaning. Emit one OpenQASM
comparison, and reject every shape that the matcher cannot prove safe.

Share repeated pure root conditions only while the classical memory snapshot is
unchanged. Never cache recursive short-circuit operands, which can be defined in
a nested region. Invalidate and isolate the cache at every classical mutation
and structured region boundary.

Finally, rebase both repositories onto current `main`. Run focused tests first,
then every affected Core and Benchpress suite. Record exact counts and commands
only after they finish. Do not remove or claim the 41 skips until all exact node
IDs pass their basis and coupling validation. Add the changelog entry only when
the pull-request number is known.

## Milestones

### Declare and preflight target capabilities

This milestone is complete when C++ and Python callers can declare any subset of
the four capabilities and read back a canonical immutable sequence. A target
with no declaration accepts straight-line programs and rejects runtime control.
The first target pass must diagnose the missing capability before a sentinel
later pass can run. Constant `if` and `index_switch` operations inspect only
their selected region.

### Preserve conditional quantum and classical sequencing

This milestone is complete when a frontend-produced register conditional with a
static rank-one QTensor compiles through cleanup, sparse mapping, native
synthesis, and conformance. Untouched tensor elements must remain ordered across
the branch. The exported classical condition must still occur after its
measurement store. Unsupported dynamic tensor forms must fail in preflight.

### Export structured control directly to Qiskit 2.5

This milestone is complete when nested `if`/`else`, `for`, `while`, and `switch`
circuits round-trip through QC and back to Qiskit without an OpenQASM
intermediate. Instruction order, bit captures, explicit empty branches, register
identity, switch defaults, descending imported ranges, and lexical loop
parameter identity must survive. Invalid structures must fail without modifying
the source program or returning a partial output.

### Retain arbitrary-width OpenQASM compatibility

This milestone is complete when Core imports, target-compiles, exports, and
reimports representative 1-, 64-, 151-, and 301-bit register conditions. A stale
register snapshot must be rejected. Eligible measurement stores must emit as
direct assignments, and Qiskit must parse the sanitized output used by
Benchpress.

### Validate all 41 Benchpress cases

This milestone remains in progress. Every formerly skipped node ID must execute
with its original target and topology and pass the existing basis and coupling
checks. The final result must report 41 passes, not only the 14 unique source
programs or the smaller integration suite.

## Concrete Steps

Run commands from the repository root. Configure and build the release tree:

    cmake --preset release
    cmake --build build/release --parallel 8

Build and run the focused Core test binaries while iterating:

    cmake --build build/release --parallel 8 --target \
      mqt-core-mlir-unittests-compiler \
      mqt-core-mlir-unittest-qco-ir \
      mqt-core-mlir-unittest-qc-translation \
      mqt-core-mlir-unittest-openqasm-target
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    build/release/mlir/unittests/Dialect/QCO/IR/\
      mqt-core-mlir-unittest-qco-ir
    build/release/mlir/unittests/Dialect/QC/Translation/\
      mqt-core-mlir-unittest-qc-translation
    build/release/mlir/unittests/Target/OpenQASM/\
      mqt-core-mlir-unittest-openqasm-target

Build the Python binding and run the translation tests in the repository's
configured Python environment:

    cmake --build build/python/MinSizeRel --parallel 8 \
      --target mqt-core-mlir-bindings
    uv run --no-sync pytest test/python/test_mlir.py \
      test/python/test_mlir_qiskit_translation.py
    uvx nox -s stubs

After the focused runs pass, run the complete configured C++ suite and project
checks:

    ctest --preset release --output-on-failure
    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check

Run the Benchpress integration suite and the exact 41 node IDs from the
Benchpress repository with its documented pytest environment. Preserve the
node-ID list in the companion Benchpress change so a retry executes the same
target/topology matrix.

## Validation and Acceptance

The target API is accepted when construction rejects invalid enum values,
deduplicates valid values, and preserves canonical ordering across copies. The
default must remain empty. Diagnostics must distinguish `conditional`,
`iteration`, `conditional-loop`, and `multiway-branch` requirements.

The mapper is accepted when a sparse register-conditional program preserves
measurement-before-condition order and passes MLIR verification after routing. A
supported static rank-one QTensor must expose each accessed element through
`qco.if`; untouched elements remain in the forwarded tensor. A routing region
that needs transient intermediate wires must receive the complete workspace.
Dynamic indices, dynamic shapes, and other unsupported tensor state must fail
before mapping.

Qiskit export is accepted when the nested structured-control tests round-trip
through Qiskit's public objects and can be imported into Core again. Conditions
must preserve root bit/register identity. Loop parameters must have one identity
inside their lexical body and remain distinct across nested loops. A controlled
dense unitary inside control flow must remain in the correct instruction
position. Flat numeric gate and global-phase behavior must remain green after
the structured-control merge. Run the same suite against the supported Qiskit
range boundaries, including 2.5.0 and 2.5.2. A conditional or delayed
measurement store must fail without changing its source QC program.

OpenQASM compatibility is accepted when complete arbitrary-width equality
round-trips, while partial or stale equality fails closed. The 301-bit source
must emit one direct register comparison per source condition rather than a
recursive Boolean expression.

Final acceptance requires every one of the 41 formerly skipped Benchpress node
IDs to pass target basis and coupling validation. The final pre-rebase run
passes all 41 after the sparse tensor, mapping-boundary, memory-order, condition
sharing, and inactive-frontier fixes.

## Idempotence and Recovery

All build, test, stub, documentation, and lint commands are repeatable. CMake
reconfiguration is safe after a rebase. Keep all feature work in its isolated
Core or Benchpress worktree and preserve unrelated user changes.

The Qiskit exporter collects and validates the generic structure before it
creates the returned circuit. If version-specific construction fails, discard
the unreturned local objects and rerun after the fix; never expose or reuse a
partial circuit. The mapper diagnoses a dependency cycle instead of applying an
unsafe reorder. Keep any input that triggers that diagnostic as a regression
before changing the dependency model.

Do not use destructive Git reset or checkout commands. Resolve rebase conflicts
by retaining the current `origin/main` behavior and adding only the scoped
classical-control changes. Do not pull unrelated feature branches into this
stack.

## Artifacts and Notes

The final rebased Core snapshot reports 4,311/4,311 CTest cases passing, with
one expected skipped test. Its focused pre-rebase feature suites included 492
QCO IR, 151 compiler, 175 QC translation, 167 OpenQASM, and 87 mapping tests;
the complete post-rebase Qiskit translation file contains 136 tests.

The narrow OpenQASM corpus previously produced:

    14/14 unique source programs passed
    cc_n301: 301 qubits, 603 emitted if statements

The focused Benchpress integration suite currently produces:

    40 passed

The direct structured-control adapter was tested with Qiskit 2.5.0 and 2.5.2.
The complete post-rebase Qiskit translation file produces:

    136 passed

The corrected exact target/topology run produced:

    41 passed in 171.03 seconds
    cc_n301 square: 1.8855 seconds
    cc_n301 heavy-hex: 2.4440 seconds
    cc_n301 all-to-all: 2.8405 seconds
    cc_n301 linear: 3.6440 seconds

The process used about 3.13 GiB maximum RSS and completed without a timeout,
memory error, or validation failure. The earlier 25/41 result identified the
sparse memory-order and QTensor sequencing defect and is retained only in the
discovery history above.

## Interfaces and Dependencies

The public target API adds
`CompilerTarget::ClassicalControl::{Conditional, Iteration, ConditionalLoop, MultiwayBranch}`
and an immutable capability sequence. The Python binding exposes the same enum
under `CompilerTarget.ClassicalControl` with uppercase member names generated by
nanobind.

The target preflight depends only on MLIR operation interfaces, QCO, QTensor,
and SCF. It does not inspect Qiskit objects or infer runtime support from native
gates. The mapping reorder depends on SSA use-def relations and memref-typed
operands or results, including those in nested operations.

The generic Qiskit translation interface contains numeric or lexically bound
parameters, classical targets and expressions, loop metadata, switch cases,
recursively owned block writers, and bit-index captures. It contains no Python
control-flow object. `Qiskit2_5.cpp` is the only component that depends on the
public Python constructors needed because the Qiskit 2.5 C API lacks them.

Direct Qiskit classical expressions use Boolean, floating-point, and unsigned
integer values up to 64 bits. The OpenQASM exporter and frontend use
arbitrary-precision integers for the narrow full-register equality form. The two
paths are complementary: direct Qiskit export closes #2071 for its supported
structured-control contract, while OpenQASM retains the wider Benchpress cases
required to close #2131's integration gaps.
