# Declare and verify target classical-control capabilities

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Compiler targets currently describe qubit sites, topology, native operations,
and timing, but they do not state which runtime classical-control forms the
hardware accepts. A program can therefore enter mapping before Core discovers
that the target cannot run a conditional or loop. This change gives C++ and
Python users an explicit, opt-in target capability list and adds a preflight
check at the start of target compilation. A target that does not declare a
required capability rejects the program before any mapping mutation.

The observable result is that `CompilerTarget(2)` still accepts straight-line
programs but rejects a dynamic `qco.if`. Constructing the same target with
`ClassicalControl::Conditional` in C++ or
`CompilerTarget.ClassicalControl.CONDITIONAL` in Python permits the conditional
to proceed to the existing compilation pipeline. The check also fails closed for
unsupported control-flow interfaces, dynamic qubit indices, captured quantum
state, and qubit tensors carried through structured control.

## Progress

- [x] (2026-08-19 14:15Z) Port the target capability API and Python binding onto
  the first-class classical-register base.
- [x] (2026-08-19 14:15Z) Add the fail-closed preflight and static
  `qco.index_switch` canonicalization.
- [x] (2026-08-19 14:15Z) Remove tensor scalarization, mapping, OpenQASM, and
  Qiskit behavior from this task.
- [x] (2026-08-19 14:15Z) Add focused C++, MLIR, and Python tests and update the
  user documentation, upgrade guide, and changelog.
- [x] (2026-08-19 14:22Z) Regenerate the Python type stubs with
  `uvx nox -s stubs`; the session reproduced the intended `mlir.pyi` API and
  changed no unrelated stub.
- [x] (2026-08-19 14:23Z) Build the compiler and QCO IR test targets. Pass 21
      focused compiler tests, including final conformance with all four
      capabilities, one focused canonicalization test, all 146 compiler tests,
      all 487 QCO IR tests, and all 49 Python MLIR tests.
- [x] (2026-08-19 14:23Z) Pass `uvx nox -s lint`, focused `rumdl` and `ruff`
  checks, and `git diff HEAD --check`.
- [x] (2026-08-19 19:47Z) Rebase the focused change onto `main` after #2158,
  resolve the changelog-link conflict without dropping either entry, rebuild
  both C++ test targets and the Python bindings, regenerate unchanged stubs,
  and pass the 21 focused compiler tests, one focused QCO IR test, all 146
  compiler tests, all 487 QCO IR tests, and the focused Python test.
- [x] (2026-08-19 19:47Z) Pass the final post-rebase Markdown and full
  repository lint checks and `git diff --check`.
- [x] (2026-08-21 15:25Z) Merge the latest `origin/main`, audit its entry-point,
      CBit metadata, mapping, and verification-helper changes, and migrate all
      20 branch-added MLIR fixtures from the legacy passthrough marker to
      `mqt.entry_point`.
- [x] (2026-08-21 15:25Z) Regenerate unchanged Python stubs and pass 24 focused
      compiler tests, all 149 compiler tests, the focused and all 487 QCO IR
      tests, 12 structured-control mapping regressions, and the focused Python
      capability test.
- [x] (2026-08-21 15:26Z) Pass the final post-merge Markdown and full repository
      lint checks and `git diff --check`.
- [x] (2026-08-21 15:42Z) Reproduce three review findings: foldable static
      selectors are treated as dynamic, nested control causes repeated subtree
      scans, and QDMI target snapshots discard QIR Adaptive control support.
- [x] (2026-08-21 16:11Z) Replace repeated target-control subtree walks with one
      cached read-only analysis that understands foldable selectors and dead
      counted loops; pass all 154 compiler and 20 MQT utility tests.
- [x] (2026-08-21 16:11Z) Derive conditional support from QDMI QIR Adaptive
      formats, add explicit optional-capability augmentation, and pass the C++
      and Python coverage, including all 50 Python MLIR tests.
- [x] (2026-08-21 16:11Z) State the capability model's boundary for classical
      operations that do not direct quantum execution and pass the final full
      repository lint and diff checks.
- [x] (2026-08-21 16:55Z) Preserve compatibility with QDMI devices that omit the
      optional supported-program-formats property, document explicit additions
      as trusted caller assertions, and pass all 234 QDMI, 154 compiler, and 50
      Python MLIR tests, generated-stub checks, full repository lint, and the
      final diff check.

## Surprises & Discoveries

- Observation: A constant `qco.index_switch` had no canonicalizer, although the
  preflight deliberately ignores unreachable regions of static control.
  Evidence: `qco.if` already folds through its canonicalization patterns, while
  `IndexSwitchOp` only implemented verification and control-flow interfaces.
- Observation: Static rank-one qubit tensors inside `qco.if` need a separate
  scalarization and mapping change. Evidence: the target preflight can classify
  the control capability, but the current mapper does not define the required
  tensor routing contract. This task therefore rejects every qubit-tensor
  operand or result on `qco.if`.
- Observation: The stub session deliberately builds without the superconducting
  QDMI device, so two existing full-file Python tests need the separately built
  release device and its generated registry fragment. Evidence: the first run
  passed 47 tests and reported two unknown `mqt.sc.iqm.garnet` fixture errors;
  the rerun with the release fragment and library passed all 49 tests.
- Observation: Main now resolves programs through `mqt::getEntryPoint`, which
  does not recognize the former `passthrough = ["entry_point"]` fixture marker.
  Evidence: the branch added 20 such fixtures; converting them to
  `mqt.entry_point` keeps them aligned with the current public entry-point
  convention without changing the production implementation.
- Observation: A `qco.if` selected by an `arith.cmpi` of integer constants is
  rejected before cleanup but succeeds when the caller invokes cleanup first.
  Evidence: direct target compilation requests `Conditional`; cleanup folds the
  comparison and removes the control operation before target compilation.
- Observation: The verifier scans each nested control operation's full subtree.
  Evidence: at nesting depth 800, the verifier took 29.5 ms of a 39.4 ms release
  compile, while 1,600 sibling conditionals compiled in about 2.8 ms.
- Observation: The bundled DDSIM device advertises both QIR Adaptive formats,
  but `CompilerTarget.from_device` returns no classical-control capabilities.
  Evidence: `snapshotCompilerTarget` does not query supported formats and calls
  the target factory without a capability list.
- Observation: MLIR permits `Operation::fold` to modify the folded operation in
  place even when the caller only intends to inspect a constant expression.
  Evidence: the first fold-aware preflight called the shared constant-folding
  helper on source operations, which could violate the preflight's no-mutation
  contract; the helper now folds a clone of each pure operation.
- Observation: QDMI highly encourages but does not require devices to expose
  their supported program formats. Evidence: the non-optional C++ getter threw
  on `QDMI_ERROR_NOTSUPPORTED`, so querying it unconditionally from the adapter
  could make a previously valid third-party target fail to construct.

## Decision Log

- Decision: Make all classical-control capabilities opt-in and independent.
  Rationale: A target that supports a conditional does not necessarily support
  counted loops, condition-terminated loops, or multiway branches. An empty list
  preserves a clear fail-closed contract. Date/Author: 2026-08-19, Codex.
- Decision: Verify capabilities before cleanup, mapping, synthesis, and
  conformance passes. Rationale: The compilation API promises that a rejected
  program remains unchanged, and an early pass gives a precise diagnostic at the
  source operation. Date/Author: 2026-08-19, Codex.
- Decision: Follow only the selected region of `if` and `index_switch` when the
  selector is constant. Rationale: Unreachable control does not require a
  runtime target capability. The cleanup pipeline then removes the static
  operation. Date/Author: 2026-08-19, Codex.
- Decision: Reject all qubit tensors carried through structured control in this
  task, including static rank-one tensors on `qco.if`. Rationale: Accepting them
  requires tensor scalarization and control-aware mapping. Those changes form a
  separate review unit. Date/Author: 2026-08-19, Codex.
- Decision: Keep the static `qco.index_switch` canonicalizer in this task.
  Rationale: The preflight's reachable-region rule must have a corresponding
  cleanup path so a constant switch cannot reach mapping. Date/Author:
  2026-08-19, Codex.
- Decision: Infer only `ClassicalControl::Conditional` from QIR Adaptive QDMI
  formats. Rationale: Forward branching is mandatory in that profile, while
  backward and multiway branching are optional. Date/Author: 2026-08-21, Codex.
- Decision: Add caller-supplied QDMI capability augmentation as a union with the
  inferred list. Rationale: QDMI has no standard properties for the optional
  loop and multiway flags, and callers must not reconstruct device topology and
  calibration data to add them. Date/Author: 2026-08-21, Codex.
- Decision: Treat an unsupported QDMI program-format property as an empty list
  and describe explicit additions as trusted caller assertions. Rationale:
  missing optional metadata cannot justify an inferred capability and must not
  make otherwise usable devices fail. Date/Author: 2026-08-21, Codex.
- Decision: Keep program requirements separate from target support in the
  preflight implementation. Rationale: One read-only analysis can classify the
  program, avoid repeated subtree walks, and preserve precise diagnostics when
  the target comparison runs. Date/Author: 2026-08-21, Codex.
- Decision: Define the current capabilities as structured control that can
  change quantum execution. Rationale: QIR classical types, functions, return
  points, and assertion behavior need a broader target model than this task.
  Date/Author: 2026-08-21, Codex.
- Decision: Fold cloned pure operations when evaluating compile-time constants.
  Rationale: `Operation::fold` may mutate its receiver, while target preflight
  must remain read-only even when compilation ultimately rejects the program.
  Date/Author: 2026-08-21, Codex.

## Outcomes & Retrospective

Implementation and validation are complete. The resulting scope is the target
capability contract, the early verifier, static switch cleanup, public bindings,
and their direct tests. The release build passed all 146 compiler and 487 QCO IR
tests. The generated Python extension passed all 49 tests in
`test/python/test_mlir.py`, and the complete repository lint session passed.
Tensor scalarization, mapping through runtime control, register-condition
export, and Qiskit structured export remain outside this task and can build on
this API without changing its contract.

The post-#2158 rebase preserved the implementation without a semantic conflict.
The rebuilt focused and complete C++ suites passed with the same counts, the
focused Python capability test passed, and stub generation produced no diff.

The latest-main merge also preserved the complete production scope. Main's
entry-point migration required only a mechanical update of the 20 branch-added
MLIR fixtures; its CBit metadata, mapping, and verification-helper changes do
not otherwise intersect this feature. The post-merge release build passed 24
focused and all 149 compiler tests, the focused and all 487 QCO IR tests, 12
relevant structured-control mapping regressions, and the focused Python
capability test. Stub generation again produced no diff.

The review follow-up is complete. The target preflight now analyzes each
operation and reachable operand expression once, caches constant folding, skips
dead counted loops, and preserves diagnostic order without repeated descendant
walks. At nesting depth 800, the verifier fell from 29.5 ms of a 39.4 ms compile
to 3.2 ms of a 12.9 ms compile. Constant evaluation folds cloned operations, so
the read-only preflight contract is preserved.

QDMI snapshots infer only forward conditional support from QIR Adaptive string
or module formats. Callers can augment that conservative inference with optional
loop or multiway capabilities through the C++ and Python factories; the target
constructor canonicalizes the union. Devices that omit the optional format
property infer nothing, and explicit additions are documented as unchecked
caller assertions. The documentation limits the four flags to structured control
that directs quantum execution and does not claim general classical-computation,
assertion, function, or QIR-profile support.

The final release binaries passed all 154 compiler and 20 MQT utility tests. The
generated Python extension passed all 50 tests in `test/python/test_mlir.py`;
stub generation, full repository lint, and `git diff --check` also passed. Per
task direction, no documentation build was required.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` defines the immutable
`mlir::CompilerTarget` public API. `mlir/lib/Compiler/Target.cpp` validates
input and stores canonical target data. This task adds a `ClassicalControl` enum
with four values: forward conditional branching, counted iteration,
condition-terminated iteration, and multiway branching. The target stores a
sorted, duplicate-free list and exposes both the list and a support query.

`mlir/lib/Compiler/TargetCompilation.cpp` constructs the standard pass pipeline
used by `QCOProgram::compileForTarget`. The first new pass walks reachable
regions and maps `qco.if` and `scf.if` to conditional support, `scf.for` to
iteration support, `scf.while` to conditional-loop support, and
`qco.index_switch` and `scf.index_switch` to multiway-branch support. A static
selector means that only one region is reachable. Any other operation that
implements MLIR's branch interfaces fails closed because Core has not assigned
it a target capability.

The preflight also rejects control forms that the later pipeline cannot safely
lower: a dynamic `qtensor.extract` or `qtensor.insert` index, quantum state
captured from outside a structured region, quantum state nested in a generic SCF
conditional or switch, and a qubit tensor passed into or returned from any
supported structured control operation.
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` and
`mlir/lib/Dialect/QCO/IR/SCF/IndexSwitchOp.cpp` add the missing constant
selector canonicalization for `qco.index_switch`.

`bindings/mlir/register_mlir.cpp` publishes the enum, constructor argument,
property, and support query to Python. `bindings/patterns.txt` supplies overload
signatures to the stub generator. `python/mqt/core/mlir.pyi` is generated output
and must only change through the repository stub-generation session.

The direct tests live in `mlir/unittests/Compiler/test_compiler_target.cpp`,
`mlir/unittests/Compiler/test_compiler_pipeline.cpp`,
`mlir/unittests/Dialect/QCO/IR/test_qco_ir.cpp`, and `test/python/test_mlir.py`.
User-facing behavior is documented in `docs/mlir/target_compilation.md`,
`UPGRADING.md`, and `CHANGELOG.md`.

This task must not modify mapping, QCO tensor scalarization, OpenQASM import or
export, or Qiskit translation. It must preserve the first-class classical-bit IR
and tests already present on the base. Follow `AGENTS.md` and
`docs/ai_usage.md`. Do not perform a GitHub action without separate human
authorization.

## Plan of Work

Extend `CompilerTarget` with the four-value enum and overload each existing
factory so source-compatible callers still use the old defaults. Store the
capabilities in immutable target storage. Validate enum values, sort the list,
remove duplicates, and provide read-only queries. Bind the same contract in
Python with an empty sequence default, then regenerate the type stub.

Add a module pass at the start of target compilation. Walk regions explicitly so
constant conditionals and switches inspect only their selected region. Emit
diagnostics that name either the missing target capability or the unsupported
control construct. Reject unsafe quantum captures, tensor state, and dynamic
qubit indexing before invoking any mutating pass. Do not use or introduce the
tensor-scalarization helper; every qubit-tensor input or result on `qco.if` must
fail at this stage.

Implement the preflight as one read-only program analysis followed by a top-down
target comparison. Cache folded integer values so selectors made from constant
expressions are static without mutating the module. Treat a counted loop with a
provably empty iteration range as dead. Compute quantum-state and capture
summaries once so nested control does not rescan descendant operations. Keep the
existing reachability, diagnostic order, and fail-closed behavior.

When adapting a QDMI device, inspect its supported program formats once. Add
`Conditional` for QIR Adaptive string or module format. Do not infer optional
loops or multiway branching from QASM 3, measurements, or provider-specific
operation names. Let C++ and Python callers supply extra capabilities, merge
them with the inferred conditional, and rely on target construction to sort and
remove duplicates.

Add an `IndexSwitchOp` canonicalization pattern that matches an integer constant
selector, inlines the matching case or default block, replaces all classical and
quantum results with the selected yield values, and erases the yield. Test both
a selected case and the default with mixed result types and reordered linear
values.

Test target construction through all C++ factory forms and the Python
constructor. Test canonical ordering, duplicate removal, support queries, the
empty default, and invalid C++ enum input. In pipeline tests, prove that missing
capabilities reject without mutating the module, independent capabilities stay
independent, unreachable static regions are ignored, unknown branch interfaces
fail closed, quantum capture and generic SCF quantum state fail, dynamic tensor
indices fail, and every structured-control qubit tensor fails even when the
target declares the control capability.

Update the target compilation guide and upgrade guide with the opt-in contract,
examples, capability mapping, reachable-region behavior, and explicit stage-one
limitations. Keep the changelog entry limited to the target API and preflight.

## Concrete Steps

Run all commands from the repository root. Configure and build the standard
release tree:

    cmake --preset release
    cmake --build --preset release --parallel 8

Run the focused C++ tests:

    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerTargetTest.*:CompilerPipelineTest.TargetCompilation*'
    ./build/release/mlir/unittests/Dialect/QCO/IR/mqt-core-mlir-unittest-qco-ir \
      --gtest_filter='QCOTest.CanonicalizesConstantIndexSwitchToSelectedCaseOrDefault'

If the QCO test binary has a different generated basename, list
`build/release/mlir/unittests/Dialect/QCO/IR` and run the binary that contains
`qco-ir`. Then run the full two binaries without filters.

Regenerate the stub and run the Python test:

    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_mlir.py \
      -k compiler_target_classical_control_is_explicit_and_canonical

The stub session must leave `python/mqt/core/mlir.pyi` equal to the generated
binding API. Finish with documentation and repository checks:

    uvx rumdl check CHANGELOG.md UPGRADING.md docs/mlir/target_compilation.md \
      .agent/plans/classical-control-capabilities.md
    uvx nox -s lint
    git diff --check

## Validation and Acceptance

The C++ target tests must show that all factory overloads store the same sorted,
duplicate-free list and reject an unknown enum value. The Python test must show
the same ordering and empty default. The pipeline tests must show that a missing
capability fails before mutation and includes the capability and operation name
in the diagnostic. A declared capability must not imply another capability.

The preflight must reject `qco.if` with a static rank-one qubit tensor as well
as dynamic or non-rank-one tensors. It must reject unsupported MLIR branch
interfaces and dynamic qubit indices. Constant `qco.if` and `qco.index_switch`
operations must inspect only the selected region, and cleanup must remove the
static switch while preserving its mixed results.

A conditional selected by a folded integer comparison and a provably empty
counted loop must compile for a target with no runtime control. A
measurement-dependent conditional must still fail. A QDMI device that advertises
QIR Adaptive must imply only `Conditional`; QASM support must not imply control.
Caller-supplied optional capabilities must be merged, and direct device and
device-ID construction must agree.

The focused tests, full affected C++ binaries, Python test, generated-stub
check, Markdown lint, full repository lint, and `git diff --check` must pass.
Record exact commands and counts in `Progress` and `Outcomes & Retrospective`.

## Idempotence and Recovery

CMake configuration, incremental builds, tests, stub generation, and lint are
safe to repeat. The target capability list is canonicalized during construction,
so repeated construction cannot accumulate duplicates. If stub generation fails
because the environment lacks a built extension, build the binding target first
and rerun the session; do not edit the generated `.pyi` file by hand.

If a preflight test fails after partially running the pipeline, compare the
program text captured before compilation with the text after failure. The pass
must remain first in `populateTargetCompilationPipeline`; do not compensate by
undoing later mapping changes. Work only in this task's worktree and preserve
all unrelated changes from the base.

## Artifacts and Notes

The final compiler binary reports 154 passing tests, including the fold-aware,
empty-loop, deep-nesting, QDMI-inference, augmentation, device-ID parity, and
end-to-end conditional regressions. The MQT utility binary reports 20 passing
constant-folding tests. All 50 Python MLIR tests pass against the generated
extension, and stub generation reproduces the checked-in interface.

The QDMI client binary reports 234 passing tests, including a registered device
whose supported-program-formats query returns `QDMI_ERROR_NOTSUPPORTED`. Its C++
getter returns an empty vector, allowing compiler-target construction to remain
fail-closed without rejecting the device solely for missing metadata.

The depth-800 performance sample reports 3.2 ms in
`VerifyTargetClassicalControlPass` and 12.9 ms total, compared with 29.5 ms and
39.4 ms before the cached analysis. The remaining dominant pass in that sample
is canonicalization, not the capability verifier.

The target preflight diagnostics are part of the observable contract. Tests
check both the rejected operation and the missing capability so later pipeline
changes cannot replace an early, precise failure with a mapping error.

## Interfaces and Dependencies

`mlir::CompilerTarget::ClassicalControl` is a four-value public enum with
`Conditional`, `Iteration`, `ConditionalLoop`, and `MultiwayBranch` values.
`CompilerTarget::classicalControl()` returns the canonical capability list, and
`CompilerTarget::supportsClassicalControl(ClassicalControl)` tests one value.
Existing constructors and factories retain source compatibility through an empty
default list.

The Python binding exposes the enum as `CompilerTarget.ClassicalControl`, the
constructor keyword as `classical_control`, the read-only property as
`classical_control`, and the query as `supports_classical_control`. The binding
depends on nanobind and the generated `python/mqt/core/mlir.pyi` stub must match
these names and signatures.

The verifier depends on MLIR region-branch interfaces and on the QCO, QTensor,
SCF, Arith, and LLVM dialect operations that it classifies. The
`qco.index_switch` canonicalizer remains in the QCO dialect because cleanup must
remove the same unreachable runtime control that preflight ignores.

The QDMI factories accept an optional additional classical-control list. They
infer `Conditional` only from QIR Adaptive string or module support and union
the supplied values with that inference. The additions are trusted caller
assertions. An unsupported program-format property produces an empty list and no
inferred capability. `MQTCompilerPipeline` directly links `MLIRMQTUtils` for
cached constant-expression evaluation.

Plan revision note: Recorded the latest-main merge, fold-aware performance
follow-up, conservative QDMI mapping and augmentation, no-mutation correction,
optional-property compatibility, scope boundary, and final validation evidence.
