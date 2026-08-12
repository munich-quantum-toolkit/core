# Make MQT QIR portable, complete, and safe to execute

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently has a QC-to-QIR compiler, a QIR runtime and JIT runner, and a
DDSIM QDMI device that executes QIR. These components do not share one complete
calling convention. In particular, parameterized gates use an MQT-specific
argument order, controlled gates use names not recognized by other QIR tools,
and the runtime does not implement every call emitted by the compiler. The JIT
and DDSIM device also share process-global simulator state, which makes
concurrent QIR jobs unsafe.

After this work, QIR produced by MQT uses a complete, explicitly documented
quantum instruction set with the common parameter-first convention, uses the
current QIR 2.1 runtime functions where the profile specifies them, and is
rejected with a precise diagnostic if its declarations do not match the
supported ABI. Calls with one or two controls use dedicated QIS functions; calls
with three or more controls use generic `__ctl` and `__ctladj` specializations
with a control Array and the original gate arguments. These functions are an MQT
QIS extension to the Base and Adaptive profiles and do not change the entry
point's `qir_profiles` attribute. The runtime does not emulate qir-runner's
legacy resource-management and output APIs. The standalone runner can select an
entry point and execute multiple deterministic shots. Each DDSIM QIR job owns
its execution state, so concurrent jobs cannot affect one another. A user can
observe the result by compiling the gate-matrix tests, running Base and Adaptive
QIR through the MQT JIT, and submitting concurrent QIR jobs through the DDSIM
QDMI device.

The implementation is delivered as a native GitHub stack with three branches.
The first layer establishes the compiler/runtime contract, the second completes
and isolates the general runner, and the third migrates DDSIM and adds
end-to-end coverage. Every layer must build and pass the tests relevant to its
own diff before the next layer is created.

## Progress

- [x] (2026-08-09 23:41Z) Refreshed `origin/main`, inspected issues 1833 and
      1834, read repository contribution and AI-use policy, installed GitHub's
      official `gh-stack` extension, and initialized the bottom stack branch
      from `origin/main`.
- [x] (2026-08-09 23:41Z) Cross-checked the current QIR compiler, runtime, JIT,
  runner, DDSIM consumer, QIR 2.1 profile documents, and qir-runner ABI.
- [x] (2026-08-10 00:08Z) Implement stack layer 1's shared gate description,
      parameter-first compiler and builder ABI, complete runtime/JIT gate
      parity, focused tests, and breaking-change migration guidance.
- [x] (2026-08-10 00:12Z) Validate stack layer 1 with the release-build QIR,
      compiler, and conversion suites; verify the public header set; pass the
      full repository lint session and `git diff --check`.
- [x] (2026-08-10 01:45Z) Commit and publish stack layer 1 as PR 2034.
- [x] (2026-08-10 02:03Z) Commit and publish stack layer 2 as PR 2035.
- [x] (2026-08-10 02:13Z) Commit and publish stack layer 3 as PR 2036, linked
      through native GitHub stack 2037.
- [x] (2026-08-10 10:25Z) Rebase the complete stack onto `origin/main` at
      `c82b3e77b` before starting the requested design revision.
- [x] (2026-08-10 11:42Z) Revise stack layer 1 to move the shared gate registry
      into the MLIR tree, reserve `r` for the qir-runner Pauli rotation, use
      `prx` for MQT's two-angle rotation, and address its CI annotations.
- [x] (2026-08-10 12:48Z) Close layer 1's fresh C++ patch-coverage gap with
      direct runtime tests for identity, iSWAP, DCX, ECR, XX+YY, and XX-YY;
      Codecov identified exactly these six shared-table rows as uncovered.
- [x] (2026-08-10 14:29Z) Revise layer 1 again so every gate uses native one-
      and two-control entry points, three or more controls use the generic
      controlled specialization, the profile attribute remains Base or Adaptive,
      and the shared callback macro is named `MQT_GATE`. Remove the unnecessary
      conversion-library include-directory changes and validate the
      direct-target and argument-Tuple runtime paths.
- [x] (2026-08-10 16:55Z) Revise stack layer 2 around the exact QIR 2.1 resource
      APIs, opaque-pointer examples, typed declaration validation, exact
      entry-point execution, native one- and two-control functions,
      three-control generic specialization, per-session state, repeated shots,
      deterministic seeding, complete metadata, and exit codes. Validate all
      fixtures with LLVM 22 plus the 69-test runtime, 21-test JIT, 5-test
      runner, and 9-test DDSIM QIR suites.
- [x] (2026-08-10 17:08Z) Revise stack layer 3 around Base-only semantic state
      extraction, eliminate the raw LLVM instruction replacement behind the CI
      analyzer failure, and validate all 25 JIT and 48 DDSIM tests plus both
      focused Python QIR execution tests.
- [x] (2026-08-10) Run the cumulative release build, all 4,622 CTest cases, the
      focused QIR and DDSIM suites, the full repository lint session, diff
      checks, and a fresh audit of each adjacent branch diff. The two
      environment-dependent job-ID tests were skipped by their fixtures.
- [x] (2026-08-10 02:18Z) Publish all three branches with native GitHub stacked
      PRs, add policy-compliant PR descriptions and labels, and verify the
      remote stack, bases, and heads.

## Surprises & Discoveries

- Observation: The QIR specification defines profile structure and runtime
  functions, but deliberately leaves QIS names backend-defined. Evidence: the
  Base and Adaptive profile documents describe calls to backend-defined QIS
  functions, while qir-runner supplies the widely used body, adjoint, and
  controlled function conventions. The compiler therefore needs both a strict
  profile verifier and a separately documented portable QIS policy.
- Observation: `JitSession` locates an `entry_point` function to read metadata,
  but later resolves the literal symbol `main` and invokes it through
  `int(int, char**)`. Evidence: `src/qir/jit/Session.cpp` contains both
  `getEntryPointFunction` and `jit_->lookup("main")`; QIR profile entry points
  are parameterless and return `i64`.
- Observation: the QIR v2.1 and legacy qir-runner allocation APIs reuse
  `__quantum__rt__qubit_allocate` with incompatible LLVM function types:
  `ptr(ptr)` and `ptr()`, respectively. Supporting both therefore adds typed
  adapter and registration complexity without improving current-spec compliance.
  The revised implementation supports the QIR 2.1 form only.
- Observation: `src/qdmi/devices/dd/Device.cpp` obtains `Runtime::getInstance()`
  in asynchronous job execution. The singleton owns quantum state, result
  storage, RNG state, output state, and measurements, so simultaneous QIR jobs
  race even though non-QIR QDMI concurrency tests pass.
- Observation: a conversion-only gate table did not prevent drift because the
  runtime declarations, definitions, and JIT symbols remained separate lists.
  Evidence: after replacing all four lists with `mlir/Conversion/GateTable.def`,
  the release build compiled every gate family and all 58 runtime, 9 JIT, 112
  builder, 267 QC-to-QIR, and 304 QC/QCO conversion tests passed.
- Observation: state extraction cannot be implemented by erasing a fixed list of
  measurement and result symbols. Evidence: that rewrite left arbitrary gates
  after the first measurement executable and depended on spellings rather than
  the Base Profile's required `irreversible` attribute. Truncating the selected
  entry point at the semantic boundary removes the entire terminal
  measurement/output region and supports backend-defined measurement names.

## Decision Log

- Decision: Break the unreleased MQT-specific QIR ABI instead of retaining
  source or binary compatibility. Rationale: compatibility shims would preserve
  ambiguous symbols, incorrect parameter order, and divergent compiler/runtime
  tables. The user explicitly prioritized compliance and broad ecosystem
  compatibility over compatibility with unreleased code. Date/Author: 2026-08-09
  / Codex.
- Decision: Treat qir-spec as the authority for QIR 2.1 profiles and runtime
  functions, and use qir-runner only as a reference for generic controlled
  specializations. Rationale: copying qir-runner's older allocation API into
  emitted Adaptive QIR would make the compiler less standards-compliant.
  Date/Author: 2026-08-09 / Codex.
- Decision: Keep MQT's complete extension QIS for compiler/runtime parity while
  using the conventional parameter-first ABI. Retain only qir-runner's
  Array/Tuple representation for generic controlled specializations because it
  provides a practical ABI for arbitrary controls; remove its legacy resource,
  output, and Pauli-rotation compatibility shims. Rationale: arbitrary controls
  materially improve interoperability, while the other shims complicate a
  runtime whose QIR code is not yet released. Date/Author: 2026-08-10 / Codex.
- Decision: Emit dedicated functions for exactly one and two controls, and use
  `__ctl` or `__ctladj` for three or more. Keep the entry point's declared Base
  or Adaptive profile instead of introducing a `custom` mode. Rationale: the
  common cases remain easy to call, arbitrary controls remain available, and
  backend-defined QIS extensions are independent of the QIR profile.
  Date/Author: 2026-08-10 / Codex.
- Decision: Place the cross-cutting gate registry at
  `mlir/include/mlir/Conversion/GateTable.def` and use `prx` as the QIR symbol
  stem for MQT's two-angle R gate. Reserve `r` for qir-runner's incompatible
  Pauli rotation instead of overloading one name with two signatures.
  Date/Author: 2026-08-10 / Codex.
- Decision: Make runtime state session-owned and use a scoped active-runtime
  binding only as the private C ABI dispatch mechanism; do not expose a public
  `Activation` helper. Rationale: this preserves plain exported C entry points
  while isolating parallel JIT sessions and DDSIM jobs without making dispatch
  mechanics part of the public runtime API. Date/Author: 2026-08-09 / Codex.
- Decision: Derive state extraction from the selected Base Profile entry point's
  `irreversible` boundary and reject every other profile. Rationale: this
  follows the profile's semantic contract, remains compatible with
  backend-defined measurement names, and fails safely rather than changing
  adaptive behavior. Date/Author: 2026-08-10 / Codex.
- Decision: Use GitHub's `gh stack` public-preview feature for publication.
  Rationale: it creates the requested linked stack object and maintains the
  correct adjacent PR bases rather than merely presenting three unrelated PRs.
  Date/Author: 2026-08-09 / Codex.

## Outcomes & Retrospective

The implementation is complete as a three-layer stack. The bottom PR owns the
shared MLIR gate registry and compiler/runtime QIS contract. The middle PR owns
the current QIR 2.1 resource APIs, strict JIT validation, session-local runtime,
and runner usability. The top PR owns DDSIM integration, Base-profile semantic
state extraction, and concurrent-job isolation. The cumulative release build,
all 4,622 CTest cases, the focused QIR and DDSIM suites, repository lint, and
diff checks pass. Two job-ID queries remain fixture-skipped because no external
job service is configured; no implementation was weakened for them.

## Context and Orientation

The compiler path starts in `mlir/lib/Conversion/QCToQIR/`. Common preparation
and gate conversion live in `QIRCommon/QIRCommon.cpp`; profile-specific resource
and control-flow lowering live in `QIRBase/QCToQIRBase.cpp` and
`QIRAdaptive/QCToQIRAdaptive.cpp`. Function names and output helpers are in
`mlir/include/mlir/Dialect/QIR/Utils/QIRUtils.h` and
`mlir/lib/Dialect/QIR/Utils/QIRUtils.cpp`. The public builder in
`mlir/include/mlir/Dialect/QIR/Builder/QIRProgramBuilder.h` must use the same
argument order and names as automatic lowering.

The host runtime C ABI is declared by `include/mqt-core/qir/runtime/QIR.h` and
implemented in `src/qir/runtime/QIR.cpp`. Simulator state and DD operations live
in `include/mqt-core/qir/runtime/Runtime.hpp` and `src/qir/runtime/Runtime.cpp`.
`src/qir/jit/Session.cpp` parses LLVM text or bitcode, binds host functions, and
invokes the entry point. The CLI wrapper is `src/qir/runner/Runner.cpp`.

The DDSIM QDMI device accepts QIR text and bitcode in
`src/qdmi/devices/dd/Device.cpp`. Sampling repeatedly executes the JIT and
collects `Runtime::getMeasurements()`. Statevector retrieval rewrites a Base
Profile module so execution stops before measurement and then takes the DD state
from the runtime. Tests mirror these components under `test/qir/`,
`mlir/unittests/Conversion/QCToQIR/`, and `test/qdmi/devices/dd/`.

A QIR profile constrains LLVM control flow, resource representation, runtime
calls, attributes, and entry-point signature. A QIS is the backend-defined set
of quantum gate functions. This work uses QIR 2.1 for the former and MQT's
explicitly documented QIS for the latter. Its generic controlled specializations
use the same Array/Tuple argument representation as qir-runner.

## Plan of Work

In the first stack layer, define the QIS names and exact LLVM types in one
shared description consumed by QC/QCO conversion, QC-to-QIR lowering, the public
C runtime declarations and definitions, and JIT registration. Change automatic
lowering and the public QIR builder to put floating parameters before qubit
pointers. Remove unreleased aliases that duplicate canonical spellings and
extend the runtime to every compiler-emitted target and control shape. Cover the
calling convention and both profiles with focused tests.

In the second layer, add a verifier that diagnoses wrong entry-point signatures,
profile attributes, runtime declarations, and QIS types before translation or
JIT execution. Replace the runtime singleton contract with a constructible
`Runtime` owned by `JitSession`. Exported C functions find the active runtime
through a private scoped thread-local binding during execution. Refactor JIT
symbol registration to inspect declarations and bind only exact typed
implementations. Add the current QIR 2.1 qubit/result allocation, array, error,
result-array output, double output, metadata, and exit-code contracts. Diagnose
all unresolved declarations together. Resolve arbitrary parameterless `i64`
entry points, expose optional entry-point selection, and add shots and
deterministic RNG seeding to the CLI.

In the third layer, construct a separate JIT session and runtime for every DDSIM
QIR job. Use a job-local sink so device execution does not write interleaved
records to process stdout. Collect counts from the job runtime. Replace the
hard-coded state-extraction symbol list with a Base-profile-aware rewrite that
cuts execution off before the measurement/output region and returns zero. Retain
explicit rejection of Adaptive state extraction. Add simultaneous QIR job tests,
serial Base and Adaptive sampling tests, and Base statevector tests.

After every layer, inspect the adjacent diff, run its focused test binaries, run
`git diff --check`, and keep the working tree clean before creating the next
branch with `gh stack add`. At the top, validate the cumulative stack, refresh
`origin/main`, cascade any necessary rebase with `gh stack rebase`, and repeat
affected checks. Publication uses `gh stack submit`; each agent-authored PR body
must begin with `🤖 *AI text below* 🤖` and must not claim personal human
review.

## Concrete Steps

Run all commands from the repository root. Configure the local release build
with:

    ./.agent/run.sh cmake --preset release

Build focused targets during iteration, then the full release preset:

    ./.agent/run.sh cmake --build --preset release

Run the compiler and QIR tests with their CTest labels or binaries discovered
from `build/release`, and run DDSIM device tests after the top layer. Finish
each layer with:

    git diff --check
    ./.agent/run.sh uvx nox -s lint
    git status --short

Create layers two and three only after the preceding layer is committed:

    gh stack add agent/1833-1834-qir-runtime
    gh stack add agent/1833-1834-qir-ddsim

Before publication, inspect `gh stack view --json`, refresh the trunk, and use:

    gh stack rebase
    gh stack submit

The final stack view must show three linked PRs ordered from the
compiler/runtime contract at the bottom to DDSIM integration at the top.

## Validation and Acceptance

The bottom layer is accepted when every operation in the shared gate table has
matching compiler lowering, C declaration, runtime implementation, and JIT
registration; no generated declaration uses the old qubits-first parameter
order; duplicate aliases are removed; and the runtime, JIT, builder, QC/QCO,
Base, and Adaptive compiler unit tests pass.

The middle layer is accepted when deliberately malformed entry-point and
declaration types fail with diagnostics that name the symbol and expected type,
two JIT sessions can execute independently, an entry point whose name is not
`main` returns its `i64` status correctly, a fixed seed reproduces a multi-shot
result sequence, output contains complete metadata and the actual shot exit
code, modern result arrays are recorded as one bit string in memory order, and
representative body, adjoint, and generic controlled modules execute through the
supported ABI without relying on legacy qir-runner resource or output adapters.

The top layer is accepted when simultaneous DDSIM QIR jobs return correct,
independent histograms without writing to global stdout, Base state extraction
returns the pre-measurement state, Adaptive sampling succeeds, and the existing
Python FoMaC QIR execution test still passes.

The entire stack is accepted when the release build, affected C++/MLIR/Python
tests, repository lint session, `git diff --check`, and clean-status audit pass
for the final cumulative head. Any environment-only limitation must be recorded
with its exact command and diagnostic instead of weakening the implementation.

## Idempotence and Recovery

Configuration, builds, and tests are repeatable because all generated state is
kept in the worktree-local `build/` and `.cache/` directories. A failed stack
rebase is recovered with `gh stack rebase --abort`; after resolving a reported
conflict, continue with `gh stack rebase --continue`. Do not reset, clean, or
remove another task's worktree. Do not force-push individual branches manually;
if a cascading rebase is required, use `gh stack push`, which applies explicit
force-with-lease checks to every branch.

## Artifacts and Notes

The initial exact revisions used for design were:

    MQT Core origin/main: d3fc13149d54b4b66763d2f2715331fd7821f630
    qir-runner main:      0b75768c123e1d3f70e9a4e5c8b28c9bc3a5afb5
    qir-spec main:        f5647346542d5a65225c3eb349847fe4df01d1b2

The existing serial Python integration test is
`test/python/fomac/test_fomac.py::test_device_executes_qir_program`. It is a
regression oracle, not evidence that concurrent runtime state is safe.

## Interfaces and Dependencies

The canonical compiler-facing QIS must use parameter-first signatures such as
`void __quantum__qis__rx__body(double, Qubit*)` and fixed controlled shorthands
such as `void __quantum__qis__cx__body(Qubit*, Qubit*)`. MQT's two-angle `R`
gate uses `void __quantum__qis__prx__body(double, double, Qubit*)`; the
incompatible qir-runner Pauli-rotation spelling `__quantum__qis__r__body` is not
part of MQT's supported QIS.

The current Adaptive runtime interface must include the QIR 2.1 allocation
forms, including `Qubit* __quantum__rt__qubit_allocate(bool* outErr)` and
buffer-based array allocation/release. The exact exported source spelling may
use the repository's opaque QIR types, but the translated LLVM declaration must
match the profile's `ptr(ptr)` and related signatures.

`JitSession::run` must invoke an exact `int64_t (*)()` entry-point function.
`JitSession` must expose its owned `Runtime&` so DDSIM can retrieve measurements
or move out the state without consulting global state. Runtime reset must clear
per-shot quantum/results/output state without silently reseeding a configured
random-number stream.

Revision note (2026-08-09): Created the initial self-contained plan after the
cross-repository ABI audit and before implementation of stack layer 1.

Revision note (2026-08-10): Rebased the published three-PR stack and revised the
design to favor the current QIR 2.1 runtime contract, MLIR ownership of the gate
registry, the unambiguous `prx` QIS spelling, native one- and two-control entry
points, and generic controlled specializations for larger control sets.
