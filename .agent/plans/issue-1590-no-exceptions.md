# Build the MQT MLIR compiler without C++ exceptions

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this work, downstream projects built against an LLVM and MLIR toolchain
configured without C++ exception handling can build and use MQT's MLIR compiler
libraries without globally re-enabling exceptions. Invalid compiler-target data
and invalid OpenQASM source still produce useful errors; Python callers retain
ordinary Python exceptions at the binding boundary. A strict CMake build proves
the contract by compiling the exception-free MLIR target set with the platform's
no-exception option and by rejecting accidental `throw` or `catch` syntax in
that target set.

The work is delivered as a small stacked series. The existing OpenQASM angle
work in pull request #2040 is the semantic predecessor because it substantially
changes the same analyzer. Each later pull request is based on the branch below
it and must remain independently buildable and testable.

## Progress

- [x] (2026-08-11 00:11Z) Revalidated issue #1590, every currently open pull
      request, current `main`, and exact pull request #2040 metadata, checks,
      and review threads.
- [x] (2026-08-11 00:11Z) Chose the stack boundaries and allocated a clean
  worktree for the first layer at the exact #2040 head.
- [x] (2026-08-11 00:43Z) Replace throwing compiler-target construction and
      adapter validation with explicit `llvm::Expected` results, preserve Python
      exceptions in nanobind, update callers, and validate the first layer. The
      binding build, generated stubs, five focused Python tests, full release
      build, all 4,629 CTests, and repository lint pass. Upgrade guidance
      records the C++ source migration.
- [x] (2026-08-11 00:52Z) Publish the first layer as draft pull request #2049,
  based on #2040 at its unchanged exact head.
- [x] (2026-08-11 01:41Z) Replace non-local OpenQASM semantic exceptions with
      explicit `FailureOr` and `LogicalResult` propagation. The complete
      OpenQASM target suite passes all 183 tests, including exact diagnostic
      coverage for recursive custom gates after body analysis.
- [ ] Introduce the CMake no-exception contract, isolate or omit integrations
  whose upstream APIs still require exception handling, add CI coverage and
  documentation, and validate the final layer.
- [ ] Review, publish, and link the exact revisions with GitHub's stacked pull
  request feature; then inspect replacement CI without merging.

## Surprises & Discoveries

- Observation: `cmake/CompilerOptions.cmake` currently adds `-fexceptions` to
  the project-options interface on every non-MSVC build, overriding the
  `-fno-exceptions` contract supplied by a typical LLVM build. Evidence:
  `enable_project_options` unconditionally appends the option.
- Observation: the exact #2040 head is currently conflicting with `main` and has
  failing C++ lint, documentation, and C++ patch-coverage checks, although most
  platform tests pass and there are no reviews or review threads. The new layers
  can be developed at that immutable head, but the stack cannot become
  merge-ready until its root is repaired.
- Observation: `mlir/lib/Compiler/Target.cpp` owns all exception syntax in the
  standalone target model, while `bindings/mlir/register_mlir.cpp` is already a
  natural Python exception boundary. This permits the core model to return
  explicit errors without weakening Python diagnostics.
- Observation: the repository-local release preset does not select the required
  LLVM installation automatically in a fresh worktree, and the global Python
  interpreter does not provide nanobind. Supplying the established LLVM 22.1.3
  `MLIR_DIR` and the worktree's `.venv/bin/python` makes both configurations
  reproducible; these were environment boundaries, not source failures.
- Observation: the semantic analyzer has 169 error-producing call sites spread
  across constant evaluation, type resolution, structured control flow, gate
  graph validation, and output finalization. Encoding failure in each return
  type let the compiler expose every missed propagation site during the
  conversion; no public frontend result type had to change.

## Decision Log

- Decision: Base the series on the exact #2040 head rather than duplicate its
  OpenQASM changes. Rationale: its analyzer changes overlap heavily with the
  semantic-error refactor and must be reviewed first. Date/Author: 2026-08-11,
  Codex.
- Decision: Use separate pull requests for target errors, OpenQASM errors, and
  build enforcement. Rationale: each layer has a distinct contract, focused
  tests, and review audience; the final flag change then demonstrates that the
  preceding semantic refactors are complete. Date/Author: 2026-08-11, Codex.
- Decision: Use `llvm::Expected<T>` for fallible target construction and
  `llvm::Error` for validation helpers. Rationale: these LLVM-native types carry
  explanatory errors through exception-disabled code and integrate with the
  compiler's existing LLVM dependency. Date/Author: 2026-08-11, Codex.
- Decision: Keep exception translation in the nanobind extension and in optional
  compatibility adapters that call exception-based upstream APIs. Rationale:
  no-exception consumers need an exception-free compiler core, while Python and
  legacy Core integrations still require their established error behavior.
  Date/Author: 2026-08-11, Codex.
- Decision: Retain only the first semantic diagnostic in the analyzer and stop
  immediately after failure. Rationale: this preserves the previous exception
  unwinding behavior exactly while making propagation visible in function
  signatures and keeping syntax-level multi-diagnostic recovery separate.
  Date/Author: 2026-08-11, Codex.

## Outcomes & Retrospective

The first two stack layers are complete locally. Compiler-target construction
and the FoMaC snapshot adapter use explicit LLVM errors throughout the C++ layer
while the nanobind boundary preserves Python `ValueError`. The OpenQASM semantic
analyzer now propagates every error explicitly and preserves its existing
diagnostics. Valid target behavior is unchanged across the compiler, mapping,
native-synthesis, and complete OpenQASM suites. The build-enforcement layer
remains in progress.

## Context and Orientation

`cmake/CompilerOptions.cmake` defines the `MQT::ProjectOptions` interface used
throughout the repository. `mlir/CMakeLists.txt` applies that interface to MLIR
libraries through `mqt_mlir_target_use_project_options`. The unconditional
exception flag currently prevents a downstream LLVM no-exception configuration
from remaining consistent.

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
the immutable compiler-target model used by mapping, target-native synthesis,
target conformance checks, Python bindings, and the FoMaC device snapshot
adapter. The current constructors throw `std::invalid_argument` and several
routing accessors throw `std::out_of_range`. The first layer replaces fallible
public constructors with named factories returning `llvm::Expected`, and
documents valid routing vertices as preconditions for fast query methods. The
nanobind code in `bindings/mlir/register_mlir.cpp` consumes each expected result
and raises `ValueError`, preserving the Python-facing contract.

`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` converts parsed OpenQASM syntax
into a typed program. Its `SemanticAnalyzer::fail` helper currently throws a
private exception from many nested semantic routines and `run` catches it at the
top. The second layer changes those routines to return success, failure, or an
expected value explicitly, storing no partially successful program after a
diagnostic. Existing parser and frontend result types in
`mlir/include/mlir/Target/OpenQASM` remain the public boundary.

The compiler pipeline links optional external surfaces. `MLIRQCTranslation` uses
the exception-based `QuantumComputation` model, `MQTCompilerFoMaCAdapter` calls
FoMaC APIs, and `MLIRQCODDFunctionality` calls the decision-diagram package. The
final layer keeps those integrations in clearly named compatibility targets that
may enable exception handling in a normal build and omits them from the strict
no-exception target set. Core MLIR dialects, conversions, OpenQASM-native
translation, compiler programs, and tools that do not require those adapters
must compile without exceptions.

Pull requests #1701 and #1845 add exception syntax in MLIR code and therefore
must be updated to the explicit-error contract before merging. Pull request 1973
expands the decision-diagram bridge and must preserve its compatibility
boundary. Pull requests #1901, #2025, #2042, and #2043 disagree about the future
of FoMaC; this plan does not resolve that separate architecture decision and
keeps the optional adapter thin. Pull request #2031's direct Qiskit C bridge is
compatible with reducing reliance on the exception-based `QuantumComputation`
translation path. Pull request #1955 supplies an appropriate benchmark harness
for later performance measurements but no performance claim is required for
acceptance here.

## Plan of Work

In the first layer, add named `create` factories to `CompilerTarget` and its
nested metadata value types. Move all validation into helpers that return
`llvm::Error`, construct values only after validation succeeds, and remove
exception headers and syntax from `Target.cpp`. Update valid C++ call sites to
consume the expected value explicitly, using `llvm::cantFail` only for local
compile-time test fixtures whose validity is the premise of the test. Change the
FoMaC snapshot function to return `llvm::Expected<CompilerTarget>` and propagate
adapter validation errors. In nanobind lambdas, convert an LLVM error to
`nb::value_error`; do not expose LLVM result wrappers to Python. Add C++ and
Python tests for the exact diagnostic messages and update generated stubs only
through the repository's stub session.

In the second layer, introduce a small semantic result vocabulary in
`OpenQASMSemantics.cpp`: routines that only validate return
`mlir::LogicalResult`, routines that compute a value return `mlir::FailureOr<T>`
or a local expected result that also retains the first `Diagnostic`, and callers
immediately propagate failure. Replace every `SemanticError`, `throw`, and
`catch` path, including unsigned-to-signed overflow, recursive gate detection,
depth limits, and output initialization. `SemanticAnalyzer::run` returns the
accumulated diagnostic on failure and a typed program only after all stages
succeed. Extend focused frontend tests so representative failures at shallow and
deeply nested call sites preserve their location and message.

In the final layer, stop forcing exceptions globally. Add a documented CMake
option for the supported strict MLIR build and a helper that applies the
compiler-specific no-exception flag (`-fno-exceptions` for GCC and Clang, the
corresponding MSVC setting when supported) to the exception-free target set.
Keep normal builds source-compatible by enabling exceptions only for explicit
compatibility targets that need exception-based Core or Python APIs. Add a
configure-time or compile-time regression check that reports the exact target
introducing exception syntax instead of silently overriding the LLVM contract.
Add a Linux CI job that configures, builds, and tests the strict target set
against the supported LLVM 22 toolchain. Document which optional integrations
are unavailable in strict mode, add upgrade guidance for the target factories,
and add a changelog entry.

## Concrete Steps

All commands run from the repository root through `.agent/run.sh` when they
produce caches or build artifacts.

For each layer, configure and build the release preset:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release

For the target-model layer, run:

    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerTargetTest.*:CompilerFoMaCAdapterTest.*'
    ./.agent/run.sh uv run --no-sync pytest test/python/test_mlir.py -k target

For the OpenQASM layer, build and run the repository's focused OpenQASM target
test binary after confirming its exact generated name in
`mlir/unittests/Target/OpenQASM/CMakeLists.txt`. Run the existing invalid-source
fixtures and the newly added propagation tests.

For the final layer, configure the new strict preset or option, build its
documented target set, and run its tests. Then build the ordinary release preset
to prove compatibility adapters still work. Finish every layer with:

    ./.agent/run.sh uvx nox -s lint
    git diff --check
    git status --short

Publish each exact signed commit only after these checks. Create draft pull
requests with the immediately preceding stack branch as their base, then link
the existing #2040 pull request and the new pull requests bottom-to-top with
GitHub's stack command. Do not merge any pull request.

## Validation and Acceptance

The first layer is accepted when invalid C++ target metadata returns an
`llvm::Error` with the established message, valid targets behave identically,
invalid Python construction still raises `ValueError`, and neither `Target.cpp`
nor its public header contains `throw`, `catch`, or a standard exception
dependency.

The second layer is accepted when all existing valid and invalid OpenQASM tests
pass, representative nested failures retain their source location and message,
and the semantic analyzer contains no `throw`, `catch`, or private exception
class. A typed program must never accompany a diagnostic.

The final layer is accepted when a clean strict configuration compiles and tests
the documented MLIR target set without an exception flag override, an ordinary
release build still includes and tests Python, QuantumComputation, FoMaC, and DD
compatibility where configured, and full lint passes. CI on each exact stack
head must be inspected; pending or failing checks remain explicitly reported and
prevent readiness.

## Idempotence and Recovery

Configuration, build, test, lint, and stub-generation commands are repeatable
inside their layer's worktree. Each layer uses its own build and cache trees. If
a downstream layer discovers an API correction, amend it on the owning upstream
branch, validate that branch, and then rebase descendants in order; never
rewrite another task's worktree. Push rewritten stack commits only with
revision-scoped force-with-lease after verifying the remote head, and prefer new
follow-up commits when a rewrite is unnecessary.

If #2040 changes, fetch its new exact head, inspect the changed OpenQASM
surfaces, and rebase the first layer only after preserving all local commits.
Then rebase each descendant in order and rerun focused tests. A conflict in the
semantic analyzer must be resolved against the current behavior and tests, not
by mechanically choosing one side.

## Artifacts and Notes

The initial live audit recorded #2040 at commit
`7068c29077e2bf24281ee016cf49771f7f262733`, based on an older `main`, with no
reviews or review threads. This commit is evidence for the initial dependency
only; refresh it before publication because remote state can change.

The first implementation layer is draft pull request #2049 at signed commit
`801bf71231fb143cf84b28789a6f0369a51dfcaf`. Its base branch is
`agent/issue-1128-angle-precision`; the local and remote bases were exact when
the pull request was created.

The local LLVM 22 configuration advertises exception handling as disabled. The
current project flag reverses that setting for MQT targets, so deleting the
override without first removing explicit exception paths would fail the build.
The stack order exists to keep each intermediate revision buildable.

## Interfaces and Dependencies

At the end of the first layer, `mlir/Compiler/Target.h` provides static factory
functions returning `llvm::Expected<DurationUnit>`, `llvm::Expected<Site>`,
`llvm::Expected<SiteTuple>`, `llvm::Expected<Operation>`, and
`llvm::Expected<CompilerTarget>`. Direct fallible constructors are not public.
`compilerTargetFromDevice` returns `llvm::Expected<CompilerTarget>`. Error text
uses LLVM string errors with an invalid-argument error code. The nanobind module
unwraps these results into the existing Python classes and raises
`nanobind::value_error` with the LLVM error text.

At the end of the second layer, semantic analysis expresses failure in return
types and the existing `AnalysisResult::diagnostics`; no public frontend API
needs an exception-capable ABI.

At the end of the final layer, normal builds retain compatibility integrations,
while the strict CMake contract names and builds an exception-free MLIR target
set. LLVM and MLIR remain version 22 or newer, C++ remains C++20, and no new
runtime dependency is introduced.

Revision note (2026-08-11): created the plan after the live issue, pull request,
source, and build-contract audit; implementation evidence will be added as each
stack layer completes.
