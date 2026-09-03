# Compose the compiler-target pipeline

Status: historical implementation record.

Later integration and pass ordering:
[QDMI bridge](1687-int-final-qdmi-bridge.md).

## Goal and scope

MQT Core now has an immutable MLIR-owned `CompilerTarget`, a target-backed
mapping pass, target-independent two-qubit gate fusion, target-native synthesis,
and final target-conformance verification. Those pieces are independently usable
but are not yet composed behind one safe high-level operation. After this
change, a C++ user can call `QCOProgram::compileForTarget(target)` or pass an
optional target to `runDefaultPipeline` and receive a program that has been
decomposed, optimized, mapped, synthesized, verified, and cleaned up in the only
valid order.

The observable proof is a compiler unit test that starts with a multi-controlled
gate, compiles it to a topology-constrained U/CZ target, and finishes with
target-site assignments and only supported operations. The same test would fail
if mapping ran before multi-qubit decomposition, if native synthesis ran before
routing, or if the final verifier were omitted.

This slice also removes the coupling-only `QCOProgram::placeAndRoute` and Python
`QCOProgram.place_and_route` convenience APIs. Mapping remains directly
benchmarkable through `qco::createMappingPass`, while fusion, native synthesis,
and conformance remain directly benchmarkable through their existing pass
factories. The MLIR Compiler Collection has not been released, so no
compatibility shim or upgrade-guide entry is required.

## Constraints

- `QCOProgram::placeAndRoute` is the only remaining caller that converts a
  coupling list into a `CompilerTarget`. Every mapping-pass caller already uses
  `createMappingPass(const CompilerTarget&, MappingPassOptions)`. Evidence: a
  repository-wide symbol search on merged `main` finds the conversion only in
  `mlir/lib/Compiler/Programs.cpp`, its declaration, one compiler test, one
  nanobind definition, and the generated stub.

- the prior draft integration ran the generic QCO pipeline before
  multi-controlled decomposition and then appended target compilation. That
  ordering cannot guarantee that routing sees only one- and two-qubit
  operations. Evidence: merged `runDefaultPipeline` currently runs cleanup, the
  textual QCO pipeline, and cleanup; there is no target path yet.

- `MQTCompilerPipeline` already links `MQTCompilerTarget`, `MLIRQCOTransforms`,
  and `MQT::MLIRSupport`, and `mqt-cc` already links `MQTCompilerPipeline`. A
  small compiler-owned target-pipeline source therefore needs no new dependency
  boundary and can later be reused by the final QDMI/mqt-cc integration.

- target-independent two-qubit gate fusion uses CZ as its fixed symmetric
  generic entangler and rewrites only strictly profitable constant runs. It
  belongs after the ordinary generic optimization passes and before mapping;
  target-native synthesis remains after mapping.

- Jeff conversion deliberately lowers `qco.static` to a generic allocation and
  therefore discards physical site identifiers. A target-aware Jeff result would
  falsely claim successful compilation while losing the mapping. Evidence:
  `QCOToJeff.cpp` converts each static allocation without carrying its index
  into the Jeff program.

- LLVM's `llvm-prefer-static-over-anonymous-namespace` and clang-tidy's
  `readability-static-definition-in-anonymous-namespace` require a file-local
  test helper to be `static` outside the anonymous namespace. Moving only that
  helper satisfied both checks without an exemption.

- Target-aware QIR tests inspect `llvm.inttoptr` operands to prove that sparse
  physical site IDs survive lowering.

## Decisions

- add `populateTargetCompilationPipeline(OpPassManager&, const CompilerTarget&)`
  under `mlir/Compiler/TargetCompilation.{h,cpp}` and build it into
  `MQTCompilerPipeline`. Rationale: the compiler subtree owns `CompilerTarget`
  and high-level program compilation, the existing link graph already provides
  every required pass, and the final mqt-cc/QDMI bridge can reuse one sequence
  without duplicating pass order.

- make the canonical sequence multi-controlled decomposition with a minimum of
  two controls, the default QCO optimization passes, target-independent
  two-qubit gate fusion, target-backed mapping, target-native synthesis,
  target-conformance verification, and QCO cleanup. Rationale: routing is
  defined only for one- and two-qubit operations, optimization should reduce
  work before routing, routing may insert SWAPs, native synthesis must lower
  those SWAPs, and conformance must inspect the final mapped program before
  cleanup.

- expose `QCOProgram::compileForTarget(const CompilerTarget&, bool, bool)` with
  timing and statistics defaulting off, and pass the same options through
  target-aware `runDefaultPipeline`. Rationale: both high-level entry points use
  exactly one pass manager and one sequence while preserving the existing
  observability controls without a private duplicate implementation.

- add `const CompilerTarget* target = nullptr` before the textual QCO pipeline
  argument of the sole `runDefaultPipeline` function. Reject a target for
  `QCImport` and raw `QCO` outputs because those checkpoints deliberately stop
  before optimization. Rationale: a non-null target must never be silently
  ignored, and a pointer expresses an optional borrowed immutable target without
  introducing a second overload or copying requirement.

- reject a custom textual QCO pipeline when a compiler target is supplied.
  Rationale: target compilation has one validated order beginning with
  multi-qubit decomposition; injecting an opaque pipeline before or inside that
  sequence either violates the routing precondition or adds a fallible,
  callback-heavy composition API. Advanced and benchmark users retain
  `QCOProgram::runPassPipeline` and the individual pass factories.

- remove `placeAndRoute` and `place_and_route` outright, regenerate the binding
  stub, and add no migration shim or `UPGRADING.md` entry. Rationale:
  compiler-target mapping is the single modern abstraction, the compiler
  collection is unreleased, and retaining a coupling-only path would recreate
  target construction and configuration redundancy at the public boundary.

- do not expose `CompilerTarget` or `compile_for_target` to Python in this
  slice. Rationale: the final INT-1687 work owns the FoMaC/QDMI adapter, Python
  target binding, and mqt-cc device experience. PIPE-01 only removes the
  obsolete Python coupling API and leaves `compile_program` passing a null
  target until that bridge lands.

- reject target-aware Jeff output alongside `QCImport` and raw QCO, while
  supporting `QCOOptimized`, QC, and QIR output. Rationale: QC and QIR preserve
  the mapped static-site semantics, whereas the current Jeff conversion
  intentionally discards static site identifiers. Extending the Jeff
  representation is outside this compact pipeline-composition slice.

## Outcome and validation

The compiler owns one target-compilation sequence. Tests cover final conformance
with an unsupported measurement target, preservation of QIR site IDs, and
rejection of `jeff` when lowering cannot preserve physical sites. Focused C++,
Python, header, clang-tidy, lint, and diff checks passed. Final hosted CI was
not recorded.

## Code and ownership

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
the validated, cheaply copyable `mlir::CompilerTarget`. An absent topology means
all-to-all connectivity; an explicit topology is canonicalized and connected. An
absent operation set means all operations are native; an explicit operation set
supplies a homogeneous gate capability set and a target-wide synthesis basis
when one can be derived.

`mlir/include/mlir/Dialect/QCO/Transforms/Mapping/Mapping.h` exposes
`qco::createMappingPass(const CompilerTarget&, MappingPassOptions)`. The pass
assigns program qubits to target site identifiers and routes two-qubit
operations over an explicit topology. It deliberately supports only one- and
two-qubit operations, so multi-controlled gates must be decomposed first.

`mlir/include/mlir/Dialect/QCO/Transforms/Passes.h` exposes
`qco::createFuseTwoQubitGates`, `qco::createTargetNativeSynthesis`, and
`qco::createVerifyTargetConformance`. Fusion is target independent and rewrites
only strictly profitable constant two-qubit windows using symmetric CZ-based
resynthesis. Native synthesis lowers unsupported one- and two-qubit operations,
including routing SWAPs, into the target-wide basis. Conformance then rejects
unassigned qubits, unknown static sites, and unsupported final operations.

`mlir/include/mlir/Support/Passes.h` and `mlir/lib/Support/Passes.cpp` own
reusable cleanup and optimization populators.
`populateDecomposeMultiControlledPipeline(pm, 2)` lowers supported
multi-controlled forms. `populateDefaultQCOOptimizationPipeline` performs the
ordinary target-independent QCO optimization. `populateQCOCleanupPipeline`
canonicalizes, normalizes global phase, eliminates common subexpressions,
shrinks QTensor allocations, and removes dead values.

`mlir/include/mlir/Compiler/Programs.h` and `mlir/lib/Compiler/Programs.cpp`
define move-aware typed programs and `runDefaultPipeline`. The local `runPasses`
helper currently templates over a populator even though every caller has the
same void contract. PIPE-01 replaces that template with
`llvm::function_ref<void(OpPassManager&)>`, retaining module diagnostics and
adding optional timing/statistics configuration.

`bindings/mlir/register_mlir.cpp` defines the Python binding and
`python/mqt/core/mlir.pyi` is generated from it by the `stubs` Nox session. The
generated stub must never be edited by hand. The binding currently exposes the
obsolete coupling-only `place_and_route`; this slice deletes that definition and
regenerates the stub. It does not yet bind `CompilerTarget`.

The focused compiler tests live in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp` and build as
`build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`.
Pass-specific mapping and synthesis behavior remains covered by their dedicated
unit tests. PIPE-01 adds only integration behavior that cannot be proven by
those isolated suites.

The QDMI adapter and Python target construction belong to
[the integration record](1687-int-final-qdmi-bridge.md).

## Acceptance

The new `TargetCompilation` populator is public from `MQTCompilerPipeline` and
contains each required pass exactly once in this order: multi-controlled
decomposition, default QCO optimization, target-independent two-qubit gate
fusion, target-backed mapping, target-native synthesis, target-conformance
verification, and QCO cleanup.

`QCOProgram::compileForTarget` succeeds on a multi-controlled circuit and a
three-site line target with U/CZ capabilities. The resulting program verifies,
uses `qco.static` target assignments, contains no `qco.alloc` or `qco.swap`, and
contains no operation rejected by `CompilerTarget::supports`.

`runDefaultPipeline` without a target preserves every current output-format and
custom-pipeline behavior. With a target, it produces the same conforming target
program for `QCOOptimized`, QC, and QIR output. A target combined with
`QCImport`, raw `QCO`, Jeff, or a custom textual QCO pipeline returns no result
with a regular diagnostic instead of silently ignoring the target.

No `placeAndRoute`, `place_and_route`, coupling-to-target conversion helper, or
generated stub entry remains. The mapping, fusion, synthesis, and conformance
pass factories remain independently callable.

All focused suites and repository lint pass. The final diff contains no
compatibility shim, no `UPGRADING.md` change, no QDMI/FoMaC adapter, no mqt-cc
device integration, and no unrelated cleanup.

## Interfaces

At completion, `mlir/include/mlir/Compiler/TargetCompilation.h` declares:

    namespace mlir {
    class CompilerTarget;
    class OpPassManager;

    void populateTargetCompilationPipeline(
        OpPassManager& pm, const CompilerTarget& target);
    }

`QCOProgram` declares:

    [[nodiscard]] bool
    compileForTarget(const CompilerTarget& target,
                     bool enableTiming = false,
                     bool enableStatistics = false);

The sole default-pipeline entry point declares:

    [[nodiscard]] std::optional<CompilerProgram>
    runDefaultPipeline(
        CompilerInput&& program, ProgramFormat output,
        const CompilerTarget* target = nullptr,
        std::string_view qcoPipeline = "mqt-qco-default",
        bool enableTiming = false, bool enableStatistics = false);

`MQTCompilerPipeline` owns `TargetCompilation.cpp` and links only the existing
MLIR compiler target, QCO transform, and support libraries. CoreFoMaC, QDMI,
CoreIR, and a dynamic provider boundary are not added by this slice.
