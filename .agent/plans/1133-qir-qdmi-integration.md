# Complete the target-aware QIR-to-QDMI path

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core already compiles OpenQASM and MLIR programs to textual LLVM IR or LLVM
bitcode and executes either form through the DDSIM QDMI device. After this
change, the same end-to-end workflow also works when compilation uses the
`CompilerTarget` snapshot of that QDMI device. All-to-all targets need neither
heuristic placement nor routing, so target compilation deterministically uses
the target's first sites instead of randomly assigning the program to arbitrary
simulator site IDs.

The behavior is visible from Python by opening `mqt.ddsim.default`, snapshotting
it as a compiler target, compiling a Bell program to QIR Base Profile bitcode,
submitting that bitcode to the same device, and observing a correct Bell
histogram over 1,024 completed shots.

## Progress

- [x] (2026-08-04) Verified that pull request #1687 merged at `a132638c8` and
      that issue #1082 closed while issues #1079 and #1133 remained open.
- [x] (2026-08-04) Created a fresh worktree from merged `origin/main` and read
      the repository, AI-usage, and ExecPlan policies.
- [x] (2026-08-04) Inventoried the QIR compiler, runtime, runner, DDSIM QDMI
      formats, FoMaC submission APIs, Python bindings, documentation, and tests.
- [x] (2026-08-04) Confirmed that untargeted textual and binary QIR submission
      tests pass.
- [x] (2026-08-04) Reproduced a target-aware DDSIM execution trap and isolated
      it to unnecessary all-to-all placement.
- [x] (2026-08-04) Kept physical placement for final conformance, but made
      all-to-all layout selection deterministic and compact instead of running
      random SABRE refinement across every simulator site.
- [x] (2026-08-04) Added focused C++ coverage, strengthened the existing Python
      QIR execution integration, and corrected the DDSIM device documentation.
- [x] (2026-08-04) Passed three focused compiler tests, all 224 compiler unit
      tests, both focused Python QIR tests, all 294 FoMaC Python tests, strict
      documentation, and repository lint.
- [x] (2026-08-04) Completed an independent exact-commit review. Qualified
      Adaptive QIR state-extraction support and strengthened the Python
      regression to verify Bell-state semantics in response.
- [x] (2026-08-04) Re-ran the two focused Python QIR tests, strict
      documentation, repository lint, and `git diff --check` after the review
      fixes.
- [x] (2026-08-04) Independently verified the amended implementation with no
      remaining code, documentation, or test findings.
- [x] (2026-08-04) Published pull request #2007 after explicit authorization and
      added its required changelog entry.

## Surprises & Discoveries

- Observation: Most of issue #1133 is already implemented on `main`. The
  repository contains the DD-based QIR runtime and runner, QIR Base and Adaptive
  compiler lowering, textual and bitcode QDMI formats, FoMaC byte submission,
  and Python tests that compile and execute both payload forms.
- Observation: the existing Python integration tests compile without a target. A
  target-aware probe snapshots DDSIM's 65,535 sites, maps two program qubits to
  arbitrary high IDs, and traps during QIR execution. Evidence: the failing LLVM
  IR used static qubit IDs 18,449 and 2,472 instead of compact IDs zero and one.
- Observation: the `CompilerTarget` contract already states that an absent
  topology means all-to-all connectivity. This case requires no routing, but the
  mapping pass still has to replace logical allocations with physical
  `qco.static` sites for final target-conformance verification.
- Observation: removing the mapping pass for all-to-all targets was therefore
  not viable. The focused compiler test immediately exposed the missing static
  placement, allowing the implementation to remain in Mapping rather than adding
  a pipeline special case.
- Observation: `docs/qdmi/ddsim_device.md` still claims that DDSIM accepts only
  OpenQASM 2 and 3 even though its device properties and tests include QIR Base
  and Adaptive Profile strings and modules.

## Decision Log

- Decision: do not add another compiler, FoMaC, QDMI, or command-line
  convenience API. Rationale: the existing `CompilerTarget.from_device`,
  `compile_program`, `QIRProgram`, and `Device.submit_job` values already form a
  direct typed workflow; another facade would duplicate ownership and format
  boundaries. Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: retain physical placement for every target, but use the identity
  dense layout when `CompilerTarget::hasExplicitTopology()` is false. Rationale:
  final conformance requires physical `qco.static` sites, while an all-to-all
  device needs no SABRE refinement or routing. Mapping program indices to the
  first target sites is deterministic, compact, and preserves explicit site
  identifiers. Explicit-topology targets retain the existing heuristic behavior.
  Date/Author: 2026-08-04, GPT-5.6 via Codex.
- Decision: strengthen the existing Python QIR execution test instead of adding
  another executable or subprocess harness. Rationale: it already crosses the
  compiler, binding, registry, FoMaC, QDMI, JIT, runtime, and result boundaries;
  supplying the device snapshot as its compiler target closes the missing seam
  without test bloat. Date/Author: 2026-08-04, GPT-5.6 via Codex.

## Outcomes & Retrospective

The initial audit prevented a large reimplementation of issue #1133 and found
one narrow integration defect that the existing component-level tests did not
expose. Topology-free targets now retain physical placement while avoiding
randomized layout refinement and routing. The target-aware QIR workflow executes
successfully through DDSIM, and all focused and complete relevant test suites,
strict documentation, and repository lint pass. Independent review found no
remaining actionable issue, and the change was published as pull request #2007
with `Closes #1133`.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` defines `mlir::CompilerTarget`. An absent
coupling topology represents all-to-all connectivity, while an absent operation
set means every operation is native.

`mlir/lib/Compiler/TargetCompilation.cpp` composes the canonical target
pipeline. It currently adds multi-qubit decomposition, generic optimization,
two-qubit fusion, mapping, cleanup, native synthesis, and final conformance in
that order. The mapping pass remains necessary for physical placement even when
its routing work is unnecessary for an all-to-all target.

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` performs both initial
placement and routing. Its random layout spans every target site. This is
appropriate for explicit hardware topology, but an all-to-all simulator with
65,535 sites can therefore receive arbitrary high static site IDs even for a
two-qubit program.

`test/python/fomac/test_fomac.py` already compiles QIR and executes it through
the DDSIM QDMI device in textual and binary form. The textual test will snapshot
the same device as a compiler target before compilation, providing the missing
end-to-end regression.

`docs/qdmi/ddsim_device.md` documents the simulator provider. It must list its
actual QIR formats and show the target-aware compile-and-submit workflow without
duplicating the lower-level payload contract in `docs/qir/index.md`.

## Plan of Work

First, update initial layout selection in
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`. When the target has no
explicit topology, return the identity dense layout immediately instead of
running randomized SABRE refinement. Keep the mapping pass so dynamic program
qubits become physical `qco.static` sites for final conformance. Do not
introduce a special simulator target.

Add a focused compiler unit test in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp`. Construct an all-to-all
target with sparse site IDs, compile a small dynamic QCO program, and verify
that compilation replaces dynamic allocations with the target's first
`qco.static` sites without introducing SWAPs. Existing explicit-topology target
tests continue to prove the heuristic mapping branch.

Then update the existing textual QIR execution test in
`test/python/fomac/test_fomac.py` to snapshot DDSIM, pass that target to
`compile_program`, submit the generated LLVM IR, and verify completed shot
counts. Retain the separate bitcode test so both QDMI payload contracts remain
covered without duplicating the target-specific assertion.

Finally, correct `docs/qdmi/ddsim_device.md` and add one concise Python example
covering target snapshot, QIR Base compilation, bitcode submission, waiting, and
result retrieval. Link to the detailed QIR documentation for format semantics.

## Concrete Steps

Run all commands from the repository root.

Configure and build the release compiler and Python bindings:

    MLIR_DIR=<path-to-MLIR-22>/lib/cmake/mlir \
      ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target \
      mqt-core-mlir-unittests-compiler mqt-core-mlir-bindings

Run the focused compiler and Python regressions:

    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerPipelineTest.QCOProgramUsesCompactAllToAllPlacement'
    ./.agent/run.sh uv run --no-sync pytest \
      test/python/fomac/test_fomac.py::test_device_executes_qir_program \
      test/python/fomac/test_fomac.py::test_device_executes_binary_qir_program \
      -q

Run the complete compiler binary, the complete FoMaC Python file, strict
documentation, and repository lint:

    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    ./.agent/run.sh uv run --no-sync pytest test/python/fomac/test_fomac.py -q
    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check

## Validation and Acceptance

The focused C++ regression must prove that target compilation assigns an
all-to-all program to the target's first physical sites without SWAPs. Existing
explicit-topology tests must continue to exercise heuristic mapping.

The focused Python regression must open `mqt.ddsim.default`, snapshot that same
device as a `CompilerTarget`, compile a Bell program to QIR Base Profile, submit
the textual LLVM IR through QDMI, wait successfully, and return exactly the
requested number of shots. The existing binary test must continue to pass.

Strict documentation must describe QASM and all four QIR formats accurately,
including the Base-only restriction for QIR state extraction. Repository lint
and `git diff --check` must pass. A fresh independent review must find no new
target semantic, QIR, packaging, or test-cohesion issue before publication.

## Idempotence and Recovery

The source and test edits are ordinary patches. Reconfiguration, builds, test
runs, documentation, and lint are safe to repeat. If the target-aware Python
test still traps, inspect its generated LLVM IR and confirm that static qubit
IDs remain compact before changing the QIR runtime. Do not increase DDSIM's
runtime allocation or special-case its stable ID to mask unnecessary mapping.
