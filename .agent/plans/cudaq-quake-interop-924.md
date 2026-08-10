# Add native CUDA-Q Quake interoperability

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, users can parse the reference-semantics Quake MLIR emitted by
CUDA-Q 0.15 as an MQT compiler program, convert it to MQT's QC dialect, optimize
it through the existing QC-to-QCO pipeline, and emit Quake again. CUDA-Q remains
an optional external tool: MQT Core neither links against CUDA-Q nor declares a
Python dependency on it. The user-visible proof is an executable documentation
example that takes a Bell kernel's Quake text through QC and QIR and submits the
QIR to the bundled DDSIM QDMI device.

## Progress

- [x] (2026-08-09 21:56Z) Allocated an isolated worktree from current
      `origin/main` and read repository policy.
- [x] (2026-08-10 00:10Z) Inspected CUDA-Q 0.15 Quake/CC assembly and current
      MQT compiler extension points.
- [x] (2026-08-10 00:18Z) Added compact compatibility dialects and native
      Quake-to-QC and QC-to-Quake conversions.
- [x] (2026-08-10 00:32Z) Added `QuakeProgram`, compiler formats, Python
      bindings, CLI behavior, and focused native/Python tests.
- [x] (2026-08-10 00:39Z) Added the MyST-NB interoperability guide, CUDA-Q 0.15
      fixture, and restrained live CUDA-Q smoke workflow.
- [x] (2026-08-10 01:07Z) Ran focused native/Python tests, executable
      documentation, generated-stub validation, the full pre-commit suite, and
      final diff checks.

## Surprises & Discoveries

- Observation: The current compiler already supports OpenQASM 3 as an output
  format in addition to the program types described in the original issue.
  Evidence: `ProgramFormat::OpenQASM3` and its Python/CLI handling are present
  in `mlir/include/mlir/Compiler/Programs.h`, `bindings/mlir/register_mlir.cpp`,
  and `mlir/tools/mqt-cc/mqt-cc.cpp`. The Quake integration should follow that
  current shape rather than an older snapshot.
- Observation: QC measurements must remain externally observable across the
  existing QC-to-QCO optimization path. Evidence: returning a generated
  `memref<Nxi1>` from Quake import prevents the measurement operations from
  being removed and also feeds the normal QIR output-record lowering.
- Observation: preserving CUDA-Q's `quake.mangled_name_map` verbatim is wrong
  after QC-to-Quake renames the entry point. Evidence: CUDA-Q 0.15 creates the
  map from the actual entry symbol and its Python entry-point rewrite name.
  Export now replaces this known attribute while retaining unrelated module
  attributes.
- Observation: the preconfigured debug tree intentionally has Python bindings
  disabled and the system Python does not provide nanobind. Binding validation
  therefore belongs to the repository's isolated Nox stub-generation session,
  which installs the declared build group.
- Observation: the documentation executor changes its working directory to the
  page directory, so the checked-in fallback fixture must be found relative to
  both `docs/mlir/` and the repository root. The notebook now handles both
  contexts explicitly.
- Observation: submitting binary QIR bitcode for 1024 shots made the executable
  example unnecessarily slow on the local DDSIM device. The documented path now
  submits textual LLVM IR for 128 shots, which is the requested QIR boundary and
  keeps the page fast and deterministic enough for warnings-as-errors builds.

## Decision Log

- Decision: Build a small MQT-owned compatibility projection of the textual
  `quake` and `cc` namespaces rather than linking CUDA-Q or copying its complete
  dialect implementation. Rationale: the integration needs only the assembly
  forms crossing `cudaq.synthesize`, while CUDA-Q's compiler is built against a
  patched MLIR fork. Date/Author: 2026-08-09, Codex.
- Decision: Expose only Quake-to-QC and QC-to-Quake conversions. Rationale:
  existing QC-to-QCO and QCO-to-QC conversions already provide the intended
  optimization path without duplicating APIs. Date/Author: 2026-08-09, Codex.
- Decision: Do not add a Python extra or Python adapter module. Rationale: the
  first iteration can interoperate through `str(cudaq.synthesize(...))` and
  CUDA-Q's existing `merge_quake_source`, keeping package metadata unchanged.
  Date/Author: 2026-08-09, Codex.
- Decision: Materialize Quake measurement results as the QC entry function's
  result register. Rationale: this preserves ordering and names through the
  existing QC/QCO/QIR path and makes the measurements observable to cleanup
  passes. Date/Author: 2026-08-10, Codex.
- Decision: Regenerate the CUDA-Q mangled-name map for the requested exported
  kernel name. Rationale: unknown metadata is preserved where practical, but a
  known symbol map cannot remain valid after renaming. Date/Author: 2026-08-10,
  Codex.

## Outcomes & Retrospective

The first native Quake interoperability slice is complete. MQT Core now owns a
small textual compatibility dialect, imports common CUDA-Q 0.15 reference-form
kernels into QC, and emits conservative reference-form Quake from QC. The public
C++/Python compiler APIs and `mqt-cc` recognize `.qke` and Quake as a normal
program format. QCO remains deliberately absent from the direct API and is used
through the existing QC conversions. The executable guide proves the
reproducible MQT-only fallback path through QC, QCO, QIR Base Profile, and
DDSIM, while one Linux CI smoke job covers live CUDA-Q synthesis and execution.

Validation completed successfully: the compiler unit-test and CLI targets built;
all five `QuakeProgramTest.*` tests passed; both focused Python API/file-format
tests passed; stub generation passed; the complete warnings-as-errors docs build
passed; and `prek run --all-files` plus `git diff --check` passed. The live
CUDA-Q 0.15 test was not run locally because CUDA-Q does not publish that wheel
for macOS; it is isolated in the added Ubuntu/Python 3.12 smoke job. Deferred by
design are SSI `wire`/`cable`, state initialization, noise, custom unitaries,
unsynthesized dynamic allocation, and `quake.phase` until a released schema is
available.

## Context and Orientation

MQT compiler programs are move-only wrappers around an MLIR module and context.
Their public definitions are in `mlir/include/mlir/Compiler/Programs.h`, their
implementations are in `mlir/lib/Compiler/Programs.cpp`, and Python bindings are
registered in `bindings/mlir/register_mlir.cpp`. `QCProgram` holds the
reference-semantics QC dialect, while `QCOProgram` holds the value-semantics QCO
dialect used for optimization. The coordinated default pipeline converts inputs
to QCO, optimizes them, and converts back to a requested result format.

CUDA-Q calls its quantum MLIR dialect Quake. `cudaq.synthesize(kernel, *args)`
specializes runtime arguments but retains reference semantics in CUDA-Q 0.15.
The resulting module uses the textual namespaces `quake` for quantum operations
and `cc` for CUDA-Q classical operations. The compatibility dialect implemented
here is not intended to be a general CUDA-Q compiler API; it exists only to
parse and emit the supported textual boundary.

Dialect definitions and conversions live under `mlir/include/mlir/Dialect/` and
`mlir/lib/Dialect/` or `mlir/include/mlir/Conversion/` and
`mlir/lib/Conversion/`, with CMake files in the same hierarchy. MLIR unit tests
live under `mlir/unittests/`. The Python compiler guide is the executable
MyST-NB page `docs/mlir/python_compiler_collection.md`; the CUDA-Q page should
use the same format and be linked from `docs/mlir/index.md`.

The implementation must preserve unrelated changes and remain entirely within
this task's worktree. It must follow `AGENTS.md` and `docs/ai_usage.md`. This
ExecPlan authorizes no GitHub publication.

## Plan of Work

First, inspect representative CUDA-Q 0.15 Quake modules and reduce the crossing
surface to the types and operations needed for common specialized kernels. Add
compact TableGen definitions whose textual dialect namespaces are exactly
`quake` and `cc`, but whose C++ namespace is owned by MQT. Include only the
required custom assembly parsing and printing and a short provenance document
with CUDA-Q's Apache-2.0 attribution.

Next, implement Quake-to-QC and QC-to-Quake conversion passes. Import must
validate the module before changing it, select the CUDA-Q entry point, inline or
reject unsupported calls deterministically, translate allocations, gates,
controls, adjoints, measurements, and the supported structured classical control
flow, and produce a valid QC program. Export must emit conservative CUDA-Q 0.15
reference-form Quake. It must reject a nonzero global phase unless the caller
explicitly permits dropping it.

Then extend the typed compiler program API with `QuakeProgram`, add Quake to the
input and result variants and coordinated pipeline, bind the type and conversion
methods to Python, and add Quake input/output handling to `mqt-cc`. Do not add
any direct method on `QCOProgram`; generic pipeline output may compose through
QC internally. Regenerate Python stubs with the repository Nox session.

Finally, add `docs/mlir/CUDAQuake.md` as a MyST-NB page. It should use a
captured CUDA-Q 0.15 Bell fixture when CUDA-Q is unavailable, convert it through
QC and QCO to QIR, and submit QIR to the bundled DDSIM device. When CUDA-Q is
importable, the same page should synthesize the fixture live. Include a short
reverse-direction example and an explicit limitations table. Add one small Linux
smoke workflow that installs CUDA-Q directly and checks the two expected
boundary crossings; do not add a version or packaging matrix.

## Concrete Steps

Run all commands from the repository root through `.agent/run.sh` when they
create caches or build artifacts.

Inspect the current surfaces with:

    rg -n "ProgramFormat|QCProgram|runDefaultPipeline" mlir bindings
    rg -n "quake\\.|cc\\." <representative CUDA-Q 0.15 source tree>

Configure and build after the initial dialect/conversion slice with:

    MLIR_DIR=/path/to/llvm-22.1/lib/cmake/mlir ./.agent/run.sh cmake --preset debug
    ./.agent/run.sh cmake --build --preset debug --target mqt-core-mlir-unittests-compiler

Run the focused native tests with:

    ./build/debug/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter='CompilerPipelineTest.*Quake*'

After binding changes, install the package, regenerate stubs, and run focused
Python tests:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uv run --no-sync pytest test/python/test_mlir.py -k quake

Build documentation and run final validation with:

    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check
    git status --short

## Validation and Acceptance

A captured CUDA-Q 0.15 Bell module must parse as `QuakeProgram`, convert to a
valid `QCProgram`, and contain one Hadamard, one controlled X, and two
measurements. A QC Bell program must emit Quake that the compatibility parser
accepts, and converting that emitted module back to QC must preserve those
operations and measurement order. One structured fixture must prove a
parameterized controlled gate, measurement feedback, and a bounded loop. A
fixture containing a deliberately unsupported operation must fail with that
operation's name. Exporting a QC program with nonzero global phase must fail by
default and succeed only when phase dropping is explicitly requested.

The Python API must expose `QuakeProgram.from_mlir_str`, `QuakeProgram.to_qc`,
and `QCProgram.to_quake` with the documented move/copy behavior. The default
compiler must accept Quake input and produce requested QC, QCO, QIR, OpenQASM,
Jeff, or Quake output through the normal pipeline.

The documentation build must execute the captured-fixture path without CUDA-Q
installed. The dedicated smoke job must synthesize a live CUDA-Q Bell kernel,
import it into MQT, and parse one MQT-emitted Quake module with CUDA-Q. All
focused tests, the docs build, generated-stub check, and lint must pass.

## Idempotence and Recovery

All source edits are additive or local conversions and can be rebuilt safely.
CMake configuration and Nox sessions are repeatable through `.agent/run.sh`.
Generated Python stubs must be regenerated rather than hand-edited. If a
TableGen definition proves too broad, remove only the unused local definition
and its conversion test; never modify or clean another worktree. If CUDA-Q is
unavailable in the local environment, use the checked-in fixture and leave the
live smoke job to its supported Linux CI environment.

## Artifacts and Notes

- `mqt-core-mlir-unittests-compiler` built successfully with the compact dialect
  and both translations linked.
- All five focused `QuakeProgramTest.*` native tests passed.
- `mqt-cc` successfully imported the checked-in Bell Quake fixture, emitted QC,
  emitted reference-form Quake with measurement names, and lowered the same
  input to QIR Base Profile.
- The focused Python Quake/`.qke` tests passed (2 passed, 44 deselected), and
  Nox regenerated the public stub successfully.
- The full MyST-NB/Sphinx documentation build passed with the CUDA-Q-free
  fallback page executed in under a second.
- The repository-wide `prek run --all-files` suite and `git diff --check`
  completed successfully.

## Interfaces and Dependencies

The final C++ API must define a move-only `mlir::QuakeProgram` parallel to
`mlir::QCProgram`, with static `fromMLIRString` and `fromMLIRFile`, `copy`, and
rvalue `intoQC` methods. `mlir::QCProgram` must gain an rvalue `intoQuake`
method accepting an options value containing the entry-point name and whether
global phase may be ignored. `ProgramFormat`, `CompilerInput`, and
`CompilerProgram` must include Quake where appropriate.

The Python API must mirror these methods using snake_case names and the existing
`copy=False` convention. There is no CUDA-Q import in MQT's Python package and
no new Python project dependency. Native code depends only on the existing
LLVM/MLIR 22.1 toolchain and MQT dialect libraries.

Revision note (2026-08-09): Initial ExecPlan created from the approved native
Quake interoperability design and current `origin/main` repository shape.
