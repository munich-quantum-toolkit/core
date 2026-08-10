# Add a thin Qiskit C-API bridge for MLIR QC programs

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core's Python bindings currently accept a Qiskit `QuantumCircuit` by calling
the legacy Python loader, materializing a `QuantumComputation`, and then
translating that object to MLIR QC. After this work, `QCProgram` imports and
exports Qiskit circuits directly through Qiskit's experimental C API. The legacy
converters and `mqt.core.load` remain unchanged, normal builds still do not
require Qiskit or Rust, and importing MQT does not import Qiskit.

The first shipping adapter supports final Qiskit 2.5.x releases. The binding
examines the installed Python package version before touching any Qiskit
capsule, selects the exact minor adapter, imports and verifies that adapter's C
API table, and reports a clear error for prereleases or unsupported minors. A
user can demonstrate the feature by importing a circuit with registers,
parameters, measurement, global phase, and nested structured control into a
`QCProgram`, compiling it, and exporting a flat supported `QCProgram` back to a
Qiskit `QuantumCircuit` without invoking `QuantumComputation`.

## Progress

- [x] (2026-08-09 21:06Z) Refreshed `origin/main`, allocated a clean isolated
      worktree, and read the workspace, repository, AI-use, and ExecPlan
      policies.
- [x] (2026-08-09 21:06Z) Mapped the existing MLIR Python binding, program
      builders, QC/QCO dialects and conversions, test suite, nox session,
      upstream workflow, and current Qiskit dependency boundary.
- [x] (2026-08-09 21:06Z) Downloaded and inspected the official Qiskit 2.5.0
      wheel, verified its SHA-256, enumerated its eight C-API headers, and
      exercised the installed 2.5.1 capsule enough to establish the adapter
      ownership and version checks.
- [x] (2026-08-09 22:32Z) Vendor the Qiskit 2.5.0 C-API headers, license, and
      machine-readable provenance without exposing them through MQT's installed
      C++ interface.
- [x] (2026-08-09 23:41Z) Add the private MQT adapter facade, exact version
      dispatcher, 2.5 adapter object target, and shared QC import/export
      implementation.
- [x] (2026-08-09 23:54Z) Add the small QC/QCO dense-unitary operation needed to
      preserve Qiskit `UnitaryGate` semantics through the compiler pipeline.
- [x] (2026-08-10 00:04Z) Bind direct `QCProgram.from_qiskit`, non-consuming
      `QCProgram.to_qiskit`, and direct Qiskit input dispatch for
      `compile_program`, preserving all legacy APIs.
- [x] (2026-08-10 00:52Z) Add focused functional, unsupported-boundary,
      dispatcher, and lazy-import tests; update documentation and changelog.
- [x] (2026-08-10 00:43Z) Add the released-minor maintenance session and
      upstream-development candidate-adapter path, including API-surface
      comparison and focused validation.
- [x] (2026-08-10 01:04Z) Regenerate Python stubs and complete debug, stable-ABI
      release, candidate-adapter, native, Python, documentation,
      repository-lint, provenance, and diff validation.

## Surprises & Discoveries

- Observation: `QCProgram.from_qiskit` and `compile_program(QuantumCircuit)`
  both currently call `mqt.core.load.load`, so the compiler bridge has no
  independent Qiskit representation boundary.
- Observation: `qk_circuit_borrow_from_python` accepts Qiskit's internal
  `CircuitData`, available as `QuantumCircuit._data`, rather than the public
  `QuantumCircuit` wrapper. The adapter must keep that Python object alive for
  the lifetime of the borrowed `QkCircuit` and must not expose this detail
  outside its translation unit.
- Observation: Qiskit's header-defined C-API tables and `qk_import()` state are
  translation-unit local. Every `Qk*` access therefore has to remain in the
  minor adapter source; putting even cleanup helpers in the shared translator
  would create a second uninitialized table.
- Observation: Qiskit parameter symbols have UUID identity, so constructing a
  second symbol from the displayed name does not establish equality. The adapter
  reads the root circuit's declared `parameters` with the stable Python C API,
  then classifies C-API parameters structurally as numeric, declared bare
  symbols, loop-local symbols, or rejected composite expressions. Export caches
  one native symbol per name so repeated MLIR uses preserve Qiskit identity.
- Observation: QC and QCO have standard gates and composable modifiers but no
  generic matrix-backed unitary operation. Direct `UnitaryGate` support cannot
  be implemented truthfully by the binding alone. A dense-unitary operation in
  each dialect, plus direct QC/QCO conversion, is the smallest representation
  extension that keeps this bridge native and preserves downstream compiler
  semantics.
- Observation: Qiskit 2.5 exposes control-flow and classical-expression
  inspection but not equivalent construction APIs. Import can lower supported
  trees to SCF and arithmetic operations, while export must reject structured
  control and expression execution before creating a partial circuit.
- Observation: The normal debug binding uses the full CPython API, while wheel
  builds define `Py_LIMITED_API`. The stable-ABI build caught an unavailable
  `PyUnicode_AsUTF8` call; using `PyUnicode_AsEncodedString` plus
  `PyBytes_AsString` keeps the dispatcher compatible with both modes.
- Observation: The second `uv sync` in the ordinary test helper would otherwise
  restore the locked Qiskit release after the weekly session installs Qiskit
  main. The upstream session must exclude Qiskit from that project-install sync
  so the headers used to build the candidate and the package used by tests stay
  identical.

## Decision Log

- Decision: Keep the Qiskit bridge in the existing MLIR nanobind extension and
  use a private, Qiskit-type-free C++ facade between shared translation and
  minor-specific access. Rationale: this confines the experimental ABI and
  compiles semantic translation once. Date/Author: 2026-08-09 / Codex.
- Decision: Represent adapters as one object target per supported Qiskit minor,
  with `QISKIT_PYTHON_EXTENSION` and the vendored minor include directory set
  privately. Rationale: Qiskit promises patch compatibility within a minor but
  permits minor ABI changes, and the C-API table must exist in exactly one
  translation unit per adapter. Date/Author: 2026-08-09 / Codex.
- Decision: Parse `qiskit.__version__` as a final Python package release before
  creating an adapter, then require the capsule-reported major/minor to match
  and the release status to be final. Rationale: unsupported packages must be
  rejected before capsule access, with no override that could opt into an unsafe
  ABI. Date/Author: 2026-08-09 / Codex.
- Decision: Preserve Qiskit bit identity in canonical top-level bit arrays and
  attach private module metadata describing ordered register membership,
  including overlapping registers and loose bits. Rationale: MLIR resources need
  one stable identity per bit, while Qiskit registers are named views that may
  overlap. Date/Author: 2026-08-09 / Codex.
- Decision: Model bare Qiskit parameters as `f64` entry-point arguments with a
  private name attribute and reject composite parameter strings. Rationale: this
  preserves values through MLIR and permits direct flat export without
  introducing a second symbolic-expression IR. Date/Author: 2026-08-09 / Codex.
- Decision: Add native dense-unitary QC and QCO operations rather than
  decomposing through Python or OpenQASM. Rationale: decomposition would either
  violate the direct C-API boundary or silently alter the requested operation; a
  matrix attribute is explicit, verifiable, and reusable by the existing QCO
  unitary interface. Date/Author: 2026-08-09 / Codex.
- Decision: Preflight the complete QC module before allocating a Qiskit output
  circuit. Rationale: export failures for structured control, unsupported
  expressions, or invalid metadata must not return or leak a partially built
  circuit. Date/Author: 2026-08-09 / Codex.

## Outcomes & Retrospective

The implementation ships one private adapter for final `>=2.5.0,<2.6.0`,
selected before capsule access and verified again through `qk_api_version()`.
The legacy Qiskit converters and broad optional dependency remain unchanged.
Import covers the fixed standard-gate surface, dense unitaries, global phase,
registers and loose bits, measurement/reset/barrier, bare parameters, supported
typed classical expressions, and nested if/while/for/switch. Export covers the
flat constructible subset and fails before allocating a native circuit for
unsupported structured or classical execution and non-constructible register
layouts.

Validation completed on Qiskit 2.5.1 with LLVM/MLIR 22.1.3 and Python 3.14:

- 317 QC dialect tests, 460 QCO dialect tests, and 3 QC/QCO round-trip tests
  passed.
- 91 installed-wheel Python MLIR/bridge tests passed. The focused 63-test bridge
  matrix also passed in both shipping and exact candidate-adapter modes.
- The stable-ABI release wheel and nanobind stub session passed. The first run
  exposed and led to the limited-ABI string conversion fix.
- The Sphinx HTML build passed with nitpicky and warnings-as-errors enabled,
  including execution of the new Qiskit compiler example. An initial retry was
  needed because the host Python did not discover a CA bundle for the external
  QDMI tag file; using the bundle installed in the nox environment resolved it.
- All repository hooks and `git diff --check` on the non-vendored diff passed.
  The eight-header count, every per-header provenance hash, the candidate
  template, and the maintenance API-surface extractor passed separately; the
  provenance checks preserve the upstream headers byte-for-byte, including their
  trailing whitespace.

No commit, push, pull request, or remote mutation was performed. The resulting
reviewable diff remains on `agent/qiskit-capi-bridge` in this worktree.

## Context and Orientation

`bindings/mlir/register_mlir.cpp` owns the Python-facing `QCProgram` bindings
and the input dispatcher used by `compile_program`. The current Qiskit branches
there call the legacy Python loader. New facade declarations and the shared
translator belong under `bindings/mlir/qiskit/`; only their MQT-owned entry
points are included from `register_mlir.cpp`.

`bindings/mlir/CMakeLists.txt` creates the `mqt.core._mlir` extension. It will
create a private adapter object target for Qiskit 2.5 and add its object plus
the shared bridge source to that extension. Vendored headers live under
`vendor/qiskit-c-api/2.5.0/`, alongside a provenance manifest and the upstream
Apache-2.0 license; they are never installed.

`mlir/include/mlir/Dialect/QC/IR/QCOps.td` and
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` define reference- and value-
semantic quantum operations. The dense-unitary additions must implement their
respective `UnitaryOpInterface`, verify a finite square `2^n` matrix against the
qubit arity, and convert directly in both QC/QCO directions. Builder helpers
will let the shared importer construct the QC form without depending on
generated operation details.

`mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` and its implementation
construct a main function, quantum/classical storage, gates, structured SCF, and
returns. The bridge needs an input-typed initializer for parameter symbols, a
floating constant helper, and the dense-unitary helper. Existing SCF helpers
provide the structured-control skeleton; expression lowering uses ordinary
arith, math, and memref operations at the current insertion point.

`test/python/test_mlir.py` contains existing Qiskit compiler tests and the rest
of the MLIR Python suite. Focused bridge cases can live in a sibling module so
they can skip only when the installed Qiskit version has no adapter. Legacy
converter tests remain untouched. C++ dialect tests cover dense-unitary
verification and QC/QCO conversion independently of Python or Qiskit.

`noxfile.py` defines the `qiskit` upstream session invoked by
`.github/workflows/upstream.yml`. That session currently installs Qiskit main
after the ordinary package setup. It must instead install the development
package before building, discover `qiskit.capi.get_include()`, enable a
temporary exact-version candidate adapter, and then run the ordinary tests. The
released-minor maintenance command will also live as a nox session backed by a
checked-in script so its header, manifest, comparison, generation, build, and
test steps are independently inspectable.

## Plan of Work

First, copy the eight verified 2.5.0 C-API headers and license from the official
wheel into the private vendor directory and add a provenance document containing
the source URL, wheel hash, embedded Qiskit version, file list, and file hashes.
Wire one private object target whose sole source is `Adapter25.cpp`; give only
that target the vendored include directory and `QISKIT_PYTHON_EXTENSION`.

Second, define the MQT-owned facade as narrow callback/view interfaces for a
circuit, one instruction, one parameter, one register, one expression node, and
one control-flow operation. The 2.5 adapter maps C-API enums and borrowed
handles to these normalized values, owns every required clear/free call, and
does not accumulate a complete circuit model. The dispatcher imports Python's
`qiskit` module only when a bridge method is called, parses a final release,
selects 2.5 exactly, then lets that adapter call `qk_import()` and verify
`qk_api_version()`.

Third, extend QC and QCO with dense-unitary operations. Use MLIR dense complex
attributes as row-major matrix storage, variadic qubit operands/results, and
operation verifiers for nonempty finite power-of-two dimensions. Teach the
QC/QCO converters, builders, unitary utilities, canonical checks, and textual
tests about the new operations before relying on them in the Python bridge.

Fourth, implement shared import. Allocate canonical quantum and classical bit
storage; retain register membership as private module attributes; create entry
arguments for bare symbols; translate standard instructions, modifier flags,
dense unitaries, global phase, barriers, reset, and measurement one instruction
at a time. Recursively lower supported expression nodes and nested if/while/
for/switch blocks with Qiskit's block-local bit maps. Validate integer widths,
types, parameters, custom instructions, delays, boxes, and loop-control
instructions at their semantic boundary with actionable diagnostics.

Fifth, implement buffered export. Preflight a copy of the module and recover
static bit identities, register metadata, parameter arguments, global phase, and
the flat supported operation sequence. Reject SCF and unsupported classical
execution. Only after preflight succeeds, allocate a `QkCircuit`, add registers,
parameters, standard/modifier gates, unitaries, measurements, reset, barriers,
and phase through the adapter, convert its `CircuitData` to the public Qiskit
`QuantumCircuit`, and return the Python object.

Sixth, replace only the compiler-specific Python routes in `register_mlir.cpp`,
add the non-consuming export method and an adapter-support query used by tests,
regenerate stubs, and document the experimental support window and failure
behavior. Add the maintenance/update script and adjust the weekly nox session so
upstream headers are available before MQT's build and a non-shipping candidate
source is compiled for the exact development version.

Finally, run generated-file checks, focused dialect and conversion tests,
focused Qiskit bridge tests, the complete Python MLIR suite, the release build,
repository lint, and diff checks. Inspect the final status and diff, recording
all successes and environmental failures here. Do not push or open a pull
request without separate authorization.

## Concrete Steps

Run all commands from the task worktree. Use `./.agent/run.sh` for environment
and build commands.

1. Populate `vendor/qiskit-c-api/2.5.0/` from the verified released wheel and
   validate the recorded SHA-256 hashes.
2. Add the Qiskit facade, dispatcher, 2.5 adapter, shared translator, and CMake
   object wiring; compile a minimal extension before adding semantic breadth.
3. Add and test `qc.unitary` and `qco.unitary`, then add their direct conversion
   patterns and builder entry point.
4. Implement flat import/export first and run focused Python tests.
5. Add classical expression and nested structured-control import, with one
   representative success and failure test per boundary.
6. Add maintenance automation, upstream candidate mode, docs, changelog, and
   generated stubs.
7. Execute the final validation sequence and update this document with exact
   target names, test counts, and any remaining limitations.

## Validation and Acceptance

Acceptance requires the MQT Python extension to import without importing or
requiring Qiskit; the legacy converters to retain their broad dependency and
behavior; final Qiskit 2.5.x to support direct import, compilation, and flat
export; and unsupported/prerelease versions to fail before capsule access with
their installed version and the supported range in the message.

The focused Python tests must cover table-driven standard gates and modifiers,
register membership including overlap and loose bits, global phase,
measurement/reset/barrier, numeric and bare parameters, a dense unitary, nested
if/while/for/switch import, flat export round trips, representative unsupported
operations/expressions/types/integers, unknown-minor dispatch, and import of the
binding without Qiskit. Existing MLIR and legacy converter tests must continue
to pass, with only compiler-bridge tests skipped under unsupported Qiskit.

Dialect tests must parse, print, verify, and convert dense unitary operations in
both directions. The normal release build, affected C++ unit tests, Python MLIR
suite, stub generation, changed-file formatting/static checks, repository lint,
and `git diff --check` on non-vendored files must pass or have a precisely
recorded external blocker. Byte-identical vendored files are validated against
their provenance hashes instead.

## Idempotence and Recovery

The vendor/update command stages no commits and overwrites only the version
directory named by its explicit argument after validating a released wheel and
using a temporary directory. Re-running it for the same version must reproduce
identical headers and provenance. If comparison, compilation, or tests fail, it
must leave the repository diff for review and report the adapter as
unregistered.

Builds and tests are local to this worktree. Generated adapter candidates live
in the build tree or an explicit temporary directory and are never installed. No
step mutates `base/`, another worktree, a remote branch, or GitHub state. Source
changes can be inspected and repaired incrementally without destructive Git
commands.

## Artifacts and Notes

Official Qiskit 2.5.0 wheel selected for vendoring:

    https://files.pythonhosted.org/packages/fc/b8/d13df60c5264f74529ecbfd4c9b827c34b89fb1f0e2a02bf2990d6010a94/qiskit-2.5.0-cp310-abi3-macosx_11_0_arm64.whl
    SHA-256: 896d24f564d5192ccdad26d9628251942155cc637059e2dbb24df891a56567de

The wheel contains the eight headers `qiskit.h`, `qiskit/attributes.h`,
`qiskit/complex.h`, `qiskit/funcs.h`, `qiskit/funcs_py.h`,
`qiskit/funcs_py_generated.h`, `qiskit/types.h`, and `qiskit/version.h`.

## Interfaces and Dependencies

The binding-facing API will expose functions equivalent to:

    bool isQiskitCompilerBridgeAvailable(nb::handle circuitOrModule);
    compiler::QCProgram importQiskitCircuit(nb::handle circuit);
    nb::object exportQiskitCircuit(const compiler::QCProgram& program);

The private dispatcher owns Python version inspection and returns a
Qiskit-type-free adapter interface. The adapter facade uses standard C++ and
nanobind/Python handles only; no public MQT header mentions `QkCircuit`,
`QkParam`, or any vendored declaration.

The bridge depends only on libraries already linked into the MLIR binding plus
the QC/QCO builder and translation targets required for its direct operations.
Only the 2.5 adapter object target depends on the vendored headers and Python's
extension-module compile interface. No installed target, exported CMake target,
or package metadata gains a Qiskit build dependency.
