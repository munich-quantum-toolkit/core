# Add a PennyLane interface for gate-based QDMI devices

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

PennyLane applications should run unchanged on gate-based quantum devices
exposed through QDMI. After this change, a user who installs the optional
`mqt-core[pennylane]` extra can create the built-in simulator with
`qp.device("mqt.ddsim.default", wires=..., shots=...)`, evaluate ordinary
sampled measurements and parameter-shift gradients, and reuse the same circuit
with device integration packages that register stable QDMI device IDs.

Programs use OpenQASM 3 whenever the device advertises that format. A device
that advertises only OpenQASM 2 uses PennyLane's serializer after the normal
device preprocessing pipeline. The final documentation demonstrates a
finite-shot, one-layer MaxCut QAOA application on the local DDSIM QDMI device.

## Progress

- [x] (2026-08-04 09:00Z) Refreshed `origin/main`, created an isolated worktree,
      read repository policy, and surveyed the QDMI Python API and packaging.
- [x] (2026-08-04 11:30Z) Implemented the optional PennyLane package,
      QASM3-first negotiation, QASM2 fallback, modern preprocessing and
      execution, focused exceptions, and the stable DDSIM entry point.
- [x] (2026-08-04 12:45Z) Added converter, preprocessing, result, gradient,
  arbitrary-wire, batch, shot-vector, and DDSIM end-to-end tests.
- [x] (2026-08-04 13:30Z) Added the marked optional dependency, ordinary test
      and documentation group coverage, lockfile, changelog entry, QAOA
      application, and technical documentation.
- [x] (2026-08-04 15:45Z) Passed the Python 3.10 and 3.14 full test sessions,
  the focused Python 3.11-3.13 sessions, the warning-clean documentation nox
  session, and the complete lint nox session; audited the final diff.
- [x] (2026-08-04 16:30Z) Integrated the latest `origin/main`, broadened the
  exact Autograd warning filter to work for compile-time warnings in CI,
  simplified public prose and the changelog, and reduced the shared test
  support from 300 to 177 lines without reducing its 33-test coverage.
- [x] (2026-08-04 23:00Z) Replaced the standalone QAOA script and subprocess
      test with an executable scientific MyST notebook and a compact real-DDSIM
      smoke test; standardized device-oriented terminology.
- [x] (2026-08-04 23:10Z) Passed the forced and cached documentation sessions,
      inspected the rendered SVG figures, passed all 33 focused Python 3.14
      tests and the complete lint session, and completed the final repository
      searches.
- [x] (2026-08-04 23:15Z) Moved the NetworkX import behind the Python 3.10
      PennyLane guard identified during independent verification; the focused
      Python 3.10 and 3.14 nox sessions passed.
- [x] (2026-08-04 23:58Z) Integrated `origin/main` at `ecd32f734` with a signed
      merge commit; the merged changelog retains the #2005 and #2007 entries and
      links.
- [x] (2026-08-05 00:02Z) Addressed all eight review threads by reorganizing the
      notebook as a quickstart, compact conversion contract, and end-to-end QAOA
      application; removed explanatory implementation history; replaced repeated
      QNode return-type suppressions with file-wide Ruff directives.
- [x] (2026-08-05 00:03Z) Stabilized the Bell-state shot-vector test with
      partitions of 1000, 1000, and 2000 shots at absolute tolerance 0.1, and
      added the statistical-robustness rule to `AGENTS.md`.
- [x] (2026-08-05 00:08Z) Passed the minimums reproduction and focused Python
      3.10-3.14 sessions, forced and cached documentation builds,
      rendered-output inspection, complete lint, diff checks, and repository
      searches.
- [ ] Complete independent verification, publish the signed revision, monitor
      replacement CI, and reply to and resolve the eight addressed threads.

## Surprises & Discoveries

- Observation: The QDMI FoMaC Python API already exposes stable device IDs,
  fresh per-call sessions, program formats, topology, operation names, raw
  shots, and typed custom job slots. Evidence: `python/mqt/core/fomac.pyi`
  provides `registered_device_ids`, `open_device`,
  `Device.supported_program_formats`, `Device.operations`,
  `Device.coupling_map`, `Device.submit_job`, and `Job.get_shots`.
- Observation: MQT Core's existing OpenQASM 3 exporter includes `stdgates.inc`,
  while remote Braket programs need explicit device gate spellings and no
  include. The PennyLane integration therefore needs a small capability-driven
  emitter behind its own converted-program boundary.
- Observation: PennyLane 0.45.1 declares Python 3.14 support, but Autograd 1.8
  contains a `return` in a `finally` block. MQT Core's warnings-as-errors test
  policy promotes Python 3.14's compile-time `SyntaxWarning` to an import
  failure. Evidence: the first Python 3.14 nox run failed while importing
  `autograd.wrap_util`; an exact message/category warning filter restored normal
  upstream import behavior and the full session passed.
- Observation: DDSIM returns QDMI shots in conventional basis-state spelling,
  with the highest-index site on the left. Evidence: applying X to site zero
  returned the raw QDMI string `01`; the PennyLane device reverses that
  representation before selecting columns in declared wire order.
- Observation: generic exception names such as `TranslationError` conflict with
  the existing Qiskit plugin in Sphinx's global Python domain. Evidence: the
  warning-as-error docs build reported ambiguous cross-references. Public
  `PennyLane...Error` names preserve focused errors and produce clean API docs
  without changing the Qiskit integration.
- Observation: optional transitive dependencies must be imported after the
  Python-version guard in test modules. Evidence: importing NetworkX before the
  Python 3.10 module skip prevented collection when the PennyLane extra was
  intentionally absent; moving the import after the guard restored the expected
  skip behavior.
- Observation: the Python 3.12 minimums job sampled a Bell-state probability of
  exactly `0.65`, which lay on the floating-point boundary of the previous
  `0.5 ± 0.15` assertion. Evidence: CI run `30959612789`, job `92160445071`,
  passed 564 tests and failed only `test_bell_results_and_shot_vector` at that
  boundary.

## Decision Log

- Decision: Keep every import of PennyLane in source, tests, and documentation
  spelled `import pennylane as qp`. Rationale: this is PennyLane's current
  project convention and avoids adding the retired `qml` alias. Date/Author:
  2026-08-04 / GPT-5.6 via Codex.
- Decision: Prefer OpenQASM 3 by advertised format, and never retry the OpenQASM
  2 path after a QASM3 translation failure. Rationale: advertised QASM3 support
  is a device contract; hiding a failed capability-driven conversion would make
  unsupported programs nondeterministic. Date/Author: 2026-08-04 / GPT-5.6 via
  Codex.
- Decision: Return raw computational-basis samples from `execute` and let
  PennyLane's preprocessing transforms reconstruct requested measurements.
  Rationale: the framework then owns observable diagonalization, non-commuting
  measurement splitting, Hamiltonian aggregation, shot-vector binning, and
  result typing. Date/Author: 2026-08-04 / GPT-5.6 via Codex.
- Decision: Keep QDMI submission sequential. Rationale: the current FoMaC
  interface has no batch-submission contract, while sequential execution remains
  correct for batches and parameter-shift expansion. Date/Author: 2026-08-04 /
  GPT-5.6 via Codex.
- Decision: Keep PennyLane enabled on Python 3.11 through 3.14 and suppress only
  Autograd 1.8's exact Python 3.14 compile-time warning in test configuration.
  Rationale: PennyLane formally supports 3.14, normal imports work, and
  vendoring or patching Autograd would add unnecessary maintenance. Date/Author:
  2026-08-04 / GPT-5.6 via Codex.
- Decision: Build documentation exclusively with the repository's `nox -s docs`
  session. Rationale: the nox session owns the complete native, AutoAPI,
  notebook, and strict-warning environment. Date/Author: 2026-08-04 / GPT-5.6
  via Codex.
- Decision: Keep the complete QAOA demonstration in the executable MyST notebook
  and test the same application contract directly through DDSIM. Rationale: one
  source now defines the narrative, executable analysis, and visual results,
  while the smoke test remains compact and independent of rendered prose.
  Date/Author: 2026-08-05 / GPT-5.6 via Codex.
- Decision: Organize the device guide as a minimal Bell-state quickstart,
  followed by the conversion contract and a complete finite-shot MaxCut QAOA
  application. Rationale: device users first see the smallest executable
  interface and then the scientific application without internal development
  history. Date/Author: 2026-08-05 / GPT-5.6 via Codex.
- Decision: Use 1000, 1000, and 2000 shots with an absolute probability
  tolerance of 0.1 for the Bell shot-vector test. Rationale: this still tests
  the expected 50/50 distribution while making a stochastic CI failure
  negligible. Date/Author: 2026-08-05 / GPT-5.6 via Codex.

## Outcomes & Retrospective

The MQT Core implementation is complete. `qp.device("mqt.ddsim.default", ...)`
executes finite-shot tapes through QDMI, prefers capability-driven OpenQASM 3,
uses PennyLane's exact QASM2 serializer only for QASM2-only devices,
reconstructs the documented measurement types through PennyLane preprocessing,
and runs the one-layer MaxCut QAOA application documented in the executable
notebook.

Validation evidence:

- `nox -s tests-3.14`: 558 passed, 3 skipped.
- `nox -s tests-3.10`: 526 passed, 6 skipped; only the three PennyLane-dependent
  modules are newly skipped.
- `nox -s tests-3.11 tests-3.12 tests-3.13 -- test/python/plugins/qdmi_pennylane`:
  33 passed in every session.
- `nox -s docs`: strict HTML build succeeded with no warnings.
- `nox -s lint`: every repository hook, including lock validation, Ruff, ty,
  Markdown, and repository policy, passed.
- `nox -s docs -- -D nb_execution_mode=force`: the new notebook executed in 4.99
  seconds; the final cached documentation build also succeeded.
- Rendered-output inspection: the deterministic input graph and the three-panel
  analysis figure have readable labels, an explicit theme-neutral background,
  and distinct cut-edge styling.
- `nox -s tests-3.14 -- test/python/plugins/qdmi_pennylane`: 33 passed.
- `nox -s tests-3.10 tests-3.14 -- test/python/plugins/qdmi_pennylane`: Python
  3.10 reported 1 passed and 3 skipped; Python 3.14 reported 33 passed.
- `nox -s minimums-3.12 -- test/python/plugins/qdmi_pennylane/test_ddsim.py`: 10
  passed, reproducing the previously failing CI dependency boundary.
- `nox -s tests-3.10 tests-3.11 tests-3.12 tests-3.13 tests-3.14 -- test/python/plugins/qdmi_pennylane`:
  Python 3.10 reported 1 passed and 3 skipped; every supported Python version
  reported 33 passed.
- `nox -s docs -- -D nb_execution_mode=force`: strict HTML build succeeded and
  executed the PennyLane notebook in 2.77 seconds; the normal cached
  documentation session also succeeded.
- Rendered-output inspection: both notebook SVGs have readable and unclipped
  labels, a theme-neutral figure background, an explicitly noisy objective
  title, and distinct cut-edge styling.
- `nox -s lint`: all repository hooks passed after applying their canonical
  Markdown and Ruff formatting.
- `git diff --check` and the repository searches passed; there are no standalone
  QAOA-script references, `qml` aliases, QDMI-provider terms, repeated inline
  QNode suppressions, or review-identified iteration phrases.

The deliberately deferred gaps are routing, parallel submission, analytic
execution, pulse programming, and non-gate device properties. The separate
Amazon Braket device integration consumes only the public stable-ID device API
and remains ordered after the MQT Core release.

## Context and Orientation

`python/mqt/core/fomac.pyi` describes the Python-facing QDMI API. A stable
device ID is an implementation-independent string registered by MQT Core or an
installed device integration package. `fomac.open_device` opens a fresh device
session, and the returned object advertises program formats, named operations,
operation loci, topology, and a `submit_job` method. A submitted job provides
raw shot strings.

`python/mqt/core/plugins/qiskit/` is the neighboring optional integration. The
new sibling package is `python/mqt/core/plugins/pennylane/`. Its public
`QDMIDevice` derives from PennyLane's modern `Device` base class. A
`ConvertedProgram` value carries the text payload, selected QDMI
`ProgramFormat`, deterministic wire mapping, and measurement order so a future
compiler or JEFF converter can replace the text emitters without changing the
device API.

PennyLane preprocessing converts a user circuit, called a quantum tape, into one
or more executable tapes and a postprocessing function. The device uses
framework transforms to validate wires and shots, defer measurements, split
non-commuting observables, diagonalize them, decompose unsupported operations,
expand broadcasts, and replace measurements with raw sampling. Execution submits
each resulting tape in order and returns the raw sample arrays expected by the
transform postprocessor.

The built-in DDSIM QDMI registration uses `mqt.ddsim.default`. The package
exposes that stable ID through a PennyLane device entry point with the same
name. Device integration packages can register thin subclasses under their own
stable QDMI IDs and translate device credentials into the generic
`session_parameters` and `job_parameters` mappings.

## Milestones

### Milestone 1: Prove conversion and preprocessing

Add typed exceptions, `ConvertedProgram`, an operation table, alias resolution,
format negotiation, and a modern `CompilePipeline`. Exact tests must show that
QASM3 is chosen over QASM2, uses device-advertised spellings, emits one qubit
and classical register followed by a whole-register measurement, and includes no
standard-library include, definitions, pragmas, or modifiers. QASM2-only tests
must exercise `qp.to_openqasm` with rotations disabled and fail clearly when
serialization is impossible.

### Milestone 2: Execute through QDMI

Implement `QDMIDevice.execute` with finite-shot validation, deterministic wire
mapping, topology validation, bound finite parameter checks, sequential
submission, wait and failure handling, and conversion of QDMI shot strings to
PennyLane raw sample arrays. Tests must cover arbitrary wire labels, batches,
shot vectors, samples, counts, probabilities, expectations, variances,
Hamiltonians, and parameter-shift gradients.

### Milestone 3: Package and demonstrate DDSIM

Register `mqt.ddsim.default`, add a Python-version-marked optional dependency,
include it in the existing test and documentation groups for Python 3.11 and
newer, and keep Python 3.10 importable without PennyLane. Add an executable MyST
notebook containing a four-node, one-layer MaxCut QAOA example and scientific
visual analysis. Execute the full example locally and ensure the ordinary
existing Python and wheel CI matrices exercise the optional integration without
adding a workflow.

### Milestone 4: Validate and hand off the device boundary

Run focused tests, a complete Python test session on a supported version,
documentation, lint, lockfile consistency, and packaging checks. Audit all new
PennyLane source and documentation for the `qp` alias and review the final diff.
The separate Amazon Braket device integration then consumes only this public
API; it is not part of the MQT Core change.

## Plan of Work

Create `exceptions.py`, `converter.py`, `device.py`, and `__init__.py` below
`python/mqt/core/plugins/pennylane`. The converter will use a typed table whose
rows associate PennyLane operation types with semantic alias groups, arity, and
parameter counts. It will resolve one advertised QDMI spelling for every
operation, validate the operation's wire tuple against advertised sites or site
pairs and the device topology, format finite numeric parameters
deterministically, and emit the minimal OpenQASM 3 program. It will invoke
`qp.to_openqasm` only when QASM3 is not advertised and QASM2 is.

The device will normalize wires with PennyLane's `Wires`, open its QDMI device
by stable ID plus session mappings, cache capability metadata, expose a
framework preprocessing pipeline, and execute transformed tapes one at a time.
It will preserve QDMI's whole-register bit ordering while remapping to the
declared PennyLane wire order. Custom session and job parameters will be
validated against the named FoMaC fields before forwarding.

Add unit tests under `test/python/plugins/qdmi_pennylane` with small mock QDMI
objects, plus DDSIM end-to-end coverage that runs only when the built native
device is available. Add the PennyLane entry point and optional dependency in
`pyproject.toml`, update `uv.lock`, add the documentation intersphinx mapping
and toctree entry, add the executable QAOA notebook, and record the new feature
in `CHANGELOG.md`.

## Concrete Steps

Run all commands from the repository root. Install and build the supported
Python environment with:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev \
      --no-build-isolation-package mqt-core

Iterate on the new tests with:

    ./.agent/run.sh uv run --no-sync pytest \
      test/python/plugins/qdmi_pennylane

Validate the executable notebook through the documentation session, then run the
supported test and lint sessions:

    ./.agent/run.sh uvx nox --non-interactive -s docs -- \
      -D nb_execution_mode=force
    ./.agent/run.sh uvx nox -s tests-3.14 -- \
      test/python/plugins/qdmi_pennylane
    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint

Finally run `git diff --check`, search every added PennyLane file for forbidden
`qml` imports, inspect `git status --short`, and update this plan with concise
evidence.

## Validation and Acceptance

Exact converter tests must show QASM3 preference and device-advertised spellings
for the documented gate subset. They must show a final `c = measure q;`,
deterministic wire order, finite bound numbers, and no include, custom
definition, pragma, or gate modifier. A QASM3 translation failure must remain
visible even when QASM2 is also advertised. QASM2-only tests must show the
expected OpenQASM 2 header, `qelib1.inc`, operations, and whole-register
measurements with each observable rotation applied once.

Preprocessing and mock execution tests must cover raw samples, counts,
probabilities, expectation values, variances, Hamiltonians, shot vectors,
arbitrary wire labels, batches, topology errors, unsupported formats,
unsupported operations, analytic shots, and parameter-shift expansion. DDSIM
tests must create `qp.device("mqt.ddsim.default", ...)`, produce a Bell
distribution within finite-shot tolerance, compute a finite parameter-shift
gradient, and run the QAOA smoke test. The executable notebook must construct
the same local application and render the input graph, noisy objective
trajectory, sampled distribution, and best observed partition.

Python 3.10 must resolve and import the base package without PennyLane.
Supported Python 3.11 through 3.14 environments must install the test group and
run the new tests through the ordinary matrix. Documentation and lint must pass.
A repository search over all added source, test, and documentation files must
find `import pennylane as qp` and no alternative PennyLane import alias.

## Idempotence and Recovery

All source edits, lock generation, builds, and tests are repeatable. Tests use
mock devices or the local DDSIM device and require no credentials. If a partial
environment installation fails, rerun the same `.agent/run.sh uv sync` commands;
do not use shared caches outside this worktree. If generated metadata changes
unexpectedly, inspect `pyproject.toml` and regenerate `uv.lock` rather than
editing resolved records by hand. Do not alter another task's worktree.

## Artifacts and Notes

Review-thread mapping for the 2026-08-04 requested-changes review:

- Removed the installation-workflow comparison and related iteration language.
- Stated unsupported execution modes as out of scope for now.
- Removed prose about plotting configuration and hid only rendering setup.
- Presented the Bell-state example as the quickstart.
- Replaced the serializer call and rotation details with one QASM2 fallback
  sentence.
- Presented finite-shot MaxCut QAOA as the end-to-end use case.
- Retained the graph definition without explaining the fixed node positions.
- Replaced repeated inline QNode return-type suppressions with file-wide
  `ANN202` directives in both test modules.

The user authorized a non-force push of the signed remediation revision and
individual disclosed replies and resolutions for these eight threads. This
authorization does not include changing the review state, merging the pull
request, or unrelated metadata mutations.

## Interfaces and Dependencies

`mqt.core.plugins.pennylane.QDMIDevice` accepts `device_id`, `wires`,
`shots=1024`, `session_parameters`, and `job_parameters`. The mappings use the
FoMaC session and job keyword names from `python/mqt/core/fomac.pyi`.

`mqt.core.plugins.pennylane.ConvertedProgram` contains `payload: str`,
`program_format: fomac.ProgramFormat`, `wire_map`, and `measurement_order`.
`convert_program` accepts a preprocessed PennyLane `QuantumScript`, an opened
QDMI device, and declared device wires, and returns that value or raises a
focused translation, validation, configuration, or execution exception.

The optional dependency is `pennylane>=0.45.1,<0.46; python_version >= "3.11"`.
The implementation uses PennyLane's public device, transform, tape, wire,
measurement, and serialization APIs, NumPy for sample arrays, and the existing
`mqt.core.fomac` binding. It does not depend on a device-specific SDK, compiler
pipeline, JEFF conversion, pulse programming, neutral-atom properties, or
parallel job submission.

Revision note: created on 2026-08-04 to capture the user-approved MQT Core half
of the QDMI–PennyLane implementation before production edits.

Revision note: updated on 2026-08-05 to consolidate the QAOA demonstration into
an executable scientific notebook, standardize QDMI device terminology, and
record the independent Python 3.10 compatibility verification.

Revision note: updated on 2026-08-05 to integrate `origin/main`, map and address
the requested-changes review, and record the statistically robust Bell-state
test parameters and validation plan.
