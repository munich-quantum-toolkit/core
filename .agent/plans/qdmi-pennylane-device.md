# Add a PennyLane interface for gate-based QDMI devices

Status: historical implementation record.

The public converter design was replaced by a device-owned converter; see the
[PennyLane audit](../audits/pennylane-plugin.md).

## Goal and scope

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

## Constraints

- The QDMI FoMaC Python API already exposes stable device IDs, fresh per-call
  sessions, program formats, topology, operation names, raw shots, and typed
  custom job slots. Evidence: `python/mqt/core/fomac.pyi` provides
  `registered_device_ids`, `open_device`, `Device.supported_program_formats`,
  `Device.operations`, `Device.coupling_map`, `Device.submit_job`, and
  `Job.get_shots`.

- MQT Core's existing OpenQASM 3 exporter includes `stdgates.inc`, while remote
  Braket programs need explicit device gate spellings and no include. The
  PennyLane integration therefore needs a small capability-driven emitter behind
  its own converted-program boundary.

- PennyLane 0.45.1 declares Python 3.14 support, but Autograd 1.8 contains a
  `return` in a `finally` block. MQT Core's warnings-as-errors test policy
  promotes Python 3.14's compile-time `SyntaxWarning` to an import failure.
  Evidence: the first Python 3.14 nox run failed while importing
  `autograd.wrap_util`; an exact message/category warning filter restored normal
  upstream import behavior and the full session passed.

- DDSIM returns QDMI shots in conventional basis-state spelling, with the
  highest-index site on the left. Evidence: applying X to site zero returned the
  raw QDMI string `01`; the PennyLane device reverses that representation before
  selecting columns in declared wire order.

- generic exception names such as `TranslationError` conflict with the existing
  Qiskit plugin in Sphinx's global Python domain. Evidence: the warning-as-error
  docs build reported ambiguous cross-references. Public `PennyLane...Error`
  names preserve focused errors and produce clean API docs without changing the
  Qiskit integration.

- optional transitive dependencies must be imported after the Python-version
  guard in test modules. Evidence: importing NetworkX before the Python 3.10
  module skip prevented collection when the PennyLane extra was intentionally
  absent; moving the import after the guard restored the expected skip behavior.

- Finite-shot tests must leave margin between expected distributions and
  tolerance boundaries. A Bell probability at the old tolerance boundary exposed
  a false-failure risk.

## Decisions

- Keep every import of PennyLane in source, tests, and documentation spelled
  `import pennylane as qp`. Rationale: this is PennyLane's current project
  convention and avoids adding the retired `qml` alias.

- Prefer OpenQASM 3 by advertised format, and never retry the OpenQASM 2 path
  after a QASM3 translation failure. Rationale: advertised QASM3 support is a
  device contract; hiding a failed capability-driven conversion would make
  unsupported programs nondeterministic.

- Return raw computational-basis samples from `execute` and let PennyLane's
  preprocessing transforms reconstruct requested measurements. Rationale: the
  framework then owns observable diagonalization, non-commuting measurement
  splitting, Hamiltonian aggregation, shot-vector binning, and result typing.

- Keep QDMI submission sequential. Rationale: the current FoMaC interface has no
  batch-submission contract, while sequential execution remains correct for
  batches and parameter-shift expansion.

- Keep PennyLane enabled on Python 3.11 through 3.14 and suppress only Autograd
  1.8's exact Python 3.14 compile-time warning in test configuration. Rationale:
  PennyLane formally supports 3.14, normal imports work, and vendoring or
  patching Autograd would add unnecessary maintenance.

- Build documentation exclusively with the repository's `nox -s docs` session.
  Rationale: the nox session owns the complete native, AutoAPI, notebook, and
  strict-warning environment.

- Keep the complete QAOA demonstration in the executable MyST notebook and test
  the same application contract directly through DDSIM. Rationale: one source
  now defines the narrative, executable analysis, and visual results, while the
  smoke test remains compact and independent of rendered prose.

- Organize the device guide as a minimal Bell-state quickstart, followed by the
  conversion contract and a complete finite-shot MaxCut QAOA application.
  Rationale: device users first see the smallest executable interface and then
  the scientific application without internal development history.

- Use 1000, 1000, and 2000 shots with an absolute probability tolerance of 0.1
  for the Bell shot-vector test. Rationale: this still tests the expected 50/50
  distribution while making a stochastic CI failure negligible.

## Outcome and validation

The plugin executes finite-shot tapes through stable QDMI device IDs, prefers
advertised QASM3, and uses QASM2 only for QASM2-only devices. PennyLane owns
measurement reconstruction and parameter-shift expansion. The executable
documentation demonstrates one-layer MaxCut QAOA.

Supported Python and minimum-dependency plugin suites passed, including the Bell
shot-vector regression. Python 3.10 retained the intended optional-module skips.
Strict documentation executed the notebook; lint, stubs, and diff checks passed.

Routing, parallel submission, analytic execution, pulse programming, and non-
gate properties were excluded. The separate Braket integration depends on the
public stable-ID API. Later converter ownership is recorded in the linked audit.

## Code and ownership

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

## Acceptance

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

## Interfaces

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
