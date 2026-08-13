# Add convenient framework adapters for registered and Slurm-selected QDMI devices

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core registers QDMI devices under stable string identifiers, but users still
need to assemble framework adapters manually. This change makes the stable ID
the common entry point for Qiskit backends and MLIR compiler targets. It also
lets a program inside a Slurm allocation obtain Qiskit, PennyLane, and compiler
adapters for the allocated device without parsing the license environment or
opening a second device session.

After the change, a user can call `QDMIBackend.from_device_id`, then construct a
sampler or estimator from that backend. A user can also call
`CompilerTarget.from_device_id`. Inside a Slurm job, the functions
`slurm.qiskit_backend`, `slurm.pennylane_device`, and `slurm.compiler_target`
adapt the one device selected by the license environment.

## Progress

- [x] (2026-08-13 19:00Z) Verified the new worktree starts at the exact reviewed
      head of PR #2043.
- [x] (2026-08-13 19:15Z) Added public typed dictionaries, the Python 3.10
      compatibility dependency, and typed stable-ID keyword arguments.
- [x] (2026-08-13 19:18Z) Preserved known stable IDs on Qiskit backends without
      changing direct construction.
- [ ] Make Qiskit provider discovery lazy and resilient.
- [ ] Add backend factories for Qiskit primitives and simplify primitive
      constructors.
- [ ] Add the MLIR compiler-target stable-ID factory and regenerate stubs.
- [ ] Add Slurm framework shortcuts, focused tests, and documentation.
- [ ] Run focused and aggregate validation, review the complete diff, and
      publish the stack.

## Surprises & Discoveries

- Observation: `QDMIProvider` currently opens every registered device in its
  constructor and silently skips failures. Evidence:
  `python/mqt/core/plugins/qiskit/provider.py` stores a populated `_backends`
  list in `__init__`.
- Observation: QDMI session fields include path-like values for `auth_file` and
  `device_config_file`, while job custom fields accept strings, booleans,
  floats, or `None`. Evidence: the generated signatures in
  `python/mqt/core/qdmi/driver.pyi` and `python/mqt/core/qdmi/__init__.pyi`.
- Observation: The real Slurm test image installs the wheel without optional
  Python dependencies. Evidence: `.github/workflows/slurm.yml` and
  `test/slurm/Dockerfile` install the wheel with `--no-deps`.

## Decision Log

- Decision: Keep session and job parameters in separate total-false `TypedDict`
  classes. Rationale: both namespaces contain `custom1` through `custom5`, but
  they are consumed at different lifecycle points and have different value
  types. Date/Author: 2026-08-13, Codex.
- Decision: Use `Unpack[QDMISessionParameters]` for stable-ID factories and
  retain mappings for the two PennyLane parameter namespaces. Rationale:
  keywords give callers completion and reject misspelled fields statically,
  while mappings keep the overlapping PennyLane namespaces explicit.
  Date/Author: 2026-08-13, Codex.
- Decision: Keep Slurm sampler and estimator construction fluent through
  `slurm.qiskit_backend()`. Rationale: direct Slurm primitive factories would
  duplicate backend methods without shortening the normal path materially.
  Date/Author: 2026-08-13, Codex.
- Decision: Resolve the Slurm environment and open the device once per shortcut
  call. Rationale: the environment selects the allocation device, and each
  returned adapter must own the same fresh session used for validation.
  Date/Author: 2026-08-13, Codex.

## Outcomes & Retrospective

Implementation is in progress. This section will record the completed behavior,
validation evidence, and any remaining external CI work.

## Context and Orientation

QDMI is the Quantum Device Management Interface. MQT Core exposes its device
registry and opened-device handles through the native `mqt.core.qdmi` module.
`python/mqt/core/plugins/qiskit` adapts an opened handle to Qiskit.
`python/mqt/core/plugins/pennylane` adapts it to PennyLane. The native
`mqt.core.mlir.CompilerTarget` class snapshots a device for compilation.

The stable registry API is declared in `python/mqt/core/qdmi/driver.pyi` and
implemented by `bindings/qdmi/qdmi.cpp`. The Qiskit backend factory is in
`python/mqt/core/plugins/qiskit/backend.py`. Provider discovery is in
`python/mqt/core/plugins/qiskit/provider.py`, and primitive implementations are
in `sampler.py` and `estimator.py`. PennyLane construction is in
`python/mqt/core/plugins/pennylane/device.py`.

Slurm selection is split between `src/fomac/Slurm.cpp`, which parses and
validates the allocation license, and `bindings/qdmi/slurm.cpp`, which exposes
Python functions. The selector must retain its existing grammar and accept only
a single local license with quantity one and device state `IDLE` or `BUSY`.
Selection does not authorize access; persistent registration and provider
credentials remain authoritative.

Generated `.pyi` files under `python/mqt/core/qdmi` and
`python/mqt/core/mlir.pyi` must never be edited by hand. Binding signatures are
adjusted through `bindings/patterns.txt` and regenerated with the `stubs` Nox
session.

## Plan of Work

First, add `python/mqt/core/typing.py` with `QDMISessionParameters` and
`QDMIJobParameters`. Add `typing-extensions` only for Python 3.10 and update the
lock file. Change stable-ID Python factories to accept unpacked typed keywords.
Apply the two mapping types to PennyLane and retain explicit runtime validation.

Second, let `QDMIBackend` retain an optional stable ID. Its stable-ID factory
sets the ID, while direct construction leaves it unset. Third, remove eager
provider state. Each discovery call reads current registered IDs, opens fresh
backends, warns using only a failing ID, and continues. Exact-ID lookup opens
only the requested device and accepts typed session overrides.

Fourth, add `sampler` and `estimator` methods to `QDMIBackend`. Remove the
MQT-specific `options` arguments from the direct primitive constructors and make
estimator shots an explicit parameter. Record this API change in `UPGRADING.md`.

Fifth, add `CompilerTarget.from_device_id`. Move the C++ conversion of typed
session overrides into one binding helper shared by `open_device` and MLIR.
Regenerate recursive stubs and override the generated callable signature with a
pattern that uses `Unpack[QDMISessionParameters]`.

Sixth, factor Slurm selection into an internal resolver that returns the stable
ID and one opened handle from one environment snapshot. The existing open
function delegates to it. Native Python bindings lazily import only the adapter
requested by the caller. PennyLane receives a private constructor for an
already-open handle. Add focused Python tests, update the real Slurm SC job to
exercise only the compiler target, and extend `docs/qdmi/slurm.md` with all
public paths.

Keep these six units as signed commits in the order above. Each commit message
must include the repository's AI-assistance trailer. Do not modify the two lower
stack branches.

## Concrete Steps

Run all commands from the repository root through `.agent/run.sh` when they can
create caches. During implementation, use focused tests such as:

    ./.agent/run.sh uv run --no-sync pytest test/python/plugins/qiskit/test_backend.py
    ./.agent/run.sh uv run --no-sync pytest test/python/plugins/qiskit/test_provider.py
    ./.agent/run.sh uv run --no-sync pytest test/python/plugins/qiskit/test_sampler.py test/python/plugins/qiskit/test_estimator.py
    ./.agent/run.sh uv run --no-sync pytest test/python/plugins/qdmi_pennylane/test_device.py
    ./.agent/run.sh uv run --no-sync pytest test/python/test_mlir.py
    ./.agent/run.sh uv run --no-sync pytest test/python/qdmi/test_slurm.py

For native bindings, configure and rebuild the release preset, then regenerate
stubs:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./.agent/run.sh uvx nox -s stubs

At completion, run the supported Python 3.10 and 3.14 sessions, minimum-Qiskit
coverage, native selector tests, strict documentation, wheel inspection, full
lint, and whitespace validation. The GitHub real-Slurm workflow supplies the
cluster proof after publication.

## Validation and Acceptance

Type checking must accept every documented QDMI session and job field and flag
unknown fields. Runtime tests must prove that all accepted fields are forwarded
unchanged, unknown fields are rejected, and inline and file device configuration
remain mutually exclusive.

A backend created from a stable ID must report that ID. A backend created from
an opened handle without an explicit ID must report `None`. Provider creation
must open no device, subsequent registry changes must be visible, exact-ID
lookup must open one fresh session, and enumeration failures must emit a warning
containing only the stable ID before continuing.

Both Qiskit primitives must work through direct constructors and backend
methods. Defaults must be 1024 shots and zero estimator precision unless the
caller overrides them. Sampling and estimation must still execute against DDSIM.

`CompilerTarget.from_device_id("mqt.ddsim.default")` must equal the target made
from an explicitly opened DDSIM handle. Unknown and incompatible devices must
retain their existing errors.

Each Slurm shortcut must read one environment snapshot and open one device.
Qiskit and PennyLane shortcuts must execute against DDSIM in focused tests. The
compiler shortcut must match the direct target. Existing malformed, compound,
remote, non-unit, unknown, and unavailable-device errors must remain unchanged.
Missing optional dependencies must produce the established installation hint.

The aggregate branch is accepted when focused tests, recursive stubs, supported
test sessions, documentation, lint, wheel inspection, and `git diff --check`
pass locally or have a recorded environment boundary, and remote CI reaches a
terminal state for the exact published revision.

## Idempotence and Recovery

Formatting, stub generation, builds, and tests are repeatable. Keep all build
and tool caches within this worktree through `.agent/run.sh`. If a generator
changes unrelated output, inspect and exclude it rather than discarding user
work. If a lower pull-request head advances, rebase only this top branch onto
the new #2043 head and rerun affected checks. Never rewrite either reviewed
lower branch.

## Artifacts and Notes

The starting revision is `a0c21c5826f54fa25d2d16bac65b65c6b59efc58`, the
reviewed head of PR #2043. PR #2043 targets the branch of PR #2025. Publication
must preserve that immediate-parent topology and link all three pull requests
with the native GitHub stack workflow.

## Interfaces and Dependencies

The public Python types are `mqt.core.typing.QDMISessionParameters` and
`mqt.core.typing.QDMIJobParameters`. Stable-ID factories use
`Unpack[QDMISessionParameters]`, supplied by the standard library on Python 3.11
and newer and by `typing-extensions` on Python 3.10.

The final Qiskit backend constructor accepts an opened QDMI device, an optional
provider, and a keyword-only optional stable ID. It exposes `device_id`,
`sampler(default_shots=1024)`, and
`estimator(default_precision=0.0, default_shots=1024)`.

The provider exposes `device_ids`, `backends`, `get_backend`, and
`get_backend_by_device_id`. The compiler target exposes `from_device_id`. The
Slurm module exposes `open_device_from_license`, `qiskit_backend`,
`pennylane_device`, and `compiler_target`. Slurm shortcuts do not accept session
or credential overrides; only the PennyLane shortcut accepts typed job
parameters.

Revision note (2026-08-13): Created the implementation plan after verifying the
reviewed stack base and inspecting the existing adapter, registry, binding, and
Slurm boundaries.
