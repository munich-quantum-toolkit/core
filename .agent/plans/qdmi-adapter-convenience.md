# Add convenient framework adapters for registered QDMI devices

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core registers QDMI devices under stable string identifiers, but users still
need to assemble framework adapters manually. This change makes the stable ID
the common entry point for Qiskit backends and MLIR compiler targets.

After the change, a user can call `QDMIBackend.from_device_id`, then construct a
sampler or estimator from that backend. A user can also call
`CompilerTarget.from_device_id`.

## Progress

- [x] (2026-08-13 19:00Z) Verified the new worktree starts at the exact reviewed
      head of PR #2043.
- [x] (2026-08-13 19:15Z) Added public typed dictionaries, the Python 3.10
      compatibility dependency, and typed stable-ID keyword arguments.
- [x] (2026-08-13 19:18Z) Preserved known stable IDs on Qiskit backends without
      changing direct construction.
- [x] (2026-08-13 19:22Z) Made provider discovery lazy, current-registry based,
      exact-ID addressable, and resilient to unavailable devices.
- [x] (2026-08-13 19:27Z) Added backend primitive factories and replaced the
      MQT-specific options mappings with explicit defaults.
- [x] (2026-08-13 19:34Z) Added the MLIR compiler-target stable-ID factory with
      shared native session conversion and a typed stub pattern.
- [x] (2026-08-13 20:35Z) Completed focused and aggregate validation, including
      the full supported Python and minimum-dependency matrices, native tests,
      strict docs, recursive stubs, lint, and wheel inspection; then reviewed
      the complete six-commit diff.
- [x] (2026-08-13 20:46Z) Published PR #2084 against `agent/slurm-integration`
      after re-verifying both lower stack heads and bases.
- [x] (2026-08-13 21:20Z) Addressed review feedback by moving shared session
      conversion out of the bindings tree and withdrawing the proposed Slurm
      framework shortcuts.
- [x] (2026-08-13 22:58Z) Rebased the six commits onto `main`, reran affected
      validation, published with an exact lease, and monitored the revised head
      to terminal CI except for the external documentation status.
- [x] (2026-08-14 09:12Z) Addressed the second review by defining parameter
      types in `mqt.core.typing`, replacing the native import hook with real
      aliases, installing the shared driver header, and rerunning affected
      validation.
- [x] (2026-08-14 11:30Z) Centralized lazy provider enumeration so exact-name
      lookup stops after its match, expanded compiler-target equivalence to all
      exposed metadata, and adopted the Scientific Python compatibility-module
      pattern for `Unpack` with verified wheel contents.
- [x] (2026-08-14 13:05Z) Simplified provider enumeration without Ruff
      suppressions, kept runtime path annotations introspectable, removed the
      redundant QDMI type re-exports, and documented the primitive migration.

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
- Observation: PR #2043's Read the Docs build is terminal-cancelled according to
  the Read the Docs API, but its GitHub status callback remains stale as
  pending. All other lower-stack checks are terminal and successful, including
  the real Slurm integration workflow.

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
- Decision: Keep domain-specific public types in `mqt.core.typing`, but isolate
  the version-dependent `Unpack` import in `mqt.core._compat.typing` and list
  only that private bridge in Ruff's `typing-modules`. Rationale: Ruff reserves
  this setting for modules that re-export `typing` or `typing_extensions`
  members; treating the public runtime `TypedDict`s as typing primitives would
  be incorrect. Date/Author: 2026-08-14, Codex.
- Decision: Keep the narrow Ruff suppression on the runtime `os` import in
  `mqt.core.typing`. Rationale: `typing.get_type_hints` resolves the public
  `os.PathLike` annotations at runtime; moving the import behind `TYPE_CHECKING`
  raises `NameError`, while hiding it behind a private alias leaks that alias
  into generated API documentation. Date/Author: 2026-08-14, Codex.
- Decision: Keep the Slurm adapter limited to `open_device_from_license` in this
  PR. Rationale: review requested that the proposed framework-specific Slurm
  shortcuts be reconsidered separately. Date/Author: 2026-08-13, Codex.
- Decision: Keep `TypedDict` and `Unpack` instead of using data classes.
  Rationale: `Unpack` requires a `TypedDict`, and unpacked keyword arguments are
  the convenient API this change provides. Date/Author: 2026-08-14, Codex.
- Decision: Keep direct primitive constructors compatible, but document backend
  factories as the canonical path. Rationale: removing public concrete Qiskit
  primitive classes is a separate breaking API decision and requires an explicit
  deprecation plan. Date/Author: 2026-08-14, Codex.
- Decision: Export the parameter dictionaries only from `mqt.core.typing`.
  Rationale: a second path through the native `mqt.core.qdmi` module adds
  binding and stub machinery without improving discoverability or compatibility
  for this unreleased API. Date/Author: 2026-08-14, Codex.
- Decision: Silently skip registered devices that open successfully but cannot
  be represented as Qiskit backends. Rationale: incompatibility is an expected
  result of enumerating a framework-independent registry, not an availability
  failure that should warn users. Date/Author: 2026-08-14, Codex.

## Outcomes & Retrospective

The six commits implement the typed configuration layer, stable backend
identity, lazy provider discovery, fluent primitives, and stable-ID compiler
targets; the proposed Slurm shortcuts were withdrawn during review. Focused
tests and the full Python 3.10 through 3.14 current/minimum dependency matrices
pass. Native Slurm and compiler tests, recursive stubs, strict documentation,
full lint, a local ABI3 wheel build, wheel-content inspection, and whitespace
validation also pass. PR #2084 now targets `main` because both lower pull
requests merged.

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

Generated `.pyi` files under `python/mqt/core/qdmi` and
`python/mqt/core/mlir.pyi` must never be edited by hand. Binding signatures are
adjusted through `bindings/patterns.txt` and regenerated with the `stubs` Nox
session.

## Plan of Work

First, define `QDMISessionParameters` and `QDMIJobParameters` in the public
`mqt.core.typing` module. Add `typing-extensions` only for Python 3.10 and
update the lock file. Change stable-ID Python factories to accept unpacked typed
keywords. Apply the two mapping types to PennyLane and retain explicit runtime
validation.

Second, let `QDMIBackend` retain an optional stable ID. Its stable-ID factory
sets the ID, while direct construction leaves it unset. Third, remove eager
provider state. Each discovery call reads current registered IDs, opens fresh
backends, warns using only a failing ID, and continues. Exact-ID lookup opens
only the requested device and accepts typed session overrides.

Fourth, add `sampler` and `estimator` methods to `QDMIBackend`. Remove the
MQT-specific `options` arguments from the direct primitive constructors and make
estimator shots an explicit parameter.

Fifth, add `CompilerTarget.from_device_id`. Put the C++ conversion of typed
session overrides in the installed QDMI driver interface shared by `open_device`
and MLIR. Regenerate recursive stubs and override the generated callable
signature with a pattern that uses `Unpack[QDMISessionParameters]`.

Sixth, add the changelog entries and record the completed implementation and
validation. The Slurm framework shortcuts explored in the original plan are out
of scope following review.

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

The documentation uses backend methods as the canonical Qiskit primitive
construction path. Direct constructors remain compatible. Defaults must be 1024
shots and zero estimator precision unless the caller overrides them. Sampling
and estimation must still execute against DDSIM.

`CompilerTarget.from_device_id("mqt.ddsim.default")` must equal the target made
from an explicitly opened DDSIM handle. Unknown and incompatible devices must
retain their existing errors.

The aggregate branch is accepted when focused tests, recursive stubs, supported
test sessions, documentation, lint, wheel inspection, and `git diff --check`
pass locally or have a recorded environment boundary, and remote CI reaches a
terminal state for the exact published revision.

## Idempotence and Recovery

Formatting, stub generation, builds, and tests are repeatable. Keep all build
and tool caches within this worktree through `.agent/run.sh`. If a generator
changes unrelated output, inspect and exclude it rather than discarding user
work. If `main` advances before publication, rebase this branch and rerun
affected checks.

## Artifacts and Notes

The branch started at `a0c21c5826f54fa25d2d16bac65b65c6b59efc58`, the reviewed
head of PR #2043. PRs #2025 and #2043 have since merged, so PR #2084 now targets
`main`.

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
existing Slurm API remains unchanged.

Revision note (2026-08-13): Created the implementation plan after verifying the
reviewed stack base and inspecting the existing adapter, registry, binding, and
Slurm boundaries.

Revision note (2026-08-13): Recorded the completed local implementation and
validation, the terminal-cancelled lower Read the Docs build, and publication of
PR #2084.

Revision note (2026-08-13): Updated the plan after review moved the parameter
types into the QDMI namespace and removed the proposed Slurm shortcuts.

Revision note (2026-08-14): Replaced the native typing import hook with real
aliases to canonical public types and moved shared session conversion into the
installed driver interface.

Revision note (2026-08-14): Applied the two internal follow-ups and aligned the
typing-module split with current Scientific Python and Boost.Histogram practice.
