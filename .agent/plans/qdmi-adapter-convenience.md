# Add convenient framework adapters for registered QDMI devices

Status: historical implementation record.

Later removal of transitional aliases:
[compatibility API removal](remove-python-compatibility-apis.md).

## Goal and scope

MQT Core registers QDMI devices under stable string identifiers, but users still
need to assemble framework adapters manually. This change makes the stable ID
the common entry point for Qiskit backends and MLIR compiler targets.

After the change, a user can call `QDMIBackend.from_device_id`, then construct a
sampler or estimator from that backend. A user can also call
`CompilerTarget.from_device_id`.

## Constraints

- `QDMIProvider` currently opens every registered device in its constructor and
  silently skips failures. Evidence:
  `python/mqt/core/plugins/qiskit/provider.py` stores a populated `_backends`
  list in `__init__`.

- QDMI session fields include path-like values for `auth_file` and
  `device_config_file`, while job custom fields accept strings, booleans,
  floats, or `None`. Evidence: the generated signatures in
  `python/mqt/core/qdmi/driver.pyi` and `python/mqt/core/qdmi/__init__.pyi`.

- The real Slurm test image installs the wheel without optional Python
  dependencies. Evidence: `.github/workflows/slurm.yml` and
  `test/slurm/Dockerfile` install the wheel with `--no-deps`.

## Decisions

- Keep session and job parameters in separate total-false `TypedDict` classes.
  Rationale: both namespaces contain `custom1` through `custom5`, but they are
  consumed at different lifecycle points and have different value types.

- Use `Unpack[QDMISessionParameters]` for stable-ID factories and retain
  mappings for the two PennyLane parameter namespaces. Rationale: keywords give
  callers completion and reject misspelled fields statically, while mappings
  keep the overlapping PennyLane namespaces explicit.

- Keep domain-specific public types in `mqt.core.typing`, but isolate the
  version-dependent `Unpack` import in `mqt.core._compat.typing` and list only
  that private bridge in Ruff's `typing-modules`. Rationale: Ruff reserves this
  setting for modules that re-export `typing` or `typing_extensions` members;
  treating the public runtime `TypedDict`s as typing primitives would be
  incorrect.

- Keep the narrow Ruff suppression on the runtime `os` import in
  `mqt.core.typing`. Rationale: `typing.get_type_hints` resolves the public
  `os.PathLike` annotations at runtime; moving the import behind `TYPE_CHECKING`
  raises `NameError`, while hiding it behind a private alias leaks that alias
  into generated API documentation.

- Keep the Slurm adapter limited to `open_device_from_license` in this PR.
  Rationale: review requested that the proposed framework-specific Slurm
  shortcuts be reconsidered separately.

- Keep `TypedDict` and `Unpack` instead of using data classes. Rationale:
  `Unpack` requires a `TypedDict`, and unpacked keyword arguments are the
  convenient API this change provides.

- Keep direct primitive constructors compatible, but document backend factories
  as the canonical path. Rationale: removing public concrete Qiskit primitive
  classes is a separate breaking API decision and requires an explicit
  deprecation plan.

- Export the parameter dictionaries only from `mqt.core.typing`. Rationale: a
  second path through the native `mqt.core.qdmi` module adds binding and stub
  machinery without improving discoverability or compatibility for this
  unreleased API.

- Silently skip registered devices that open successfully but cannot be
  represented as Qiskit backends. Rationale: incompatibility is an expected
  result of enumerating a framework-independent registry, not an availability
  failure that should warn users.

## Outcome and validation

Typed configuration, stable backend identity, lazy discovery, fluent primitives,
and stable-ID compiler targets were implemented. Framework-specific Slurm
shortcuts were excluded. Focused and supported Python current/minimum matrices,
native Slurm/compiler tests, stubs, strict documentation, lint, and wheel
inspection passed.

## Code and ownership

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

## Acceptance

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

Acceptance covers focused and supported test sessions, recursive stubs,
documentation, lint, and wheel inspection. Historical local results do not
establish hosted CI success.

## Interfaces

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
