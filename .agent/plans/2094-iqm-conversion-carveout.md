# Move IQM conversion support to QDMI-on-IQM

## Why this matters

MQT Core's Qiskit adapter can submit a circuit to any QDMI device. To do that it
must turn a Qiskit `QuantumCircuit` into a _program format_ the device accepts —
a string in some agreed syntax. Core handles OpenQASM 2 and OpenQASM 3, which
are vendor-neutral. It also handled IQM JSON, a format only IQM hardware
understands, and shipped `MoveGate`, a Qiskit gate only IQM hardware has.

That put one vendor's format inside a generic adapter. Nothing in Core uses it:
no device Core ships advertises IQM JSON, and the only consumer is the external
package `iqm-qdmi` from
[QDMI-on-IQM](https://github.com/iqm-finland/QDMI-on-IQM).

After this change, a package that owns a device also owns the conversion into
its format, and registers it with Core. Core keeps only the vendor-neutral
codecs. Someone installing `iqm-qdmi` sees no difference: their circuits still
go to IQM hardware as IQM JSON. Someone writing a new device library can now add
their own format without changing Core.

Tracked as [#2094](https://github.com/munich-quantum-toolkit/core/issues/2094),
under the v4 cleanup tracker
[#2085](https://github.com/munich-quantum-toolkit/core/issues/2085).

## Terms

- **QDMI** — the Quantum Device Management Interface, the C API through which
  Core talks to a device. `mqt.core.qdmi` is its Python binding.
- **Program format** — the syntax a device accepts a program in, named by a
  member of the `ProgramFormat` enum (`QASM2`, `QASM3`, `IQM_JSON`, `QPY`, …).
- **Program codec** — a function that converts one Qiskit circuit into one
  program format. Signature: `(circuit, device) -> str`.
- **Entry point** — a record a Python package writes into its own metadata at
  install time, which another package can read without importing it. Declared
  under `[project.entry-points."<group>"]` in `pyproject.toml`.
- **Target** — Qiskit's model of what a backend can run: which gates, on which
  qubits, with what duration and error.

## Starting state

Before this change, in `python/mqt/core/plugins/qiskit/`:

- `converters.py` held one function, `qiskit_to_iqm_json(circuit, device)`.
- `gates.py` held one class, `MoveGate`, an opaque two-qubit gate named `move`.
- `backend.py::QDMIBackend._convert_circuit` began with
  `if ProgramFormat.IQM_JSON in supported_program_formats:` and called
  `qiskit_to_iqm_json`, before falling through to OpenQASM 3 and OpenQASM 2.
- `backend.py::_build_gate_mappings_for_backend` hard-coded `"move": MoveGate()`
  into the map from device operation names to Qiskit gates.
- `__init__.py` exported both symbols publicly.
- `test/python/plugins/qiskit/test_mock_backend.py` held twelve
  `test_qiskit_to_iqm_json_*` tests and three tests about `move`.

The IQM device advertises only `QIR_BASE_STRING` and `IQM_JSON`, never any
OpenQASM. So deleting the branch without a replacement would leave every IQM
circuit with nothing to convert to. The seam is not optional polish; it is what
keeps IQM execution working.

## Design

### The seam

A new module `python/mqt/core/plugins/qiskit/codecs.py` holds a registry keyed
by `ProgramFormat`.

- `register_program_codec(fmt, codec, *, replace=False)` adds one at run time.
- `unregister_program_codec(fmt)` removes one. Tests use it to clean up.
- `program_codec(fmt)` returns the codec or `None`.

The first of those calls loads the entry point group
`mqt.core.qiskit.program_codecs` once and caches the result. The entry point
name is the `ProgramFormat` member name; the value points at the codec:

```toml
[project.entry-points."mqt.core.qiskit.program_codecs"]
IQM_JSON = "iqm.qdmi.converters:qiskit_to_iqm_json"
```

Entry points, rather than only a runtime call, because the device library is
discovered by Core, not the other way round: a user can enumerate devices with
`QDMIProvider` and get a plain `QDMIBackend` over an IQM device without ever
importing `iqm.qdmi`. An import-time registration would miss that path.

An entry point naming an unknown format, or one that fails to import, produces a
warning and is skipped. One broken package must not make every other codec
unreachable.

`_convert_circuit` walks the device's supported formats in the order the device
reports them and uses the first one with a registered codec. Only then does it
fall back to the built-in OpenQASM 3 and OpenQASM 2 exporters. A provider codec
therefore wins over OpenQASM, which is the precedence IQM JSON had before.

### The gate

`MoveGate` moves to `iqm.qdmi.gates`. Core's `_build_gate_mappings_for_backend`
takes a second argument, `extra_gates`, and `QDMIBackend` gains a `_EXTRA_GATES`
class variable, empty by default. `__init_subclass__` rebuilds
`_QISKIT_TO_QDMI_GATE_MAP` and `_OPERATION_TO_GATE_MAP` for each subclass from
its own `_GATE_ALIASES` and `_EXTRA_GATES`, so a subclass adds a device-native
gate without touching global state. `_map_operation_to_gate` and
`_map_qiskit_gate_to_operation_names` become classmethods so a subclass reads
its own maps rather than the base class's.

### Deliberately not in scope

- The alias `"r": {"prx"}` stays in Core's `_GATE_ALIASES`. `prx` is IQM
  terminology, but Core's own superconducting device model uses it, and #2085
  keeps that device supported.
- `json/sc/iqm-garnet.json` and `json/sc/iqm-emerald.json` stay. They are device
  models for the generic SC device, not conversion logic.

Both are worth revisiting; neither belongs in this change.

## Steps

1. Add `python/mqt/core/plugins/qiskit/codecs.py` as described above.
2. In `backend.py`: replace the `from .converters import ...` and
   `from .gates import ...` imports with `from .codecs import program_codec`;
   replace the IQM branch in `_convert_circuit` with the registry walk; add the
   `extra_gates` parameter and the `_EXTRA_GATES` / `__init_subclass__` seam;
   turn the two `_map_*` static methods into classmethods.
3. Delete `converters.py` and `gates.py`. Update `__init__.py` to export the
   codec API instead of `qiskit_to_iqm_json` and `MoveGate`.
4. In `test_mock_backend.py`: delete the `test_qiskit_to_iqm_json_*` block and
   the `move` tests. Add a `registered_codec` fixture that registers a codec on
   `ProgramFormat.CUSTOM1` and removes it afterwards, plus tests that the
   backend uses it and prefers it over OpenQASM. Add a test that a subclass with
   `_EXTRA_GATES` puts its gate in the Target while the base class does not.
   Rename the mock's two-qubit operation `move` to `hop` so no IQM naming
   remains.
5. Update `docs/qdmi/qdmi_backend.md`, `CHANGELOG.md`, and `UPGRADING.md`.

The matching change in QDMI-on-IQM adds `python/iqm/qdmi/converters.py` and
`gates.py`, declares the entry point, and ports the conversion tests. It is a
separate pull request in a separate repository.

## How to see it working

Build and install the package, then run the adapter tests:

```console
./.agent/run.sh uv sync --inexact --only-group build --only-group test
MLIR_DIR=<llvm>/lib/cmake/mlir ./.agent/run.sh uv sync --inexact --no-build-isolation-package mqt-core
./.agent/run.sh uv run --no-sync pytest test/python/plugins/qiskit -q
```

`test_backend_uses_registered_codec` and
`test_backend_prefers_registered_codec_over_qasm` are the two that prove the
seam carries a program to the device.

To see the cross-repository behavior, install both packages into one environment
and check that Core finds the codec without importing `iqm.qdmi`:

```console
$ python -c "
from mqt.core.plugins.qiskit import program_codec
from mqt.core.qdmi import ProgramFormat
print(program_codec(ProgramFormat.IQM_JSON))"
<function qiskit_to_iqm_json at ...>
```

## Decision log

- **Entry points over a subclass-only hook.** A `_PROGRAM_CODECS` class variable
  on `QDMIBackend` would be smaller, and `iqm-qdmi` does construct its own
  subclass for hardware. But `QDMIProvider` builds plain `QDMIBackend` objects
  for every registered device, and a registered IQM device reached that way
  would have no format left to convert to. Entry points cover both paths and
  require no import.
- **Registration order, not a priority number.** Provider codecs beat the
  built-in OpenQASM codecs; among themselves the device's own ordering decides.
  A priority parameter has no consumer, and #2085 asks not to add abstraction
  without one.
- **`_EXTRA_GATES` rather than leaving `MoveGate` in Core.** After the
  star-topology work in QDMI-on-IQM, that backend hides `move` from the Target
  entirely, so Core's copy would have had no user at all. The seam keeps the
  capability available to any device with a non-standard native gate.
- **A warning, not an exception, for a bad entry point.** A user with two device
  packages installed should not lose the working one because the other is
  broken.
