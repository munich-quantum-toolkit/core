# Move IQM serialization to QDMI-on-IQM

Status: historical implementation record for issue `#2094` and PR `#2114`.

## Outcome and ownership

The Qiskit backend selects a registered serializer for an advertised device
format. Core registers its OpenQASM serializers through the same interface;
QDMI-on-IQM owns IQM JSON serialization and `MoveGate`. This boundary lets a
device package support its format without adding vendor branches to Core.

`python/mqt/core/plugins/qiskit/serializers.py` owns registration, discovery,
and preference. `backend.py` owns submission. `src/qdmi/Client.cpp` and
`include/mqt-core/qdmi/Client.hpp` own binary payload classification, exposed
through `is_binary_program_format`. Tests live under
`test/python/plugins/qiskit/` and `test/python/qdmi/`.

## Decisions

- Use one registry keyed by `ProgramFormat`. Text serializers return `str` and
  binary serializers return `bytes`; validate the result before submission.
  Payload kind comes from the C++ client rather than a second Python table.
- A serializer receives `(circuit, backend)`. Built-in OpenQASM emission needs
  Qiskit target operation names; `backend.device` exposes device queries when
  needed. A device-only callback would force built-ins outside the registry.
- Discover entry points under `mqt.core.qiskit.program_serializers` lazily.
  Stable-ID discovery can create a backend without importing the device package,
  so import-time registration alone is insufficient.
- Runtime registration takes precedence over discovered serializers.
  `register_program_serializer` must not trigger discovery: loading first can
  occupy a format before the runtime registration or break backend import.
  Lookup and unregister trigger discovery. Duplicate registration requires
  `replace=True`; invalid entry points warn and are skipped.
- Keep a private registry with explicit not-started, loading, and loaded states.
  Publish discovered entries together, handle reentrant lookup consistently, and
  reset the state if metadata discovery aborts. A cached function alone cannot
  guard reentry before its first call returns. Concurrent discovery was outside
  this implementation's contract.
- Keep format preference in one ordered tuple, independent of device report
  order. Device-native formats precede standardized formats; profile capability
  matters before binary versus text encoding. The recorded order was IQM JSON,
  custom formats, Adaptive QIR module/string, QPY, QASM3, Base QIR
  module/string, then QASM2. Unknown future formats follow known ones in
  reported order.
- Exclude `CALIBRATION` and `BATCH_JOB` from circuit serialization in the
  adapter. Do not expose a broad “has payload” predicate: calibration payloads
  are optional, and batch jobs carry handles rather than serialized circuits.
- Keep the existing `TranslationError` hierarchy. A new name for the registry
  does not require an unrelated public exception rename.
- Device-specific gate classes belong to device packages and can be supplied
  through `_EXTRA_GATES`. The `r`/`prx` alias and bundled Garnet/Emerald JSON
  models stay in Core because Core's SC provider uses them.

The public functions are `register_program_serializer`,
`unregister_program_serializer`, `program_serializer`, and
`preferred_program_formats`. No QIR or QPY serializer was added by this task.
The registry is useful because Core owns the generic backend while other
packages own vendor formats; a plain provider-local function would not connect
those owners.

## Validation and remaining integration

Historical checks passed 110 Qiskit plugin tests, 304 QDMI Python tests, the
full Python suite with four skips, 274 native QDMI tests, stubs, and lint.
Focused tests prove registered payload submission, binary byte preservation,
format preference independent of device order, and runtime-registration
priority. The native format test uses an exhaustive switch to expose enum
additions.

Run the relevant Python checks from the repository root:

```sh
uv run --no-sync pytest test/python/plugins/qiskit test/python/qdmi -q
```

End-to-end IQM discovery required a matching QDMI-on-IQM package, originally
tracked by its PR `#189`, exporting `qiskit_to_iqm_json`, `MoveGate`, and the
serializer entry point. That cross-repository check was not completed by this
Core-only validation. Before relying on it, install matching versions and check
that `program_serializer(ProgramFormat.IQM_JSON)` resolves without explicitly
importing `iqm.qdmi`.
