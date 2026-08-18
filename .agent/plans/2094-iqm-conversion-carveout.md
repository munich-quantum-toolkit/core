# Move IQM conversion support to QDMI-on-IQM

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core's Qiskit adapter can submit a circuit to any QDMI device. To do that it
must turn a Qiskit `QuantumCircuit` into a _program_ in a _program format_ the
device accepts. Today the adapter also carries IQM JSON, a format only IQM
hardware understands, and `MoveGate`, a Qiskit gate only IQM hardware has. That
puts one vendor's format inside a vendor-neutral adapter. No device MQT Core
ships advertises IQM JSON, and the only consumer is the external package
`iqm-qdmi` from QDMI-on-IQM.

After this change, the package that owns a device also owns the conversion into
that device's format and registers it with MQT Core. MQT Core keeps only the
vendor-neutral formats, and it registers those through the same interface rather
than hard-coding them. Someone who installs `iqm-qdmi` sees no difference: their
circuits still reach IQM hardware as IQM JSON. Someone writing a new device
library can add their own format without changing MQT Core.

Two observable outcomes define success. First, a package that registers a
serializer for a format can submit a circuit to a device that advertises that
format, and the program that reaches the device is the one the serializer
produced. Second, nothing in MQT Core mentions IQM JSON or `move` outside the
superconducting device models, and the Qiskit adapter still submits OpenQASM 2
and OpenQASM 3 as before.

Tracked as issue #2094 in `munich-quantum-toolkit/core`, under the v4 cleanup
tracker #2085. The pull request is #2114.

## Progress

- [x] (2026-08-14) First implementation under the name "program codec": registry
      module, entry point group, `_EXTRA_GATES` seam, tests, docs.
- [x] (2026-08-17 10:53Z) Review of this plan by @burgholzer on #2114 requested
      changes to the design. Seven comments, all on this file.
- [x] (2026-08-17) Merged `main` into the branch. Conflicts in `CHANGELOG.md`,
      `UPGRADING.md`, and `python/mqt/core/plugins/qiskit/backend.py` resolved
      by keeping every change from `main` and re-applying only the seam.
- [x] (2026-08-17) Revised this plan against the review: serializer terminology,
      two payload signatures, an explicit format preference order, MQT Core's
      own OpenQASM serializers registered through the same registry, the
      ecosystem cross-check, and the mid-term `mqt-cc` outlook.
- [x] (2026-08-17 16:20Z) Milestone 1: expose the binary-payload classification
      in Python. `qdmi::isBinaryProgramFormat` is public in
      `include/mqt-core/qdmi/Client.hpp`, bound as
      `mqt.core.qdmi.is_binary_program_format`, and covered by
      `QDMITest.BinaryProgramFormatClassification` in
      `test/qdmi/test_client.cpp` and one test in
      `test/python/qdmi/test_qdmi.py`. A second classifier was bound and then
      withdrawn; see the Decision Log.
- [x] (2026-08-17 16:40Z) Milestone 2: the serializer registry lives in
      `python/mqt/core/plugins/qiskit/serializers.py` and is covered by
      `test/python/plugins/qiskit/test_serializers.py`.
- [x] (2026-08-17 16:55Z) Milestone 3: `QDMIBackend._serialize_circuit` is one
      ordered walk, `QDMIBackend.device` is public, and the two OpenQASM
      serializers are module-level functions registered at import of
      `backend.py`.
- [x] (2026-08-17 17:00Z) Milestone 4: the gate seam and the removal of the IQM
      pieces. The `_EXTRA_GATES` seam, the classmethod gate maps, the deletion
      of `converters.py` and `gates.py`, and the test renames landed with the
      first implementation and needed no change under the revised design. Only
      the `__init__.py` exports moved to the serializer names.
- [x] (2026-08-17 17:30Z) Milestone 5: documentation, changelog, upgrade guide,
      and full validation.

## Surprises & Discoveries

- Observation: MQT Core's C++ QDMI client already knows which program formats
  carry a binary payload and which carry no generic payload at all, so the
  Python side does not need to invent that classification. Evidence: in
  `src/qdmi/Client.cpp`, an anonymous namespace defines `isBinaryProgramFormat`
  (true for `QDMI_PROGRAM_FORMAT_QIRBASEMODULE`,
  `QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE`, `QDMI_PROGRAM_FORMAT_QPY`) and
  `hasNoGenericProgramPayload` (true for `QDMI_PROGRAM_FORMAT_CALIBRATION`,
  `QDMI_PROGRAM_FORMAT_BATCHJOB`), and the text overload of `Device::submitJob`
  throws `std::invalid_argument` with the message
  `"Binary program formats require exact-byte submission"` when it is handed a
  binary format.

- Observation: the Python `Device.submit_job` already accepts either payload
  type, so carrying `bytes` through the adapter needs no binding change.
  Evidence: `python/mqt/core/qdmi/__init__.pyi` declares two overloads, one with
  `program: str` ("Submits a text job to the device.") and one with
  `program: bytes` ("Submits an exact byte payload to the device.").

- Observation: `register_program_serializer` must not read the entry points,
  even though the other registry functions do. Reading them first would let an
  entry point occupy a format before any runtime registration, which inverts the
  documented precedence, and it would make `backend.py` fail at import time with
  `ValueError: A program serializer for QASM3 is already registered` as soon as
  any installed package declared a `QASM3` entry point. Evidence: the first
  version of `test_runtime_registration_beats_an_entry_point` failed with that
  exact message. The registration now inserts without loading, and the entry
  point loop keeps using `setdefault`, so a registration always wins.

- Observation: naming the bound parameter `format`, as the interface section
  first stated, fails `uvx nox -s lint`. Evidence: `ruff` reports
  `builtin-argument-shadowing` for the generated stub
  `python/mqt/core/qdmi/__init__.pyi`, saying that the argument `format` shadows
  a Python builtin, and that stub must not be edited by hand. The bindings
  therefore use `"program_format"_a`, which is also the name `Device.submit_job`
  already uses.

- Observation: Qiskit has no entry point group for turning a circuit into a
  submission payload, so there is no existing interface to conform to. Evidence:
  Qiskit's `pyproject.toml` defines exactly eight groups, all
  transpiler-related: `qiskit.unitary_synthesis`, `qiskit.synthesis`, and
  `qiskit.transpiler.{init,translation,routing,optimization,layout,scheduling}`.
  `qiskit.qasm3` and `qiskit.qpy` are closed modules with no plugin lookup.

## Decision Log

- Decision: call the seam a _program serializer_, not a codec, converter, or
  translator. Rationale: Qiskit itself calls this operation serialization —
  `qiskit.qasm3.dumps` is documented as "Serialize a `QuantumCircuit` object in
  an OpenQASM 3 string", the class behind it is
  `qiskit.qasm3.exporter.Exporter`, and `qiskit.qpy.dump` writes the binary
  form. IQM uses the same word: `IQMBackend.serialize_circuit` and
  `serialize_instructions`. "Codec" is not established in this community.
  "Translator" would collide with `qiskit.transpiler.translation` and
  `BasisTranslator`, which mean basis translation, a different operation that
  this same adapter also relies on. "Converter" is common at provider level
  (`qiskit_to_ionq`, `circuit_to_aqt`) but says nothing about producing a wire
  payload, and MQT Core already uses "convert" for in-memory conversions between
  its own IR and Qiskit (`qiskit_to_mqt`, `mqt_to_qiskit`), which would give one
  word two meanings. Date/Author: 2026-08-17, @marcelwa after review by
  @burgholzer.

- Decision: two serializer signatures, one per payload kind, distinguished by
  the program format rather than by separate registries. Rationale: the review
  asks for a text signature and a binary signature. A format fixes its payload
  kind, so one registry keyed by `ProgramFormat` is enough, and the backend
  checks the returned type against the format before submission. Date/Author:
  2026-08-17, @marcelwa after review by @burgholzer.

- Decision: take the payload classification from the C++ client through new
  bindings instead of restating it in Python. Rationale: `src/qdmi/Client.cpp`
  already decides which formats are binary and which carry no program payload,
  and it rejects a mismatched submission. A second copy in Python would drift
  the first time QDMI adds a format. Date/Author: 2026-08-17, @marcelwa.

- Decision: record the format preference as one explicit, ordered tuple in MQT
  Core rather than relying on the order a device happens to report, and give
  device-native formats precedence over standardized ones. Rationale: the review
  asks for the order to be encoded rather than left to a mapping. A package
  registers a serializer for its own device's native format because it wants
  that format used, and that is the precedence IQM JSON has today. The
  standardized formats follow in order of what a circuit may contain, and
  encoding breaks a tie only within one profile, because the profile decides
  what a program may say while the encoding decides only how it travels. One
  tuple in one module means the whole policy can be re-read, and changed, in one
  place. Date/Author: 2026-08-17, @marcelwa after review by @burgholzer.

- Decision: bind only `is_binary_program_format`, and let the serializer module
  keep its own `NON_CIRCUIT_FORMATS`. Rationale: an earlier revision also bound
  `has_program_payload`, renamed from the private `hasNoGenericProgramPayload`.
  Dropping "generic" made the name claim more than QDMI supports. QDMI declares
  `CALIBRATION` as `void*` "A calibration program" and says only that triggering
  a calibration run "does not require a program to be set", so its payload is
  optional rather than absent, while `BATCH_JOB` takes a list of job handles
  rather than a byte blob. One predicate cannot state both. What the registry
  actually needs is narrower and belongs to this adapter: which formats can hold
  a serialized circuit. MQT Core rejecting every `CALIBRATION` submission is a
  separate, pre-existing question about `Device::submitJob`, tracked outside
  this plan. Date/Author: 2026-08-18, @marcelwa after review by @burgholzer.

- Decision: rank QPY and OpenQASM 3 above the QIR base profile. Rationale: an
  earlier draft grouped the QIR formats together and put both profiles above QPY
  and OpenQASM 3, which treats "QIR" as one capability tier. It is not. The QIR
  base profile forbids classical feedback and mid-circuit control, while QPY
  carries a Qiskit circuit without loss and OpenQASM 3 expresses control flow,
  so both are more capable than QIR base and less capable than QIR adaptive.
  Date/Author: 2026-08-18, @marcelwa after review by @burgholzer.

- Decision: MQT Core registers its own OpenQASM 2 and OpenQASM 3 serializers
  through the same registry. Rationale: the review asks whether this would
  streamline the backend, and it does. The backend becomes one ordered walk with
  no special cases, the built-in formats obey the same preference tuple as every
  other format, and a provider whose device needs a different OpenQASM 3 export
  can replace ours the same way it registers anything else. Date/Author:
  2026-08-17, @marcelwa after review by @burgholzer.

- Decision: a serializer receives the backend, not the device. Rationale: MQT
  Core's own OpenQASM 3 serializer needs `backend.target.operation_names` to
  decide which gate definitions to suppress, so a device-only signature would
  force the built-in formats to stay hard-coded, which contradicts the decision
  above. Passing the backend also matches the closest ecosystem precedent,
  `qiskit_to_ionq(circuit, backend, ...)` in qiskit-ionq, which reads
  `backend.name`, `backend.gateset()`, and `backend.options`. `QDMIBackend`
  gains a public `device` property so a serializer can still reach the device.
  Date/Author: 2026-08-17, @marcelwa.

- Decision: `register_program_serializer` does not read the entry points, while
  `program_serializer` and `unregister_program_serializer` do. Rationale: a
  runtime registration takes precedence over an entry point, and it can only do
  so if it may run before the entry points are read. MQT Core's own OpenQASM
  registrations run while `backend.py` is still importing, so a reading
  registration would also turn a third-party `QASM3` entry point into an import
  error. Date/Author: 2026-08-17, @marcelwa.

- Decision: the two bound classifiers name their parameter `program_format`, not
  `format`. Rationale: `format` is a Python builtin, so `ruff` rejects it in the
  generated stub, which must not be edited by hand. `Device.submit_job` already
  uses `program_format`. Date/Author: 2026-08-17, @marcelwa.

- Decision: entry points, not only a runtime registration call. Rationale:
  `QDMIProvider` builds a plain `QDMIBackend` for every registered device, so a
  user can reach a registered IQM device without ever importing `iqm.qdmi`. An
  import-time registration would miss that path. Date/Author: 2026-08-14,
  @marcelwa.

- Decision: a bad entry point warns and is skipped rather than raising.
  Rationale: a user with two device packages installed must not lose the working
  one because the other is broken. Date/Author: 2026-08-14, @marcelwa.

- Decision: keep the exception class name `TranslationError` even though the
  seam is called serialization. Rationale: it belongs to a hierarchy shared with
  the PennyLane plugin, which has `PennyLaneTranslationError`,
  `PennyLaneUnsupportedFormatError`, and `PennyLaneUnsupportedOperationError`
  under the same `QDMIPluginError` base. Renaming one plugin's class alone would
  break that symmetry, and renaming both plugins' classes is a separate change
  with its own upgrade note. Date/Author: 2026-08-17, @marcelwa.

- Decision: `MoveGate` leaves MQT Core, and a subclass supplies a non-standard
  native gate through `_EXTRA_GATES`. Rationale: after the star-topology work in
  QDMI-on-IQM, that backend hides `move` from the `Target` entirely, so MQT
  Core's copy would have had no user. Every provider examined solves the same
  problem the same way: qiskit-ionq defines `GPIGate`, `GPI2Gate`, `MSGate`, and
  `ZZGate` in `qiskit_ionq/ionq_gates.py` and injects them into its `Target`;
  qiskit-pasqal-provider defines `HamiltonianGate`; qiskit-on-IQM defines
  `MoveGate`. Date/Author: 2026-08-14, @marcelwa.

## Outcomes & Retrospective

All five milestones are done and the design is as described above. The Qiskit
backend now holds no format-specific branch: `_serialize_circuit` walks
`preferred_program_formats` and calls the first registered serializer it finds.
MQT Core's OpenQASM 2 and OpenQASM 3 exporters sit in that registry beside any
serializer a device package registers, so a provider can also replace them.
Nothing in MQT Core converts to IQM JSON or defines `move` any more.

The validation that closed the work, all from the repository root:
`uv run --no-sync pytest test/python/plugins/qiskit -q` passed 110 tests,
`uv run --no-sync pytest test/python/qdmi -q` passed 304,
`uv run --no-sync pytest test/python -q` passed 685 with 4 skipped,
`./build/release/test/qdmi/mqt-core-qdmi-test` passed 274, `uvx nox -s stubs`
produced a diff holding only the two new function stubs, and `uvx nox -s lint`
passed the full hook set.

Two lessons. First, a lazily loaded registry has an ordering contract that must
be decided per function, not per module: reading the entry points inside
`register_program_serializer` looked harmless and would have made the adapter's
own import fail against a third-party entry point for the same format. Second,
the classification test in `test/qdmi/test_client.cpp` states its expectations
through a `switch` with no default case, so the compiler, not a reviewer, is
what notices a program format added to QDMI later.

What remains is outside this repository: pull request #189 in
`iqm-finland/QDMI-on-IQM` must ship `qiskit_to_iqm_json` and `MoveGate` and
declare the `mqt.core.qiskit.program_serializers` entry point, and its
serializer must take the backend rather than the device. Until both land in one
environment, the cross-repository check in `Validation and Acceptance` cannot
run.

## Context and Orientation

### Terms

- **QDMI** is the Quantum Device Management Interface, the C API through which
  MQT Core talks to a device. `mqt.core.qdmi` is its Python binding and
  `include/mqt-core/qdmi/Client.hpp` its C++ client.
- A **program format** is the syntax in which a device accepts a program. Each
  one is a member of the `ProgramFormat` enum: `QASM2`, `QASM3`,
  `QIR_BASE_STRING`, `QIR_BASE_MODULE`, `QIR_ADAPTIVE_STRING`,
  `QIR_ADAPTIVE_MODULE`, `CALIBRATION`, `QPY`, `IQM_JSON`, `BATCH_JOB`, and
  `CUSTOM1` through `CUSTOM5`. A device reports the ones it accepts through
  `Device.supported_program_formats()`.
- A **program serializer** is a function that turns one Qiskit circuit into one
  program in one format. A format whose payload is text has a serializer
  returning `str`; a format whose payload is binary has one returning `bytes`.
- An **entry point** is a record a Python package writes into its own installed
  metadata, which another package can read without importing it. It is declared
  under `[project.entry-points."<group>"]` in `pyproject.toml`.
- A **`Target`** is Qiskit's model of what a backend can run: which operations,
  on which qubits, with what duration and error.

### The files that matter

Everything in the Qiskit adapter lives under `python/mqt/core/plugins/qiskit/`.
Before this change:

- `converters.py` holds one function, `qiskit_to_iqm_json(circuit, device)`.
- `gates.py` holds one class, `MoveGate`, an opaque two-qubit gate named `move`.
- `backend.py` holds `QDMIBackend`, a `qiskit.providers.BackendV2`. Its
  `_convert_circuit` method begins with a branch on
  `ProgramFormat.IQM_JSON in supported_program_formats`, then falls through to
  OpenQASM 3 and OpenQASM 2. Its module-level `_build_gate_mappings_for_backend`
  hard-codes `"move": MoveGate()` into the map from device operation names to
  Qiskit gates. Its `run` method converts each circuit and calls
  `self._device.submit_job(program=..., program_format=..., num_shots=...)`.
- `exceptions.py` holds the plugin's exception hierarchy, including
  `TranslationError`, `UnsupportedFormatError`, and `UnsupportedOperationError`.
- `__init__.py` re-exports the public names when Qiskit is installed.
- The tests are in `test/python/plugins/qiskit/test_mock_backend.py`, which
  builds a `MockQDMIDevice` in-process rather than opening a real device. It
  holds twelve `test_qiskit_to_iqm_json_*` tests and three tests about `move`.

On the C++ and binding side:

- `src/qdmi/Client.cpp` implements `qdmi::Device::submitJob` in two overloads,
  text and bytes, and holds the two payload classifiers described under
  `Surprises & Discoveries` in an anonymous namespace.
- `include/mqt-core/qdmi/Client.hpp` declares that class.
- `bindings/qdmi/qdmi.cpp` binds the whole client, including the `ProgramFormat`
  enum, into `mqt.core.qdmi`.
- `python/mqt/core/qdmi/__init__.pyi` is the generated stub for that module. It
  must never be edited by hand; `uvx nox -s stubs` regenerates it.

### Why the seam is required rather than optional

The IQM QDMI device advertises `QIR_BASE_STRING` and `IQM_JSON` and no OpenQASM
at all. Removing the IQM JSON branch without a replacement would therefore leave
every circuit bound for IQM hardware with no format to serialize into, and MQT
Core has no QIR serializer. The registry is what keeps that path working.

### Where this is going

In v4 and later, MQT Core intends to use `mqt-cc`, its own compiler, to produce
programs for a device instead of relying on Qiskit's exporters. The expected
targets are: any version of QIR a backend advertises, preferring a binary module
over a string and the adaptive profile over the base profile; OpenQASM 3 through
MQT Core's own exporter, which is in several respects more capable than
Qiskit's; and QPY, probably through the Qiskit C API once that exists. OpenQASM
2 will not generally be emitted, and a distinction between OpenQASM dialects
along the lines of the QIR base and adaptive profiles may or may not be worth
making.

This matters for the design in two ways. The registry is the seam those
serializers plug into, so they will replace MQT Core's registered built-ins
rather than add branches to the backend. And the preference tuple is where the
resulting choice between QIR, QPY, and OpenQASM is recorded, so that choice
stays a single readable list rather than control flow spread through a method.
When `mqt-cc` can emit QIR for a device that also has a vendor format, the
question of which should win becomes a real one; today it does not, because MQT
Core has no QIR serializer at all.

### What the Qiskit ecosystem does

The review asked whether this design matches how Qiskit has been extended for
other providers. It was checked against qiskit-ionq, qiskit-aqt-provider,
qiskit-pasqal-provider, qiskit-quantinuum-provider, qiskit-ibm-runtime,
qiskit-on-iqm, qBraid, and Qiskit itself. The findings that shaped this plan:

Conversion normally lives in a plain module function that the provider's own
`run` calls, with no registration at all: `qiskit_ionq/helpers.py`'s
`qiskit_to_ionq(circuit, backend, ...)`,
`qiskit_aqt_provider/circuit_to_aqt.py`'s `qiskit_to_aqt_circuit(circuit)`,
`qiskit_pasqal_provider`'s `gen_seq(register, device, circuit)`. That works
because each of those packages owns both the backend and the format. MQT Core
owns the backend but not the vendor formats, which is why it needs a
registration seam and they do not.

Qiskit defines no entry point group for this, so there is no established
interface to adopt. The nearest thing in Qiskit itself is the transpiler stage
plugin: a backend returns a plugin _name_ from `get_translation_stage_plugin`,
and the framework resolves it through the `qiskit.transpiler.translation` group.
AQT, IQM, and IBM all use that mechanism. It is a good precedent for entry
points as the discovery channel, and a further reason not to use the word
"translation" for anything else in the same adapter.

The closest structural precedent outside Qiskit is qBraid, which registers
program formats under a `qbraid.programs` entry point group as `ProgramSpec`
objects carrying a `serialize` callable and a `validate` callable, and lets a
device hold a list of them with an explicit override for which to target. That
is the same shape as this plan: an entry point group, a callable per format, and
an explicit selection policy.

On multiple formats per backend, no provider negotiates the format with the
device at run time. IonQ decides between its flat JSON and OpenQASM 3 with a
predicate on the circuit, `circuit_requires_qasm3`, and tags the payload with a
version string. qBraid uses an explicit list plus `set_target_program_type`. IBM
negotiates only a QPY _version_, as
`min(SERVICE_MAX_SUPPORTED_QPY_VERSION, QISKIT_QPY_VERSION)`. An explicit,
ordered, inspectable list is therefore the conventional answer, and it is what
this plan uses.

On non-standard native gates, every provider that has one defines a `Gate`
subclass in its own package and injects it into its `Target`. That is what
`_EXTRA_GATES` lets a `QDMIBackend` subclass do.

### Deliberately not in scope

The alias `"r": {"prx"}` stays in `QDMIBackend._GATE_ALIASES`, and
`json/sc/iqm-garnet.json` and `json/sc/iqm-emerald.json` stay where they are.
`prx` is IQM terminology, but MQT Core's own superconducting device model uses
it, and those two files are device models rather than conversion logic. The
reviewer agreed that all three stay.

Renaming `TranslationError` is out of scope, for the reason in the Decision Log.
Adding a QIR or QPY serializer is out of scope: MQT Core has none today, and the
registry is what will accept them later.

## Interfaces and Dependencies

At the end of this work the following must exist.

In `mqt.core.qdmi`, one module-level function bound from C++:

    def is_binary_program_format(program_format: ProgramFormat) -> bool: ...

It is true for `QIR_BASE_MODULE`, `QIR_ADAPTIVE_MODULE`, and `QPY`. It wraps
`qdmi::isBinaryProgramFormat`, which moves from the anonymous namespace in
`src/qdmi/Client.cpp` into the public `include/mqt-core/qdmi/Client.hpp`,
keeping its `constexpr noexcept` signature over `QDMI_Program_Format`.

Which formats cannot hold a serialized circuit is a question about this adapter
rather than about QDMI, so the serializer module answers it itself:

    NON_CIRCUIT_FORMATS: frozenset[ProgramFormat]

It holds `CALIBRATION` and `BATCH_JOB`.

In a new module `python/mqt/core/plugins/qiskit/serializers.py`:

    ENTRY_POINT_GROUP = "mqt.core.qiskit.program_serializers"

    class TextProgramSerializer(Protocol):
        def __call__(self, circuit: QuantumCircuit,
                     backend: QDMIBackend, /) -> str: ...

    class BinaryProgramSerializer(Protocol):
        def __call__(self, circuit: QuantumCircuit,
                     backend: QDMIBackend, /) -> bytes: ...

    ProgramSerializer = TextProgramSerializer | BinaryProgramSerializer

    PROGRAM_FORMAT_PREFERENCE: tuple[ProgramFormat, ...]

    def register_program_serializer(fmt: ProgramFormat,
                                    serializer: ProgramSerializer,
                                    *, replace: bool = False) -> None: ...
    def unregister_program_serializer(fmt: ProgramFormat) -> None: ...
    def program_serializer(fmt: ProgramFormat) -> ProgramSerializer | None: ...
    def preferred_program_formats(
            formats: Iterable[ProgramFormat]) -> list[ProgramFormat]: ...

`PROGRAM_FORMAT_PREFERENCE` is ordered from most to least preferred:

    IQM_JSON, CUSTOM1, CUSTOM2, CUSTOM3, CUSTOM4, CUSTOM5,
    QIR_ADAPTIVE_MODULE, QIR_ADAPTIVE_STRING,
    QPY, QASM3,
    QIR_BASE_MODULE, QIR_BASE_STRING,
    QASM2

`CALIBRATION` and `BATCH_JOB` are absent because a serialized circuit is not
what they carry.

`preferred_program_formats` takes the formats a device reports, drops any in
`NON_CIRCUIT_FORMATS`, and returns the rest ordered by
`PROGRAM_FORMAT_PREFERENCE`. A format that the tuple does not name — a member
added to QDMI later — sorts after every format the tuple does name, keeping the
order in which the device reported it.

`register_program_serializer` raises `ValueError` when the format already has a
serializer and `replace` is false, and also when the format is in
`NON_CIRCUIT_FORMATS`, because no serialized circuit can go there.

In `QDMIBackend`:

    @property
    def device(self) -> QDMIDevice: ...

    def _serialize_circuit(
        self, circuit: QuantumCircuit,
        supported_program_formats: Iterable[ProgramFormat],
    ) -> tuple[str | bytes, ProgramFormat]: ...

    _EXTRA_GATES: ClassVar[dict[str, Instruction | type[Instruction]]] = {}

`python/mqt/core/plugins/qiskit/converters.py` and `gates.py` no longer exist.

The matching change in QDMI-on-IQM, pull request #189 in
`iqm-finland/QDMI-on-IQM`, owns `qiskit_to_iqm_json` and `MoveGate`, declares

    [project.entry-points."mqt.core.qiskit.program_serializers"]
    IQM_JSON = "iqm.qdmi.serializers:qiskit_to_iqm_json"

and takes the backend rather than the device as its second argument. It is a
separate pull request in a separate repository and is not part of this plan's
work, but this plan's interface is what it must match.

## Plan of Work

### Milestone 1: expose the payload classification in Python

At the end of this milestone, Python can ask whether a program format carries a
binary payload, and the answer comes from the same code the C++ submission path
uses.

In `include/mqt-core/qdmi/Client.hpp`, add one `constexpr noexcept` free
function in namespace `qdmi`, `isBinaryProgramFormat`, with a Doxygen comment
that names the formats it covers and says why the distinction exists: a binary
format must be submitted as exact bytes. In `src/qdmi/Client.cpp`, delete the
anonymous-namespace copy of that function and use the public one.
`hasNoGenericProgramPayload` stays private and keeps its name. The behavior of
both `submitJob` overloads must not change.

In `bindings/qdmi/qdmi.cpp`, immediately after the `ProgramFormat` enum, define
the function on `qdmiModule` with `"program_format"_a` and a Google-style
docstrings, following the `job.def(...)` calls in the same file for style. Then
regenerate the stub with `uvx nox -s stubs` and check that the only change to
`python/mqt/core/qdmi/__init__.pyi` is the two new signatures.

Add C++ coverage in `test/qdmi/test_client.cpp`: a test that asserts the
classification for every `QDMI_Program_Format` member, so a format added later
fails the test rather than being silently misclassified. Add Python coverage in
`test/python/qdmi/test_qdmi.py` for the two bound functions.

### Milestone 2: the serializer registry

At the end of this milestone, `python/mqt/core/plugins/qiskit/serializers.py`
exists with the interface above and its behavior is covered by tests, but the
backend does not use it yet.

The module keeps a private dict from `ProgramFormat` to serializer and a private
flag recording whether the entry points have been read. The first call to any of
the registry functions reads the group `mqt.core.qiskit.program_serializers`
through `importlib.metadata.entry_points` and caches the result. Set the flag
before the loop, because loading an entry point imports a module that may call
back into this one. An entry point whose name is not a `ProgramFormat` member,
one whose format carries no program payload, and one that fails to import each
produce a `UserWarning` naming the entry point and the reason, and are skipped;
a runtime registration for the same format wins, so the loop uses `setdefault`.

Write the module docstring so that it explains, for a reader who has never seen
this repository, what a program serializer is, which two signatures exist and
how the format decides between them, how a package declares one through an entry
point, and that the preference order lives in `PROGRAM_FORMAT_PREFERENCE`.

Cover in `test/python/plugins/qiskit/test_mock_backend.py`, or in a new
`test_serializers.py` beside it if that file grows unwieldy: registering and
looking up a serializer; the `ValueError` on a duplicate without `replace`;
`replace=True` overriding; unregistering a format that has no serializer being a
no-op; `ValueError` when registering for `CALIBRATION` or `BATCH_JOB`; the
warnings for an unknown name, a payload-less format, and a failing import;
`preferred_program_formats` ordering a shuffled list, dropping `CALIBRATION` and
`BATCH_JOB`, and putting an unnamed format last. Every test that registers a
serializer must remove it again, through a fixture, so registry state does not
leak between tests.

### Milestone 3: the backend reduced to one ordered walk

At the end of this milestone the backend has no format-specific branch left and
carries a binary payload through to the device unchanged.

Add the public `device` property to `QDMIBackend`, beside the existing
`device_id` property.

Move the two OpenQASM bodies out of the conversion method into module-level
functions in `backend.py`, `_serialize_to_qasm3(circuit, backend)` and
`_serialize_to_qasm2(circuit, backend)`, and register them at the end of the
module with `register_program_serializer`. `_serialize_to_qasm3` keeps the
existing exclusion list and keeps deriving its basis gates from
`backend.target.operation_names`; the comment explaining why the exclusion list
exists must survive the move, because that reasoning is not obvious.
Registration happens at import of `backend.py`; note in a comment that
`mqt.core.plugins.qiskit.__init__` imports this module whenever Qiskit is
installed, so the built-in formats are always available once the adapter is.

Rename `_convert_circuit` to `_serialize_circuit` and reduce it to: materialize
the device's formats, raise `UnsupportedFormatError` if there are none, then
walk `preferred_program_formats(...)`, take the first format with a registered
serializer, call it, and check the result. A `str` for a binary format or a
`bytes` for a text format is a `TranslationError` naming the format and both
types. `UnsupportedOperationError` from a serializer propagates unchanged, so a
circuit the chosen format cannot express fails loudly rather than silently
arriving in a weaker format; any other exception becomes a `TranslationError`
that names the format. If the walk ends with nothing serialized, raise
`UnsupportedFormatError` listing the formats that were considered.

Update `run` to hold `str | bytes` and pass it to `submit_job` unchanged; both
payload types already have an overload. Rename the local `program_str` and the
`converted_circuits` element type accordingly.

Update the existing conversion tests to the new method name and add: a device
advertising `CUSTOM1` and `QASM3` uses the registered `CUSTOM1` serializer and
the program that reaches `submit_job` is the one it returned; a device
advertising several formats picks the one the preference tuple ranks first, not
the one it reported first; a serializer for a binary format returning `bytes`
reaches `submit_job` as `bytes`; a serializer returning the wrong type for its
format raises `TranslationError`; a device advertising only `CALIBRATION` raises
`UnsupportedFormatError`; and replacing the built-in `QASM3` serializer changes
what the backend submits.

### Milestone 4: the gate seam and the removal of the IQM pieces

At the end of this milestone MQT Core no longer contains IQM conversion logic,
and a subclass can add a device-native gate outside Qiskit's standard library.

`_build_gate_mappings_for_backend` takes a second parameter, `extra_gates`, and
folds it into the canonical gate mapping in place of the former hard-coded
`"move": MoveGate()`. `QDMIBackend` gains the `_EXTRA_GATES` class variable,
empty by default, and an `__init_subclass__` that rebuilds
`_QISKIT_TO_QDMI_GATE_MAP` and `_OPERATION_TO_GATE_MAP` from the subclass's own
`_GATE_ALIASES` and `_EXTRA_GATES`, so a subclass adds a gate without touching
global state. `_map_operation_to_gate` and `_map_qiskit_gate_to_operation_names`
become classmethods so a subclass reads its own maps rather than the base
class's.

Delete `converters.py` and `gates.py`. Update `__init__.py` to export
`ProgramSerializer`, `program_serializer`, `register_program_serializer`, and
`unregister_program_serializer` instead of `qiskit_to_iqm_json` and `MoveGate`,
keeping `__all__` sorted as the linter requires.

In the tests, delete the twelve `test_qiskit_to_iqm_json_*` tests and the three
`move` tests, and rename the mock device's two-qubit operation from `move` to
`hop` so no IQM naming remains in MQT Core's tests. Add a test that a
`QDMIBackend` subclass declaring `_EXTRA_GATES` has that gate in its `Target`
while the base class does not, and a test that the subclass's map does not leak
into the base class's.

### Milestone 5: documentation, changelog, upgrade guide, validation

Rewrite the "Program Codecs" section of `docs/qdmi/qdmi_backend.md` as "Program
Serializers": what a serializer is, the two signatures and how the format
decides between them, the entry point declaration, the runtime call, and the
preference order with the reasoning behind it. Update the numbered list further
down that describes what happens when a circuit runs, which currently names
`qiskit_to_iqm_json`.

Update the `CHANGELOG.md` entries for #2114 to the serializer names, and add one
for the two new `mqt.core.qdmi` functions. Update the `UPGRADING.md` section so
it names `iqm.qdmi.serializers`, `register_program_serializer`, and the new
entry point group, and so its example serializer takes the backend.

Then run the full validation below, inspect the whole diff, and fill in
`Outcomes & Retrospective`.

## Concrete Steps

All commands run from the repository root.

Install the development dependencies once:

    uv sync --locked --only-group dev

Build and install the package after any C++ or binding change. `MLIR_DIR` must
point at the `lib/cmake/mlir` directory of an LLVM/MLIR 22.1 or newer install:

    MLIR_DIR=<llvm-prefix>/lib/cmake/mlir \
      uv sync --inexact --no-dev --no-build-isolation-package mqt-core

Regenerate the stubs after the binding change, and inspect the result:

    uvx nox -s stubs
    git diff python/mqt/core/qdmi/__init__.pyi

Run the narrowest tests while iterating, then widen:

    uv run --no-sync pytest test/python/plugins/qiskit -q
    uv run --no-sync pytest test/python/qdmi -q
    uv run --no-sync pytest test/python -q

Build and run the C++ tests for the new classifiers:

    cmake --preset release
    cmake --build --preset release --target mqt-core-qdmi-test
    ./build/release/test/qdmi/mqt-core-qdmi-test

Build the documentation, which also regenerates the autoapi pages:

    uvx nox --non-interactive -s docs

Finish with the full hook set:

    uvx nox -s lint

## Validation and Acceptance

The two tests that prove the seam carries a program to the device are
`test_backend_uses_registered_serializer` and
`test_backend_prefers_registered_serializer_over_qasm` in
`test/python/plugins/qiskit/test_mock_backend.py`. Both register a serializer on
`ProgramFormat.CUSTOM1` for a mock device that advertises `CUSTOM1` and `QASM3`,
run a circuit, and assert on the program the mock's `submit_job` received. The
first proves the registry is consulted at all; the second proves a registered
format outranks OpenQASM.

`test_backend_respects_format_preference` proves the order comes from
`PROGRAM_FORMAT_PREFERENCE` and not from the device: the mock reports `QASM2`
first and `QASM3` second, and the backend must submit `QASM3`.

`test_backend_submits_binary_payload` proves a binary format survives as bytes:
it registers a serializer on a binary format, and the mock's `submit_job` must
receive a `bytes` object identical to what the serializer returned.

For the C++ and binding work, `./build/release/test/qdmi/mqt-core-qdmi-test`
must pass, including the new test that classifies every `QDMI_Program_Format`
member, and `uv run --no-sync pytest test/python/qdmi -q` must pass.

To see the cross-repository behavior, install this package and `iqm-qdmi` into
one environment and check that MQT Core finds the serializer without importing
`iqm.qdmi`:

    $ python -c "
    from mqt.core.plugins.qiskit import program_serializer
    from mqt.core.qdmi import ProgramFormat
    print(program_serializer(ProgramFormat.IQM_JSON))"
    <function qiskit_to_iqm_json at 0x...>

That check needs the matching QDMI-on-IQM release and is not part of this
repository's test suite.

`uvx nox -s lint` must pass with no new warnings, and the working tree must
contain no generated file that the build did not produce.

## Idempotence and Recovery

Every step here is repeatable. The registry functions are idempotent given
`replace=True`, and `unregister_program_serializer` on an unregistered format
does nothing.

Two steps can leave a confusing state if interrupted. Regenerating the stubs
writes into `python/mqt/core/qdmi/__init__.pyi`; if the result looks wrong, run
`git checkout -- python/mqt/core/qdmi/__init__.pyi` and `uvx nox -s stubs` again
against a freshly built package, because a stale build produces a stale stub.
And a test that registers a serializer without removing it afterwards leaks into
later tests in the same process, which shows up as an unrelated test choosing an
unexpected format; always register through a fixture that unregisters on
teardown.

Deleting `converters.py` and `gates.py` is the only destructive step. Do it
after the tests that covered them are gone, so no run is left importing a
missing module, and recover with `git checkout` against the merge base if
needed.

## Artifacts and Notes

The shape of the walk that replaces the format branches, as it should read when
Milestone 3 is done:

    formats = list(supported_program_formats)
    if not formats:
        msg = "The device reports no supported program formats"
        raise UnsupportedFormatError(msg)

    for fmt in preferred_program_formats(formats):
        serializer = program_serializer(fmt)
        if serializer is None:
            continue
        try:
            program = serializer(circuit, self)
        except UnsupportedOperationError:
            raise
        except Exception as exc:
            msg = f"Failed to serialize the circuit to {fmt.name}: {exc}"
            raise TranslationError(msg) from exc
        _check_payload_type(program, fmt)
        return program, fmt

    msg = ("No program serializer for any format the device supports: "
           f"{[fmt.name for fmt in formats]}")
    raise UnsupportedFormatError(msg)

The evidence that the C++ client already rejects a mismatched payload, from
`src/qdmi/Client.cpp`:

    Job Device::submitJob(const std::string& program, ...) const {
      if (isBinaryProgramFormat(format)) {
        throw std::invalid_argument(
            "Binary program formats require exact-byte submission");
      }
      if (hasNoGenericProgramPayload(format)) {
        throw std::invalid_argument(
            "Calibration and batch jobs do not use a generic program payload");
      }
      ...
