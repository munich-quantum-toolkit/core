# Make the neutral-atom QDMI device runtime configurable

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, every fresh MQT neutral-atom QDMI session parses and owns its
device description. Two stable IDs can use the same shared library and prefix
while reporting different names, capacities, sites, operations, and calibration.
Direct QDMI v1 callers can select inline JSON with CUSTOM1 or a JSON file with
CUSTOM2. With no explicit source, the provider loads
`mqt-core-qdmi-na-device.json` beside its own shared library, so build, install,
copied-runtime, and wheel layouts remain relocatable.

## Progress

- [x] (2026-07-29 00:00Z) Audited the generated-header pipeline, NA
  configuration structs, provider singleton, FoMaC consumer, and tests.
- [x] (2026-07-29 22:24Z) Added strict schema-version-1 parsing and validation.
- [x] (2026-07-29 22:24Z) Added source selection and module-relative bundled
  fallback.
- [x] (2026-07-29 22:24Z) Moved sites, zones, operations, units, and calibration
  into each session.
- [x] (2026-07-29 22:24Z) Removed generated initializers and generator-only APIs
  while preserving the configuration model used by NA FoMaC.
- [x] (2026-07-29 23:34Z) Added direct ABI, Driver integration, packaging, and
  concurrent independent-session tests.
- [x] (2026-07-29 23:38Z) Updated documentation and migration notes.
- [x] (2026-07-29 23:52Z) Completed the full build, install-tree and relocated
      provider checks, documentation build, stub audit, and repository lint
      gate.
- [x] (2026-07-30 14:30Z) Rebased the provider migration onto the merged
      configuration transport from #1967, repeated the affected validation, and
      published draft PR #1974 against `main`.
- [x] (2026-07-30 17:22Z) Simplified strict key validation and session
      materialization, then removed the remaining generator-era site and
      operation factories after independent review.
- [x] (2026-07-30 17:46Z) Moved materialized local-operation site vectors into
      session storage to avoid duplicating the potentially large pair list.

## Surprises & Discoveries

- Observation: `na::forEachRegularSites` already provides the runtime
  materialization inputs that the generated header emits. Evidence:
  `src/qdmi/devices/na/Generator.cpp` reports each site identifier, coordinates,
  module, and submodule to a callback.
- Observation: the legacy nlohmann `WITH_DEFAULT` macros accept missing nested
  fields. Evidence: strict validation now walks every nested object before
  deserialization and tests reject nested omissions and unknown keys.
- Observation: the legacy FoMaC round-trip expects versioned configuration
  serialization as well as parsing. Evidence: explicitly serializing
  `schema-version` repaired the focused C++ round-trip test.
- Observation: arbitrary CUSTOM1/CUSTOM2 values in generic Driver tests become
  NA configuration once this provider migrates. Evidence: those unrelated
  pass-through assertions now exercise CUSTOM3/CUSTOM4.
- Observation: the complete Sphinx build consumes generated MLIR reference pages
  even when the change itself is outside MLIR. Evidence: the first build
  reported missing generated includes; building the `mlir-doc` target before
  rerunning Sphinx produced a warning-free build.
- Observation: nlohmann JSON reports unsigned integers through
  `is_number_integer()` as well. Evidence: independent review identified that a
  coordinate above `INT64_MAX` could otherwise reach signed conversion; the
  validator now range-checks unsigned coordinates and has a regression case.
- Observation: bounding generated sites does not bound local two-qubit
  materialization, which examines site pairs. Evidence: independent review
  identified the remaining quadratic path; parsing now enforces a cumulative
  ten-million candidate-pair ceiling before materialization, with a regression
  below the site-count ceiling.

## Decision Log

- Decision: Retain the public configuration value types for the first runtime
  migration while removing header-generation functions and targets. Rationale:
  `src/na/fomac/Device.cpp` consumes these values, and renaming them is
  orthogonal to eliminating global provider state. Date/Author: 2026-07-29 /
  Codex.
- Decision: Parse and materialize into temporaries and assign them to a session
  only after all validation succeeds. Rationale: a caller must be able to fix
  configuration and retry initialization on the same allocated session.
  Date/Author: 2026-07-29 / Codex.
- Decision: Put source loading and module-relative lookup in `qdmi::detail`
  under `qdmi/common/DeviceConfiguration.hpp`. Rationale: the superconducting
  provider can reuse one tested QDMI v1 adapter without coupling either
  technology model to CUSTOM parameter enums. Date/Author: 2026-07-30 / Codex.
- Decision: Keep the unified cross-technology validation executable out of this
  PR. Rationale: adding it now would either depend on the not-yet-migrated SC
  parser or create a short-lived NA-only interface; the shared parser API
  remains available for validation. Date/Author: 2026-07-30 / Codex.
- Decision: Construct session-owned sites and operations directly with their
  owner instead of retaining generated-model factories and post-construction
  ownership mutation. Rationale: the session is now the sole materialization
  path, so the factories only added temporary objects and indirection.
  Date/Author: 2026-07-30 / Codex.

## Outcomes & Retrospective

The runtime provider no longer has a singleton or generated initializer. Each
session strictly parses and owns its configured sites, zones, operations,
calibration, and units. The Driver integration test opens one provider library
under two stable IDs with different JSON files and observes different device
models. Direct ABI tests cover source precedence, retry, malformed assignments,
environment conflicts, foreign handles, and concurrent independent sessions.

The rebased release build completed 427 targets. The NA provider suite completed
41 tests with 40 passes and the pre-existing unsupported job-ID query skipped;
the Driver suite passed all 112 tests; NA FoMaC passed 2 C++ tests and 12 Python
tests. Six imported-target and runtime-file CTest fixtures passed. Stub
generation produced no tracked change. The complete documentation build passed
after generating `mlir-doc`, and the full repository lint session and
`git diff --check` passed.

Installing into an isolated prefix placed the provider library, manifest, and
`mqt-core-qdmi-na-device.json` together. Opening that relocated installation
without an explicit configuration reported the bundled name, 100-qubit capacity,
and 103 total sites. The only install diagnostic was an unrelated, non-fatal
Cap'n Proto attempt to create a `/usr/local/bin/capnpc` symlink; all MQT Core
artifacts were installed and the relocated provider check succeeded.

## Context and Orientation

Before this plan, `include/mqt-core/qdmi/devices/na/Generator.hpp` defined the
neutral-atom JSON value model, `src/qdmi/devices/na/Generator.cpp` parsed JSON
and enumerated lattice sites, and `src/qdmi/devices/na/Device.cpp` included a
build-generated `DeviceMemberInitializers.hpp` and initialized one singleton
model. The runtime model and parser now live in
`include/mqt-core/qdmi/devices/na/Configuration.hpp` and
`src/qdmi/devices/na/Configuration.cpp`. `src/na/fomac/Device.cpp` consumes the
same value model.

A QDMI session is the opaque handle allocated before initialization. A site or
operation handle is a pointer owned by that session and is valid only while the
session remains alive.

## Plan of Work

Require top-level `schema-version` equal to 1, reject unknown and missing
fields, and validate names, capacities, lattices, generated sites, regions,
units, fidelities, operation arities, and shuttling identifiers. Expose the same
parser for provider initialization and applications that validate descriptions.

Store optional inline and file configuration strings in the allocated session.
Validate NUL-terminated QDMI assignments, allow capability probes and clearing,
and reject changes after successful initialization. Select explicit session
configuration before technology-specific environment variables, and those before
the adjacent bundled JSON. Map missing, inaccessible, malformed, allocation, and
internal errors to stable QDMI statuses.

Materialize the complete device model inside the session. Give sites and
operations an owner pointer and reject foreign handles and foreign supplied
sites before dereferencing query state. Replace singleton allocation with
`new (std::nothrow)` and direct deletion. Keep device initialize/finalize
stateless.

Remove `DeviceMemberInitializers.hpp`, generation custom commands,
`writeHeader`, and the `generate` command. Stage the uniquely named default JSON
through `RUNTIME_FILES`. Adapt FoMaC only where required by parser/schema
changes. Add direct provider and Driver integration tests.

## Concrete Steps

From the repository root:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./build/release/test/qdmi/devices/na/mqt-core-qdmi-na-device-test
    ./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test
    ./.agent/run.sh uv run --no-sync pytest test/python/na/test_na_fomac.py
    ./.agent/run.sh uvx nox -s lint

Run CTest relocation and install tests selected by `qdmi-na-device`,
`qdmi-driver`, and `qdmi-imported-device`. Inspect the copied and installed
runtime directories for both the library and `mqt-core-qdmi-na-device.json`.

## Validation and Acceptance

The bundled default opens with no source-tree path. Inline, file, environment,
and bundled sources follow the documented precedence. Invalid initialization
leaves the session allocated and configurable. Two sessions from the same
library simultaneously report different models. Query-before-init, foreign site,
foreign operation, malformed assignment, missing file, and post-init mutation
tests return the documented error class. The default configuration preserves
existing NA behavior and FoMaC conversion.

## Idempotence and Recovery

All build and test commands are repeatable and worktree-local. Initialization
uses temporary state, so retrying after replacing a bad source is safe.
Environment-variable tests restore prior process state. Runtime-copy commands
use `copy_if_different`. No external action is authorized by this plan.

## Artifacts and Notes

The installed runtime artifacts were `lib/libmqt-core-qdmi-na-device.dylib`,
`lib/mqt-core-qdmi-na-device.qdmi.json`, and `lib/mqt-core-qdmi-na-device.json`.
No NA `DeviceMemberInitializers.hpp` remains.

## Interfaces and Dependencies

The provider accepts QDMI v1 CUSTOM1 as inline JSON and CUSTOM2 as a file path.
It reads `MQT_CORE_QDMI_NA_CONFIG_JSON` or `MQT_CORE_QDMI_NA_CONFIG_FILE` only
when neither explicit value is present. It uses nlohmann JSON and spdlog already
present in MQT Core. The provider library links the NA configuration parser
directly and receives its bundled JSON through
`mqt_configure_qdmi_device(... RUNTIME_FILES ...)`.

Revision note: updated on 2026-07-30 after #1967 merged to record the clean
rebase onto `main`, repeated validation, draft PR #1974, and the follow-up
simplification of validation and session-owned construction.
