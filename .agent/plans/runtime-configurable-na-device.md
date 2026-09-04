# Make the neutral-atom QDMI device runtime configurable

Status: historical implementation record.

## Goal and scope

After this change, every fresh MQT neutral-atom QDMI session parses and owns its
device description. Two stable IDs can use the same shared library and prefix
while reporting different names, capacities, sites, operations, and calibration.
Direct QDMI v1 callers can select inline JSON with CUSTOM1 or a JSON file with
CUSTOM2. With no explicit source, the provider loads
`mqt-core-qdmi-na-device.json` beside its own shared library, so build, install,
copied-runtime, and wheel layouts remain relocatable.

## Constraints

- `na::forEachRegularSites` already provides the runtime materialization inputs
  that the generated header emits. Evidence: `src/qdmi/devices/na/Generator.cpp`
  reports each site identifier, coordinates, module, and submodule to a
  callback.

- the legacy nlohmann `WITH_DEFAULT` macros accept missing nested fields.
  Evidence: strict validation now walks every nested object before
  deserialization and tests reject nested omissions and unknown keys.

- the legacy FoMaC round-trip expects versioned configuration serialization as
  well as parsing. Evidence: explicitly serializing `schema-version` repaired
  the focused C++ round-trip test.

- arbitrary CUSTOM1/CUSTOM2 values in generic Driver tests become NA
  configuration once this provider migrates. Evidence: those unrelated
  pass-through assertions now exercise CUSTOM3/CUSTOM4.

- the complete Sphinx build consumes generated MLIR reference pages even when
  the change itself is outside MLIR. Evidence: the first build reported missing
  generated includes; building the `mlir-doc` target before rerunning Sphinx
  produced a warning-free build.

- nlohmann JSON reports unsigned integers through `is_number_integer()` as well.
  Evidence: independent review identified that a coordinate above `INT64_MAX`
  could otherwise reach signed conversion; the validator now range-checks
  unsigned coordinates and has a regression case.

- bounding generated sites does not bound local two-qubit materialization, which
  examines site pairs. Evidence: independent review identified the remaining
  quadratic path; parsing now enforces a cumulative ten-million candidate-pair
  ceiling before materialization, with a regression below the site-count
  ceiling.

- the Windows NA test copied only the provider DLL beside the test executable,
  while module-relative fallback also requires the bundled JSON there. Evidence:
  all 34 Windows failures reported the missing JSON beside the copied DLL;
  `mqt_copy_qdmi_runtime` already copies both plus the manifest.

## Decisions

- Retain the public configuration value types for the first runtime migration
  while removing header-generation functions and targets. Rationale:
  `src/na/fomac/Device.cpp` consumes these values, and renaming them is
  orthogonal to eliminating global provider state.

- Parse and materialize into temporaries and assign them to a session only after
  all validation succeeds. Rationale: a caller must be able to fix configuration
  and retry initialization on the same allocated session.

- Put source loading and module-relative lookup in `qdmi::detail` under
  `qdmi/common/DeviceConfiguration.hpp`. Rationale: the superconducting provider
  can reuse one tested QDMI v1 adapter without coupling either technology model
  to CUSTOM parameter enums.

- Keep the unified cross-technology validation executable out of this PR.
  Rationale: adding it now would either depend on the not-yet-migrated SC parser
  or create a short-lived NA-only interface; the shared parser API remains
  available for validation.

- Construct session-owned sites and operations directly with their owner instead
  of retaining generated-model factories and post-construction ownership
  mutation. Rationale: the session is now the sole materialization path, so the
  factories only added temporary objects and indirection.

- Reuse `mqt_copy_qdmi_runtime` for Windows NA tests instead of adding another
  asset-specific copy command. Rationale: it is the single existing contract for
  colocating a QDMI provider, manifest, and runtime files.

- Keep lattice enumeration dependency-free and preserve signed 64-bit
  coordinates with exact integer differences, checked coordinate arithmetic, and
  bounded span multiplication. Rationale: converting absolute coordinates to
  floating point before subtraction collapses adjacent values above 2^53, while
  constraining determinant products to signed 64-bit rejects valid large lattice
  vectors.

## Outcome and validation

Each session owns its strictly parsed model, replacing generated initialization
and the singleton. Direct ABI and Driver tests cover independent sessions,
source precedence, retry, malformed configuration, and foreign handles. Runtime
assets are copied together before test discovery.

Provider, driver, FoMaC/Python, runtime-file, release, documentation, and lint
checks passed, with the expected unsupported job-ID skip. A relocated
installation opened the bundled 100-qubit/103-site device without explicit
configuration.

## Code and ownership

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

## Acceptance

The bundled default opens with no source-tree path. Inline, file, environment,
and bundled sources follow the documented precedence. Invalid initialization
leaves the session allocated and configurable. Two sessions from the same
library simultaneously report different models. Query-before-init, foreign site,
foreign operation, malformed assignment, missing file, and post-init mutation
tests return the documented error class. The default configuration preserves
existing NA behavior and FoMaC conversion.

## Interfaces

The provider accepts QDMI v1 CUSTOM1 as inline JSON and CUSTOM2 as a file path.
It reads `MQT_CORE_QDMI_NA_CONFIG_JSON` or `MQT_CORE_QDMI_NA_CONFIG_FILE` only
when neither explicit value is present. It uses nlohmann JSON and spdlog already
present in MQT Core. The provider library links the NA configuration parser
directly and receives its bundled JSON through
`mqt_configure_qdmi_device(... RUNTIME_FILES ...)`.
