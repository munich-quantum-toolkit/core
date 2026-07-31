# Make the superconducting QDMI device runtime configurable and calibrated

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, every fresh superconducting QDMI session owns a strict
runtime model selected from inline JSON, a JSON file, environment variables, or
the adjacent bundled default. The same provider can expose different stable
device IDs simultaneously. SC queries additionally report a duration unit,
operation duration and fidelity defaults or site-tuple overrides, and optional
T1/T2 defaults or qubit overrides. The bundled calibration is explicitly
synthetic. These operation and calibration semantics directly address
[issue #1331](https://github.com/munich-quantum-toolkit/core/issues/1331).

## Progress

- [x] (2026-07-30 00:00Z) Audited the SC generator, singleton provider,
  generated-header CMake path, default JSON, and tests.
- [x] (2026-07-30 08:00Z) Implemented the strict schema-version-1 value model
  and validation.
- [x] (2026-07-30 08:00Z) Added source selection and session-owned topology and
  handles.
- [x] (2026-07-30 08:00Z) Implemented duration, fidelity, and T1/T2 fallback
  semantics.
- [x] (2026-07-30 08:00Z) Removed generation APIs, commands, headers, and custom
  build targets.
- [x] (2026-07-30 08:00Z) Added direct ABI, Driver, calibration, ownership, and
  relocation tests.
- [x] (2026-07-30 08:00Z) Updated SC and index documentation, migration notes,
  and validation evidence.
- [x] (2026-07-30 02:10 CEST) Restacked the SC-only commit onto the completed NA
      change and passed final combined build, C++, Python, runtime-copy, and
      lint validation.
- [x] (2026-07-31 06:40 CEST) Verified that PRs #1972 and #1974 are merged and
      replayed only the SC commit onto merge commit `9e1c4db5a`.
- [x] (2026-07-31 06:45 CEST) Removed unnecessary trailing return types while
      retaining the one lambda return annotation required for its braced
      `Calibration` result.
- [x] (2026-07-31 06:50 CEST) Repeated the complete 423-step release build,
      repository lint, focused Python tests, and warning-as-error documentation
      build on the rebased branch.
- [x] (2026-07-31 07:20 CEST) Independently reviewed exact head `0bf1f1b828`,
      identifying MF-01 (explicit two-qubit tuples could contradict the ordered
      coupling map) and NIT-01 (ambiguous source-precedence documentation).
- [x] (2026-07-31 07:30 CEST) Enforced exact ordered coupling membership for
      explicit two-qubit tuples, added parser and ABI regressions, and clarified
      explicit versus environment source precedence.
- [x] (2026-07-31 07:45 CEST) Repeated the complete SC and Driver binaries,
      runtime-file/imported-device CTest coverage, focused Python tests, full
      repository lint, release build, diff checks, and warning-as-error
      documentation build after remediation.
- [x] (2026-07-31 08:10 CEST) Independently verified MF-01 and NIT-01, finding
      MF-02: the Windows SC test copied only the provider DLL and omitted its
      adjacent runtime JSON.
- [x] (2026-07-31 08:15 CEST) Replaced the Windows-only DLL copy with the shared
      runtime-copy helper and deferred GoogleTest discovery until the provider,
      manifest, and JSON are colocated.
- [x] (2026-07-31 08:45 CEST) Reconfigured and rebuilt the SC target, repeated
      all 40 SC tests and six runtime-copy/import tests, and passed full
      repository lint and diff checks after MF-02.
- [x] (2026-07-31 09:00 CEST) Independently verified MF-02 at exact head
      `9f6853a3a`; no further actionable issue was found.
- [x] (2026-07-31 09:10 CEST) Published draft PR #1980 from the verified head
      and added its required changelog reference in a signed follow-up.

## Surprises & Discoveries

- Observation: current SC two-site support normalizes pairs by handle address,
  which erases configured orientation. Evidence:
  `MQT_SC_QDMI_Operation_impl_d::sortSites` swaps each pair according to
  `std::less` before sorting.
- Observation: tuple overrides must be checked against the effective support
  expansion during parsing, not deferred to handle materialization. Evidence:
  the parser now validates explicit tuples, one-site expansion, and ordered
  coupling expansion before a session commits state.
- Observation: generic Driver tests previously used CUSTOM1 and CUSTOM2 as
  arbitrary provider properties, which conflicts with their standardized
  configuration-source meaning. Evidence: the full Driver suite passed after
  reserving CUSTOM1/CUSTOM2 and moving generic property tests to CUSTOM3/4.
- Observation: the repository install command reports an unrelated inability to
  create `/usr/local/bin/capnpc`, but still installs the SC provider, JSON,
  manifest, and CMake runtime-file metadata into the selected prefix.
- Observation: the documentation host needed the nox environment's explicit
  certificate-authority bundle to fetch the QDMI tag; the subsequent
  documentation build completed successfully.
- Observation: the temporary parallel-development helper rejected simultaneous
  raw CUSTOM1/CUSTOM2 values, while PR B's authoritative helper correctly gives
  inline JSON precedence as required by the design. Evidence: the first
  final-stack SC run exposed the mismatched test expectation; the corrected test
  now verifies inline precedence and retry on a fresh session.
- Observation: PR #1328's human review identifies the concrete gaps behind #1331
  as a useful SC operation set and location-specific fidelity data. The rebased
  default retains `r`, `cz`, and `measure`, while direct and Driver tests
  exercise ordered operation support, tuple overrides, defaults, and site
  calibration.
- Observation: the first exact-head independent review found that explicit
  two-qubit tuples were checked for arity, range, and uniqueness but not against
  the ordered coupling map. QDMI requires every advertised two-site operation
  tuple to be a coupling edge, so the parser now enforces that invariant before
  materialization.
- Observation: the second exact-head review found that the SC Windows test
  retained a DLL-only copy even though provider initialization now loads an
  adjacent JSON file. The merged NA provider already established the required
  pattern: copy all target-declared runtime files and discover GoogleTests only
  after that copy has run.

## Decision Log

- Decision: Store supported site tuples as ordered vectors of site identifiers
  and compare identifiers in configured order. Rationale: provider pointer
  addresses are allocation details and oriented operations require explicit
  tuple order. Date/Author: 2026-07-29 / Codex.
- Decision: Resolve calibration in the order site override, operation or qubit
  default, then `QDMI_ERROR_NOTSUPPORTED`. Rationale: this is deterministic and
  represents missing calibration without inventing zero values. Date/Author:
  2026-07-29 / Codex.
- Decision: Keep the shared configuration-source helper and generic CUSTOM3/4
  test transition exclusively in PR B. Rationale: PR C consumes those interfaces
  and should contain only SC-specific behavior after restacking. Date/Author:
  2026-07-30 / Codex.
- Decision: Do not retain the prototype's unified validation and
  default-printing CLI. Rationale: PR C only needs a strict parser linked into
  the provider, and removing the old generator surface avoids introducing a new
  public command outside the agreed PR split. Date/Author: 2026-07-30 / Codex.
- Decision: Treat #1331 as completed by the SC refactor rather than adding a
  separate compatibility layer. Rationale: operations are now materialized from
  strict runtime configuration with ordered site support and per-tuple
  duration/fidelity, while site T1/T2 values have defaults and overrides; this
  resolves the issue's two explicit requirements without expanding the QDMI
  abstraction. Date/Author: 2026-07-31 / Codex.
- Decision: Require explicit two-qubit operation tuples to match an exact
  ordered coupling edge. Rationale: QDMI conformance requires every advertised
  supported pair to belong to the coupling map, and preserving orientation is
  part of the runtime schema contract. Date/Author: 2026-07-31 / Codex.
- Decision: Use `mqt_copy_qdmi_runtime` for the Windows SC test target and
  `PRE_TEST` discovery. Rationale: the shared helper follows target metadata and
  keeps the provider, manifest, and JSON together without duplicating asset
  lists or platform-specific copy commands. Date/Author: 2026-07-31 / Codex.

## Outcomes & Retrospective

The SC provider now owns topology, operations, ordered support tuples, and
calibration per session. Strict parsing validates version, required and unknown
fields, topology, uniqueness, numeric domains, support, and overrides.
Calibration queries implement tuple override then operation default, and qubit
override then qubit default, returning NOTSUPPORTED for absent values. Before
the final 2026-07-31 verification, the rebased SC suite reports 38 passed with
one expected unsupported-job-property skip, the full Driver suite reports 114
passed, the focused Python runtime-configuration selection reports two passed,
and the imported/runtime-file CTest selection reports six passed. The complete
423-step release build, warning-as-error documentation build, repository lint,
and diff checks also pass. Earlier stacked validation additionally covered
installation and wheel relocation. After MF-01 and NIT-01, the SC suite reports
39 passed with the same one expected skip; the Driver, Python, CTest, lint,
release-build, documentation, and diff checks remain green.

MF-02 uses the same Windows runtime-copy and deferred-discovery pattern already
validated for the merged NA provider. Local macOS validation confirms CMake
configuration, the SC target, all SC tests, and the shared runtime-copy/import
tests; the Windows CI jobs remain the platform-specific oracle.

## Context and Orientation

`include/mqt-core/qdmi/devices/sc/Generator.hpp` and
`src/qdmi/devices/sc/Generator.cpp` define and parse the current minimal JSON
model. `src/qdmi/devices/sc/Device.cpp` includes generated initializer macros
and creates one singleton containing all sites and operations.
`src/qdmi/devices/sc/CMakeLists.txt` runs a generator executable at build time.
The new model is created during QDMI session initialization instead.

An operation site override is a calibration value attached to one exact ordered
tuple supported by that operation. A qubit override changes T1 or T2 for one
site while inheriting any other default.

## Plan of Work

Replace the generator schema with explicit configuration types containing schema
version, duration unit, qubit calibration defaults and overrides, couplings,
operations, optional supported tuples, default duration/fidelity, and tuple
overrides. Reject unknown or missing keys and validate every numeric, topology,
uniqueness, arity, support, and calibration constraint.

Use the same configuration-source selection and QDMI parameter semantics as NA,
with SC-specific environment variables and adjacent default filename.
Materialize sites, ordered coupling handles, operations, and calibration in an
immutable session model. Associate each handle with its owner and reject foreign
handles and supplied site tuples.

Remove the singleton and generated-header pipeline, retaining a parser library
linked directly into the provider. Stage the default JSON as a runtime file and
document all source and calibration semantics. Add direct ABI and Driver
integration tests, including two distinct models at once.

## Concrete Steps

From the repository root:

    MLIR_DIR=/Users/burgholzer/CLionProjects/llvm-22.1.3/lib/cmake/mlir \
      ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./build/release/test/qdmi/devices/sc/mqt-core-qdmi-sc-device-test
    ./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test
    ./.agent/run.sh uvx nox -s tests-3.14 -- \
      -k 'device_configuration_arguments or sc_open_device_accepts_runtime_configuration'
    SSL_CERT_FILE=.nox/docs/lib/python3.14/site-packages/certifi/cacert.pem \
      ./.agent/run.sh uvx nox -s docs
    ./.agent/run.sh uvx nox -s lint

Run selected CTest coverage for `qdmi-sc-device`, `qdmi-driver`, and imported
runtime copying. Inspect build, install, and copied directories for
`mqt-core-qdmi-sc-device.json`.

## Validation and Acceptance

The 100-qubit bundled device retains the existing `r`, `cz`, and `measure`
topology while reporting its new synthetic duration, fidelity, and T1/T2 data. A
custom five-qubit inline model reports different metadata and calibration
without affecting a simultaneous default session. One-qubit and two-qubit sites
expand when omitted; higher arity requires explicit tuples. Ordered tuples,
foreign handles, malformed assignments, source precedence, retry after failure,
and post-init immutability are covered by automated tests.

## Idempotence and Recovery

The provider commits a model only after parsing, validation, and materialization
finish, so initialization can be retried. Test environment changes are scoped
and restored. Build, install, copy, documentation, and lint commands are
repeatable. Draft publication occurs only after independent exact-head
verification; the PR-numbered changelog is then added as a signed follow-up.

## Artifacts and Notes

The SC test binary exercises strict parsing, source precedence, retry,
per-session ownership, ordered tuples, and calibration fallback. The install
prefix contains both `mqt-core-qdmi-sc-device.qdmi.json` and
`mqt-core-qdmi-sc-device.json`; its exported CMake target records the JSON in
`QDMI_RUNTIME_FILES`. The wheel contains both files, and
`open_device("mqt.sc.default")` from `/private/tmp` reports the bundled
100-qubit device without relying on the source tree.

## Interfaces and Dependencies

The provider accepts CUSTOM1/CUSTOM2 and
`MQT_CORE_QDMI_SC_CONFIG_JSON`/`MQT_CORE_QDMI_SC_CONFIG_FILE`.
`mqt-core-qdmi-sc-device.json` is the bundled adjacent fallback. Configuration
parsing uses nlohmann JSON and diagnostics use spdlog. QDMI duration, fidelity,
site T1/T2, duration-unit, and scale-factor properties are returned through the
existing QDMI v1 ABI.

Revision note: completed on 2026-07-30 after runtime, calibration, integration,
packaging, relocation, Python, documentation, and lint validation. The plan was
updated with exact test counts, prerequisite ownership, and environment-boundary
observations.
