# Make the superconducting QDMI device runtime configurable and calibrated

Status: historical implementation record.

## Goal and scope

After this change, every fresh superconducting QDMI session owns a strict
runtime model selected from inline JSON, a JSON file, environment variables, or
the adjacent bundled default. The same provider can expose different stable
device IDs simultaneously. SC queries additionally report a duration unit,
operation duration and fidelity defaults or site-tuple overrides, and optional
T1/T2 defaults or qubit overrides. The bundled calibration is explicitly
synthetic. These operation and calibration semantics directly address
[issue #1331](https://github.com/munich-quantum-toolkit/core/issues/1331).

## Constraints

- current SC two-site support normalizes pairs by handle address, which erases
  configured orientation. Evidence: `MQT_SC_QDMI_Operation_impl_d::sortSites`
  swaps each pair according to `std::less` before sorting.

- tuple overrides must be checked against the effective support expansion during
  parsing, not deferred to handle materialization. Evidence: the parser now
  validates explicit tuples, one-site expansion, and ordered coupling expansion
  before a session commits state.

- generic Driver tests previously used CUSTOM1 and CUSTOM2 as arbitrary provider
  properties, which conflicts with their standardized configuration-source
  meaning. Evidence: the full Driver suite passed after reserving
  CUSTOM1/CUSTOM2 and moving generic property tests to CUSTOM3/4.

- Inline JSON takes precedence over a configuration file when both raw
  configuration slots are supplied. Test the shared configuration-source
  contract rather than copying an intermediate helper behavior.

- PR #1328's human review identifies the concrete gaps behind #1331 as a useful
  SC operation set and location-specific fidelity data. The bundled default
  retains `r`, `cz`, and `measure`, while direct and Driver tests exercise
  ordered operation support, tuple overrides, defaults, and site calibration.

- Every advertised two-site operation tuple must also be an ordered coupling
  edge. Validate this before materializing session state, in addition to arity,
  range, and uniqueness.

- Copy all provider-declared runtime files before GoogleTest discovery. Copying
  only the Windows DLL misses the adjacent JSON model required at
  initialization.

- Use explicit `std::pair<uint64_t, uint64_t>` values in coupling tests.
  `uint64_t` can alias different unsigned types across platforms; a ULL literal
  does not define that contract.

- Include the device interface that declares generated session and job handles
  directly; do not rely on transitive inclusion through `Device.hpp`.

- the same scoped environment-variable helper had been copied into the NA, SC,
  and registry tests. A shared test-only utility removes that duplication
  without adding test infrastructure to the installed QDMI API.

## Decisions

- Store supported site tuples as ordered vectors of site identifiers and compare
  identifiers in configured order. Rationale: provider pointer addresses are
  allocation details and oriented operations require explicit tuple order.

- Resolve calibration in the order site override, operation or qubit default,
  then `QDMI_ERROR_NOTSUPPORTED`. Rationale: this is deterministic and
  represents missing calibration without inventing zero values.

- Keep the shared configuration-source helper and generic CUSTOM3/4 test
  transition exclusively in PR B. Rationale: PR C consumes those interfaces and
  should contain only SC-specific behavior after restacking.

- Do not retain the prototype's unified validation and default-printing CLI.
  Rationale: PR C only needs a strict parser linked into the provider, and
  removing the old generator surface avoids introducing a new public command
  outside the agreed PR split.

- Treat #1331 as completed by the SC refactor rather than adding a separate
  compatibility layer. Rationale: operations are now materialized from strict
  runtime configuration with ordered site support and per-tuple
  duration/fidelity, while site T1/T2 values have defaults and overrides; this
  resolves the issue's two explicit requirements without expanding the QDMI
  abstraction.

- Require explicit two-qubit operation tuples to match an exact ordered coupling
  edge. Rationale: QDMI conformance requires every advertised supported pair to
  belong to the coupling map, and preserving orientation is part of the runtime
  schema contract.

- Use `mqt_copy_qdmi_runtime` for the Windows SC test target and `PRE_TEST`
  discovery. Rationale: the shared helper follows target metadata and keeps the
  provider, manifest, and JSON together without duplicating asset lists or
  platform-specific copy commands.

- Preserve the QDMI job methods as instance methods and suppress only the
  corresponding static-method diagnostics, matching the NA provider. Rationale:
  these methods implement opaque-handle ABI behavior that may gain per-job
  state, while changing them to static functions would only satisfy a local
  implementation detail.

- Place `ScopedEnvironmentVariable` in `test/qdmi/TestUtils.hpp` rather than a
  production QDMI utility. Rationale: it is test scaffolding used by three test
  binaries, so a test include path provides one implementation without expanding
  the runtime or public API.

## Outcome and validation

The provider owns topology, ordered support tuples, and calibration per session.
Parsing validates topology, numeric domains, support, and overrides before
materialization. Queries use tuple/operation and qubit/default precedence;
absent calibration returns NOTSUPPORTED.

Recorded validation covered native and Python tests, packaging and relocation,
release builds, documentation, and lint. The final shared
provider/driver/registry suites passed with two expected job-ID skips. Windows
runtime-file ordering remained a platform CI check.

## Code and ownership

`include/mqt-core/qdmi/devices/sc/Generator.hpp` and
`src/qdmi/devices/sc/Generator.cpp` define and parse the current minimal JSON
model. `src/qdmi/devices/sc/Device.cpp` includes generated initializer macros
and creates one singleton containing all sites and operations.
`src/qdmi/devices/sc/CMakeLists.txt` runs a generator executable at build time.
The new model is created during QDMI session initialization instead.

An operation site override is a calibration value attached to one exact ordered
tuple supported by that operation. A qubit override changes T1 or T2 for one
site while inheriting any other default.

## Acceptance

The 100-qubit bundled device retains the existing `r`, `cz`, and `measure`
topology while reporting its new synthetic duration, fidelity, and T1/T2 data. A
custom five-qubit inline model reports different metadata and calibration
without affecting a simultaneous default session. One-qubit and two-qubit sites
expand when omitted; higher arity requires explicit tuples. Ordered tuples,
foreign handles, malformed assignments, source precedence, retry after failure,
and post-init immutability are covered by automated tests.

## Interfaces

The provider accepts CUSTOM1/CUSTOM2 and
`MQT_CORE_QDMI_SC_CONFIG_JSON`/`MQT_CORE_QDMI_SC_CONFIG_FILE`.
`mqt-core-qdmi-sc-device.json` is the bundled adjacent fallback. Configuration
parsing uses nlohmann JSON and diagnostics use spdlog. QDMI duration, fidelity,
site T1/T2, duration-unit, and scale-factor properties are returned through the
existing QDMI v1 ABI.
