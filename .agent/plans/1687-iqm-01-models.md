# Add reusable IQM Garnet and Emerald superconducting models

Status: historical implementation record.

## Goal and scope

After this change, users can select stable, offline models of IQM Garnet and
Emerald through the built-in superconducting QDMI provider. The registry IDs
`mqt.sc.iqm.garnet` and `mqt.sc.iqm.emerald` open the same provider library with
different installed JSON configurations. The models preserve their topology,
portable operation set, reported qubit names, and the available calibration
snapshot without carrying credentials, calibration identifiers, raw service
responses, or invented operation durations.

This is the first independently reviewable part of the larger pull request
[#1687](https://github.com/munich-quantum-toolkit/core/pull/1687). It is
deliberately limited to device configuration, provider packaging, documentation,
and tests. Compiler targets, mapping, native synthesis, Python compiler APIs,
and command-line integration belong to later changes.

## Constraints

- the SC schema already represents optional per-qubit T1/T2 and optional
  per-site-tuple duration/fidelity. It only lacks the optional site name that
  QDMI already exposes through `QDMI_SITE_PROPERTY_NAME`.

- the IQM provider obtains site names from the ordered `qubits` array of the
  static architecture response and assigns the array position as the QDMI site
  index. IQM qubits in these captures use `QB1` through `QB20` for Garnet and
  `QB1` through `QB54` for Emerald.

- the captured Garnet model contains 20 qubits, 30 topology edges, complete
  T1/T2 coverage, and 20 `r`, 30 `cz`, and 20 `measure` fidelities. The Emerald
  model contains 54 qubits, 90 topology edges, T2 for all 54 qubits, T1 for 53
  qubits, and 54 `r`, 81 `cz`, and 54 `measure` fidelities.

- the authenticated interface did not report operation durations. An absent
  duration is meaningful and must continue to produce `QDMI_ERROR_NOTSUPPORTED`.

- one shared provider library can already own multiple runtime JSON files, but
  its generated manifest currently registers only one device definition. The
  packaging helper needs a small extension to emit additional definitions that
  select a runtime file through `session.device-config.file`.

- the exact calibration capture is from 2 August 2026. A later live refresh
  could drift independently of the larger PR, so this change keeps the exact
  reviewed snapshot rather than silently mixing capture dates.

- QDMI-on-IQM revision `5bef1d49245ae17877203618ad65865405fab361` maps the
  service through these endpoint templates: `api/v1/quantum-computers`,
  `api/v1/quantum-computers/%s/artifacts/static-quantum-architectures`,
  `api/v1/calibration-sets/%s/%s/dynamic-quantum-architecture`, and
  `api/v1/calibration-sets/%s/%s/metrics`. The first two select the computer and
  return names/topology, the third returns default-calibration operation site
  tuples, and the fourth returns T1/T2 and fidelity metrics.

- CTest registers the individual GoogleTest cases, not targets named after the
  three executables. A target-name regular expression therefore selected no
  tests; running each built executable directly exercised 171 tests.

## Decisions

- keep schema version 1 and add an optional `name` member to each
  `qubitProperties.overrides` entry. Rationale: this is a backward-compatible
  optional property and follows the existing per-site override structure; a
  version bump would make all current configuration files unnecessarily
  incompatible.

- allow a qubit override that supplies only `name`, while continuing to require
  at least one of `name`, `t1`, or `t2`. Reject empty names. Rationale: names
  are independent of calibration availability, and an empty configured string is
  not useful metadata.

- preserve the reported IQM names and the provider's index order as `QB1`
  through `QBN`. Rationale: QDMI index and name are distinct properties;
  retaining both permits stable compiler indices while allowing callers to
  recover provider-facing hardware names.

- reuse the exact sanitized topology and calibration files from pull request
  #1687 and add only site names. Rationale: the files were already reviewed,
  contain no credentials or calibration identifiers, and preserve
  nanosecond-level coherence precision through unit `us` with scale factor
  `0.001`.

- do not add any duration field. Rationale: neither the captured device data nor
  the IQM QDMI interface supplied operation durations.

- extend `mqt_configure_qdmi_device` with a `CONFIGURATIONS` list whose entries
  pair a stable registry ID with the basename of an existing runtime file.
  Reject malformed or empty pairs, unknown runtime files, and duplicate IDs
  including the provider's primary ID. Rationale: the generated manifest remains
  relocatable and unambiguous, and all variants reuse one library and prefix
  without duplicating provider targets.

- retain the source revision, endpoint interpretation, retrieval date, and
  sanitization record solely as technical provenance. No attribution notice is
  required for the IQM data, as confirmed by the maintainer during review.
  Rationale: this is enough to reproduce and refresh the snapshots without
  adding legal/process prose to user documentation or the assets.

## Outcome and validation

Schema version 1 gained optional site names. Garnet and Emerald snapshots retain
topology and calibration data, add provider-facing qubit names, and omit
unavailable durations. SC provider, registry, driver, runtime packaging, and
installation checks passed, with the existing optional job-ID skip. Lint passed.

## Code and ownership

`include/mqt-core/qdmi/devices/sc/Configuration.hpp` and
`src/qdmi/devices/sc/Configuration.cpp` define and parse the strict JSON model.
`include/mqt-core/qdmi/devices/sc/Device.hpp` and
`src/qdmi/devices/sc/Device.cpp` materialize one model per QDMI session and
answer QDMI property queries.

The JSON assets live in `json/sc/`. The SC provider is assembled in
`src/qdmi/devices/sc/CMakeLists.txt`. `cmake/AddMQTQDMIDevice.cmake` generates
relocatable registry fragments, copies runtime files next to provider libraries,
and installs those files.

Schema and provider behavior are tested in `test/qdmi/devices/sc/`. Generated
registry discovery is tested in `test/qdmi/registry/test_device_registry.cpp`,
and opening every configured device is tested in
`test/qdmi/driver/test_driver.cpp`.

## Acceptance

The implementation is accepted when:

- Garnet parses as 20 sites and 30 ordered topology edges, and Emerald parses as
  54 sites and 90 ordered topology edges.
- Both assets expose exactly `r`, `cz`, and `measure`; all recorded
  per-site-tuple fidelities remain available.
- Garnet exposes 20 T1 and 20 T2 values. Emerald exposes 53 T1 and 54 T2 values.
- Site indices remain zero-based while reported names are `QB1` through `QB20`
  or `QB54`, respectively.
- No operation or site tuple in either model reports a duration.
- The IDs `mqt.sc.iqm.garnet` and `mqt.sc.iqm.emerald` resolve to the SC
  provider plus the correct packaged JSON file in a build-tree registry.
- The two JSON files are included in the provider's runtime-file property and
  install rules.
- Empty IDs/files and duplicate configuration IDs fail during CMake
  configuration, while a valid additional configuration builds and installs.
- Existing unnamed SC configurations remain backward compatible.
- Public documentation records the retrieval date, field provenance, and
  snapshot limitation. This ExecPlan records the technical source and
  sanitization without claiming that the calibration is current.
- Focused C++ tests, repository lint, and `git diff --check` pass.
