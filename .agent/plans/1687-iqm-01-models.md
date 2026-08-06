# Add reusable IQM Garnet and Emerald superconducting models

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This plan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

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

## Progress

- [x] (2026-08-03) Verified the task starts from commit
      `fe1935473e940f44ac312376934aabcbfb4a0e8c` in a clean isolated worktree.
- [x] (2026-08-03) Audited the current superconducting schema, provider,
      generated registry manifests, runtime-asset installation, and related
      tests.
- [x] (2026-08-03) Audited the exact Garnet and Emerald JSON captured at pull
      request #1687 commit `dd9619bd27a34ced8ed68a4ee4533cb85771f144`.
- [x] (2026-08-03) Reviewed QDMI-on-IQM revision
      `5bef1d49245ae17877203618ad65865405fab361`, specifically
      `src/iqm_device.cpp` and `src/internal/iqm_api_config.cpp`, to establish
      the meaning and source of site names, topology, T1/T2, and operation
      fidelities.
- [x] (2026-08-03) Extended the schema and runtime provider with optional site
      names.
- [x] (2026-08-03) Added the two sanitized configurations and registered them as
      additional configurations of the existing SC provider.
- [x] (2026-08-03) Added focused schema, provider, registry, driver, and
      packaging coverage.
- [x] (2026-08-03) Hardened the reusable CMake helper against empty
      configuration IDs/files and duplicate IDs, including collisions with the
      provider's primary ID, and added configure-time rejection tests.
- [x] (2026-08-03) Updated the SC provider documentation and changelog with
      concise provenance and snapshot limitations.
- [x] (2026-08-03) Configured with LLVM/MLIR 22.1.3, built the three affected
      test executables, and ran 42 SC tests, 15 registry tests, and 114 driver
      tests. One pre-existing optional SC job-ID test remained skipped.
- [x] (2026-08-03) Passed the eight runtime-file packaging and invalid-helper
      configure tests, verified an SC runtime-component install contains both
      assets and the generated manifest, and confirmed the two JSON files match
      the exact #1687 snapshot after removing only the added site-name lines.
- [x] (2026-08-03) Passed the complete repository lint session, secret and
      duration scans, `git diff --check`, and final diff inspection.
- [x] (2026-08-03) Prepared the completed implementation for one signed commit
      with the repository-required assistance trailer.

## Surprises & Discoveries

- Observation: the SC schema already represents optional per-qubit T1/T2 and
  optional per-site-tuple duration/fidelity. It only lacks the optional site
  name that QDMI already exposes through `QDMI_SITE_PROPERTY_NAME`.
- Observation: the IQM provider obtains site names from the ordered `qubits`
  array of the static architecture response and assigns the array position as
  the QDMI site index. IQM qubits in these captures use `QB1` through `QB20` for
  Garnet and `QB1` through `QB54` for Emerald.
- Observation: the captured Garnet model contains 20 qubits, 30 topology edges,
  complete T1/T2 coverage, and 20 `r`, 30 `cz`, and 20 `measure` fidelities. The
  Emerald model contains 54 qubits, 90 topology edges, T2 for all 54 qubits, T1
  for 53 qubits, and 54 `r`, 81 `cz`, and 54 `measure` fidelities.
- Observation: the authenticated interface did not report operation durations.
  An absent duration is meaningful and must continue to produce
  `QDMI_ERROR_NOTSUPPORTED`.
- Observation: one shared provider library can already own multiple runtime JSON
  files, but its generated manifest currently registers only one device
  definition. The packaging helper needs a small extension to emit additional
  definitions that select a runtime file through `session.device-config.file`.
- Observation: the exact calibration capture is from 2 August 2026. A later live
  refresh could drift independently of the larger PR, so this change keeps the
  exact reviewed snapshot rather than silently mixing capture dates.
- Observation: QDMI-on-IQM revision `5bef1d49245ae17877203618ad65865405fab361`
  maps the service through these endpoint templates: `api/v1/quantum-computers`,
  `api/v1/quantum-computers/%s/artifacts/static-quantum-architectures`,
  `api/v1/calibration-sets/%s/%s/dynamic-quantum-architecture`, and
  `api/v1/calibration-sets/%s/%s/metrics`. The first two select the computer and
  return names/topology, the third returns default-calibration operation site
  tuples, and the fourth returns T1/T2 and fidelity metrics.
- Observation: CTest registers the individual GoogleTest cases, not targets
  named after the three executables. A target-name regular expression therefore
  selected no tests; running each built executable directly exercised 171 tests.

## Decision Log

- Decision: keep schema version 1 and add an optional `name` member to each
  `qubitProperties.overrides` entry. Rationale: this is a backward-compatible
  optional property and follows the existing per-site override structure; a
  version bump would make all current configuration files unnecessarily
  incompatible. Date/Author: 2026-08-03, Codex.
- Decision: allow a qubit override that supplies only `name`, while continuing
  to require at least one of `name`, `t1`, or `t2`. Reject empty names.
  Rationale: names are independent of calibration availability, and an empty
  configured string is not useful metadata. Date/Author: 2026-08-03, Codex.
- Decision: preserve the reported IQM names and the provider's index order as
  `QB1` through `QBN`. Rationale: QDMI index and name are distinct properties;
  retaining both permits stable compiler indices while allowing callers to
  recover provider-facing hardware names. Date/Author: 2026-08-03, Codex.
- Decision: reuse the exact sanitized topology and calibration files from pull
  request #1687 and add only site names. Rationale: the files were already
  reviewed, contain no credentials or calibration identifiers, and preserve
  nanosecond-level coherence precision through unit `us` with scale factor
  `0.001`. Date/Author: 2026-08-03, Codex.
- Decision: do not add any duration field. Rationale: neither the captured
  device data nor the IQM QDMI interface supplied operation durations.
  Date/Author: 2026-08-03, Codex.
- Decision: extend `mqt_configure_qdmi_device` with a `CONFIGURATIONS` list
  whose entries pair a stable registry ID with the basename of an existing
  runtime file. Reject malformed or empty pairs, unknown runtime files, and
  duplicate IDs including the provider's primary ID. Rationale: the generated
  manifest remains relocatable and unambiguous, and all variants reuse one
  library and prefix without duplicating provider targets. Date/Author:
  2026-08-03, Codex.
- Decision: retain the source revision, endpoint interpretation, retrieval date,
  and sanitization record solely as technical provenance. No attribution notice
  is required for the IQM data, as confirmed by the maintainer during review.
  Rationale: this is enough to reproduce and refresh the snapshots without
  adding legal/process prose to user documentation or the assets. Date/Author:
  2026-08-03, Codex.

## Repository Orientation

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

## Plan of Work

First, add `std::optional<std::string> name` to `sc::Device::QubitOverride`.
Teach the strict parser to accept `name`, reject an empty name, and accept an
override when at least one of name, T1, or T2 is present. Preserve the existing
checks for valid unique qubit indices and positive coherence values.

Add an optional name to `MQT_SC_QDMI_Site_impl_d`. During session
initialization, apply configured names alongside T1/T2. Handle
`QDMI_SITE_PROPERTY_NAME` in `queryProperty`: return the configured
NUL-terminated string when present and `QDMI_ERROR_NOTSUPPORTED` when absent.
This keeps every existing unnamed configuration valid.

Add `json/sc/iqm-garnet.json` and `json/sc/iqm-emerald.json` from exact pull
request #1687 commit `dd9619bd27a34ced8ed68a4ee4533cb85771f144`. Add the
reported name corresponding to each index. Preserve all topology, T1/T2, and
fidelity values byte-for-byte apart from the added names. Verify
programmatically that no object contains a `duration` key.

Extend `mqt_configure_qdmi_device` so a provider manifest can contain the
existing default entry plus named configuration entries. Validate that every
entry has exactly an ID and filename and that the filename is among
`RUNTIME_FILES`. Add the two assets and IDs in
`src/qdmi/devices/sc/CMakeLists.txt`. The existing runtime copy and install
loops then package all three SC JSON files.

Expand configuration tests to assert the model names, qubit and edge counts,
exact operation names, calibration coverage, per-site-tuple fidelity counts,
site-name endpoints, and complete absence of operation durations. Add schema
coverage for name-only overrides and empty-name rejection. Expand provider tests
to prove configured names are queryable and absent names remain unsupported.
Expand registry and driver tests to prove that five built-in definitions exist,
both stable IDs use packaged configuration files, and both models can be opened.

Update `docs/qdmi/sc_device.md` to describe optional names and the two shared
models. Record that topology and site names came from the static architecture,
operation site tuples from the default dynamic architecture, and
T1/T2/fidelities from default calibration-set quality metrics retrieved on 2
August 2026. State that the files contain no credentials or calibration
identifiers and intentionally omit unavailable durations. In this ExecPlan, pin
the interpretation to QDMI-on-IQM revision
`5bef1d49245ae17877203618ad65865405fab361` and record its source paths and the
fact that no provider source, documentation text, or raw service response is
redistributed. Add one narrowly scoped `Unreleased` changelog entry without
inventing a pull request reference; add the reference in a post-publication
follow-up.

## Milestones

### Milestone 1: Schema and provider support optional site names

At the end of this milestone, a JSON qubit override may contain a non-empty
`name`, and a caller can query it through QDMI. Existing unnamed models still
parse and return `QDMI_ERROR_NOTSUPPORTED` for name queries.

From the repository root, build and run the SC provider tests:

    ./.agent/run.sh cmake --preset debug
    ./.agent/run.sh cmake --build --preset debug --target \
      mqt-core-qdmi-sc-device-test
    ./.agent/run.sh ./build/debug/test/qdmi/devices/sc/\
      mqt-core-qdmi-sc-device-test

The new name parsing and querying tests must pass together with all existing SC
tests.

### Milestone 2: Reusable IQM models are registered and packaged

At the end of this milestone, building the provider places both new JSON files
beside its library and the generated manifest contains `mqt.sc.iqm.garnet` and
`mqt.sc.iqm.emerald`. Registry and driver tests open the configured variants and
report the correct device names.

From the repository root, build and run:

    ./.agent/run.sh cmake --build --preset debug --target \
      mqt-core-qdmi-registry-test mqt-core-qdmi-driver-test
    ./.agent/run.sh ./build/debug/test/qdmi/registry/\
      mqt-core-qdmi-registry-test
    ./.agent/run.sh ./build/debug/test/qdmi/driver/\
      mqt-core-qdmi-driver-test

The registry test must find five built-in definitions and regular files for both
configuration paths. The driver test must open five devices and include
`IQM Garnet` and `IQM Emerald`.

### Milestone 3: Documentation, full validation, and atomic handoff

At the end of this milestone, the docs explain the models' source and limits,
all focused tests and repository lint pass, the diff contains no secret or
duration field, and one signed commit records the implementation.

From the repository root, run:

    ./.agent/run.sh ./build/debug/test/qdmi/devices/sc/\
      mqt-core-qdmi-sc-device-test
    ./.agent/run.sh ./build/debug/test/qdmi/registry/\
      mqt-core-qdmi-registry-test
    ./.agent/run.sh ./build/debug/test/qdmi/driver/\
      mqt-core-qdmi-driver-test
    ./.agent/run.sh ctest --test-dir build/debug --output-on-failure \
      -R 'mqt-core-qdmi-(runtime-file|configuration-rejects)'
    ./.agent/run.sh uvx nox -s lint
    git diff --check
    git status --short

Also inspect the JSON mechanically:

    jq -e '
      .operations | all(
        (has("duration") | not) and
        (.siteOverrides | all(has("duration") | not))
      )
    ' json/sc/iqm-garnet.json json/sc/iqm-emerald.json

Both `jq` invocations must print `true`. Search the final diff for common token,
credential, and calibration-identifier field names before committing; there must
be no match in either asset.

## Validation and Acceptance

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

## Idempotence and Recovery

All configuration, build, test, and lint commands are repeatable. The checked-in
snapshot is authoritative for this change; validation must not rewrite it from a
live service. If configuration fails midway, rerun CMake through `.agent/run.sh`
and rebuild the named targets. Do not delete another task's worktree or build
directory.

No external GitHub mutation is authorized by this plan. Do not push, open or
edit a pull request, comment, resolve threads, or merge. Do not persist or print
credentials while validating provenance.

## Outcomes & Retrospective

IQM-01 is complete. The SC schema remains version 1 and now carries optional
site names without changing unnamed configurations. Garnet and Emerald are
installed beside the existing provider, discoverable by stable IDs, and retain
the exact reviewed topology/calibration data plus provider-facing qubit names.
Durations remain absent.

Validation built the SC provider, registry, and driver targets with LLVM/MLIR
22.1.3. The SC binary passed 41 tests with one pre-existing optional job-ID test
skipped; the registry passed 15 tests and the driver passed 114. Eight
configure/build/install tests covered valid runtime-file packaging plus empty
and duplicate configuration rejection. A runtime-component install smoke check
contained both IQM JSON files and the SC manifest. Repository lint and final
diff/secret/duration checks passed.

No external GitHub action was taken. The changelog entry intentionally has no
pull request link because no replacement pull request exists yet; publication
must add that reference in a revision-scoped follow-up before release.
