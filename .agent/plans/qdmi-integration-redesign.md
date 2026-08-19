# Reconcile QDMI device management with current main

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core users need one QDMI object model that supports both the convenient
process-wide device catalog and isolated registries for applications and tests.
After this change, a caller can continue to register and open a device through
the default `mqt.core.qdmi.driver` functions, or it can construct a
`DeviceRegistry` and immutable `DeviceManager` that do not share registrations
with the rest of the process. Each open creates a fresh device session, and
devices, sites, operations, jobs, and child devices retain the state required by
their native QDMI handles.

The work reconciles pull request #1901 with the QDMI changes already on `main`.
The implementation must preserve typed device-configuration transport, stable
device IDs, runtime superconducting and decision-diagram devices, Slurm, Qiskit,
PennyLane, MLIR compiler targets, job retrieval, queue properties, and custom
operation lists. It must not restore the removed neutral-atom stack, TOML
configuration, or the old QDMI C client and singleton driver.

## Progress

- [x] (2026-08-18 10:40Z) Refreshed the local `main` and PR #1901 references,
  reviewed the three unresolved discussions and the old CI failures, and
  created a fresh reconciliation branch from `origin/main`.
- [x] (2026-08-18 10:40Z) Replaced the historical design assumptions with the
  hybrid default-registry and explicit-manager design in this ExecPlan.
- [x] (2026-08-18 13:25Z) Added the public registry, manager, and default
  convenience interfaces on top of the configuration behavior from current
  `main`.
- [x] (2026-08-18 13:25Z) Replaced the singleton driver and C client
      implementation with direct, lifetime-safe device state while preserving
      all current object operations.
- [x] (2026-08-18 13:25Z) Updated Python bindings and all QDMI consumers while
  preserving lazy default-registry behavior in Qiskit, PennyLane, Slurm, and
  MLIR.
- [x] (2026-08-18 13:25Z) Updated migration, API, and release documentation and
  regenerated stubs.
- [x] (2026-08-18 13:25Z) Passed the focused and complete native suites, the
      Python 3.10 and 3.14 test and minimum-version sessions, focused
      integration tests, device-free configuration, stub generation, repository
      lint, and diff checks.
- [x] (2026-08-18 14:10Z) Rebased the implementation onto
      `3354fdaab1732254d75fb9a212351d97f182e086`, rebuilt the complete release
      and non-unity lint configurations, reran all 4,053 native tests and 430
      focused Python integration tests, regenerated stubs with nanobind 2.15,
      and passed Clang-Tidy on all changed C++ translation units.
- [x] (2026-08-18 23:30Z) Rebased the reconciliation onto
      `b66b1dc9b63ca9f3f0b5de88f22bd19b55e2479f`. Preserved the program
      serializer work from #2114 and the calibration-job API from #2148 in the
      direct device model. The rebase produced a signed reconciliation commit,
      and `git verify-commit` accepted its EDDSA signature.
- [x] (2026-08-18 23:55Z) Passed the focused QDMI build and 132 device, 14
      manager, and 14 registry tests. Passed 265 focused Python QDMI and Qiskit
      serializer tests. Stub generation produced no tracked diff. A clean
      documentation build passed with Graphviz 15.1.
- [x] (2026-08-19 00:20Z) Passed `qiskit-3.14` against Qiskit 2.6.0.dev0 with
      667 tests passed and three expected skips. Rust 1.97 built the Qiskit
      source dependency successfully.
- [x] (2026-08-19 00:35Z) Passed the 228-step release build and all 4,093 CTest
      tests; only `ScQDMIJobSpecificationTest.QueryJobId` was skipped. Passed a
      device-free reconfigure and build with both bundled QDMI devices disabled,
      including all 14 registry tests. Passed the lint preset configuration and
      its full 482-step non-unity build. Passed `run-clang-tidy-22` on all 18
      existing changed translation units with warnings as errors.
- [ ] Complete link validation. The link checker found current-main internal C++
      target mismatches and unrelated external DNS, 403, and 404 failures before
      it was stopped.
- [ ] Publish the lease-protected update to PR #1901 and reply to its open
  discussions after explicit authorization.

## Surprises & Discoveries

- Observation: PR #1901 predates more than one hundred commits on `main` and now
  conflicts with it. Its two commits contain removed neutral-atom and TOML
  behavior while missing later QDMI features. Evidence: the cached PR head is
  `4925b38bf532846bccac0f28319a2cab2f372179`, while cached `origin/main` is
  `43047338aab5d85dbbd4c6bf88dcac31bb2eac74`.
- Observation: current `main` still implements stable-ID registration through
  the singleton `qdmi::Driver`, but its public `Device`, `Site`, `Operation`,
  and `Job` wrappers already contain the newer queue, retrieval, custom-value,
  and child-device behavior. The direct implementation must start from these
  current wrappers rather than the older PR copies.
- Observation: network access recovered during implementation. Refreshed
  `origin/main` is `b66b1dc9b63ca9f3f0b5de88f22bd19b55e2479f`. It adds the
  nanobind 2.15 update, Qiskit program serializers, calibration jobs, the
  PennyLane SpecAudit, CMake 4.4 support, and later Qiskit and MLIR fixes after
  the original local base. PR #1901 still points to
  `4925b38bf532846bccac0f28319a2cab2f372179` and remains conflicted.
- Observation: the complete native run passed 4,053 tests. The focused Python
  QDMI, Qiskit, PennyLane, and MLIR run passed 430 tests. The supported Nox
  sessions passed with 580 tests and seven skips on Python 3.10, 623 tests and
  three skips on Python 3.14, 474 tests and 22 skips for Python 3.10 minimums,
  and 517 tests and 18 skips for Python 3.14 minimums.
- Observation: a Python 3.14 failure after a docs-only build came from a cached
  wheel that disabled the superconducting provider. A forced reinstall with the
  normal provider set resolved the failure; the full Python 3.14 session then
  passed.
- Observation: Graphviz 15.1 now provides `dot`, and the clean HTML
  documentation build passes. Rust and Cargo 1.97 are also available. The
  Qiskit-main Python 3.14 session passes against Qiskit 2.6.0.dev0 with 667
  tests passed and three expected skips. Link checking remains inconclusive
  because it found current-main internal C++ target mismatches and unrelated
  external network failures before the run was stopped.
- Observation: the non-unity lint build exposed one transitive include in the
  Slurm test that the unity release build hid. The final lint configuration
  builds all 617 targets, and Clang-Tidy passes all 18 changed C++ translation
  units with warnings treated as errors.
- Observation: after the final rebase, the release build completed 228 steps and
  CTest passed 4,093 of 4,093 tests with only
  `ScQDMIJobSpecificationTest.QueryJobId` skipped. The device-free build and 14
  registry tests pass with both bundled QDMI device targets disabled. The
  non-unity lint build completes all 482 steps, and `run-clang-tidy-22` passes
  on all 18 existing changed translation units with warnings as errors.

## Decision Log

- Decision: Keep the public name `DeviceManager`; do not rename it to
  `DeviceExplorer`. Rationale: opening fresh sessions and isolating bulk-open
  failures is management behavior, and the name already appears in PR #1901.
  Date/Author: 2026-08-18, user and Codex.
- Decision: A `DeviceManager` is an immutable registry snapshot and is never a
  singleton. Rationale: explicit managers must be isolated and safe to use in
  tests. Only the compatibility registry is process-wide. Date/Author:
  2026-08-18, user and Codex.
- Decision: Preserve the current names `DeviceDefinition`,
  `DeviceSessionConfig`, and `DeviceConfigurationSource`. Rationale: current
  C++, Python, Qiskit, Slurm, and compiler code already uses these types; the
  former PR's `SessionParameters` name would create needless migration work.
  Date/Author: 2026-08-18, Codex.
- Decision: Each manager open creates a fresh session. Replacing a definition
  changes future opens, while live devices retain their prior session and
  library state. Rationale: registration and runtime lifetime must be separate.
  Date/Author: 2026-08-18, user and Codex.
- Decision: Keep the process default API as free C++ functions and Python module
  functions backed by a locked default registry. Rationale: current adapters
  depend on registrations made after module import, while explicit managers need
  stable snapshots. Date/Author: 2026-08-18, user and Codex.
- Decision: Retain `include/mqt-core/qdmi/Client.hpp` as a forwarding include
  for the device object model, but remove the public legacy `Session`, `Driver`,
  and `MQT::CoreQDMIDriver` target. Rationale: recent object-model includes can
  keep compiling without preserving the conflicting runtime architecture.
  Date/Author: 2026-08-18, Codex.
- Decision: Private implementation headers use `.hpp`, and new implementation
  functions use conventional return-type syntax. Rationale: these changes
  directly address the two current inline review discussions. Date/Author:
  2026-08-18, PR review.
- Decision: Preserve the #2114 program-format payload contract in the direct
  object model. `qdmi::isBinaryProgramFormat` is public in `Device.hpp`, Python
  exposes `is_binary_program_format`, and the Qiskit serializer registry uses
  the same classification. Rationale: one classifier prevents the C++ client and
  Python serializer selection from drifting. Date/Author: 2026-08-18, upstream
  #2114 and Codex.
- Decision: Preserve the #2148 calibration API as two
  `Device::submitCalibrationJob` overloads and Python `submit_calibration_job`.
  Route normal and calibration submission through one private helper with
  optional program bytes and an optional shot count. Rationale: QDMI permits a
  calibration job with no program and no shot count, while a batch job needs a
  list of job handles that the generic byte-payload API cannot express.
  Date/Author: 2026-08-18, upstream #2148 and Codex.

## Outcomes & Retrospective

The local implementation now provides one QDMI object model with explicit
registry snapshots and process-default convenience functions. It removes the
duplicate singleton and C-client layers, preserves the configuration system and
later QDMI behavior from `main`, and directly addresses the three open PR
discussions. The code and generated stubs pass the available native, Python,
device-free, repository-lint, non-unity build, and Clang-Tidy checks from the
first reconciliation. After the rebase onto
`b66b1dc9b63ca9f3f0b5de88f22bd19b55e2479f`, the focused native and Python
contract tests, complete release and device-free builds, all 4,093 CTest tests,
the complete non-unity lint build, Clang-Tidy, Qiskit-main, stub generation, and
the HTML documentation build pass. The reconciliation commit is
cryptographically signed and verified. Link checking remains inconclusive
because of current-main internal target mismatches and external failures. Remote
publication and discussion replies require explicit authorization.

## Context and Orientation

`include/mqt-core/qdmi/Device.hpp` provides the C++ object wrappers, while
`include/mqt-core/qdmi/Client.hpp` is a compatibility forwarding header. The
private `src/qdmi/DeviceApi.hpp` and `src/qdmi/DeviceState.hpp` own library and
session lifetime. `src/qdmi/DeviceRegistry.cpp` discovers `qdmi.json` files,
CMake-generated manifests, environment configuration, and runtime fallback
definitions without loading device libraries. `bindings/qdmi/qdmi.cpp` binds the
object wrappers into `mqt.core.qdmi` and the registry interfaces into
`mqt.core.qdmi.driver`.

A device definition is inert metadata: a stable ID, a native library path, a
symbol prefix, and default session configuration. A registry stores those
definitions. A manager takes a value snapshot of a registry and opens fresh
native device sessions. The process default registry is a compatibility layer
used by module-level registration and opening functions. It is not a manager
singleton.

Native QDMI libraries expose C functions with a configured symbol prefix.
Private `DeviceApi` loads those functions and calls device-wide initialize and
finalize functions. Private `DeviceState` owns one native session. Public
objects share the state so a job or site remains valid after its original
`Device` wrapper or `DeviceManager` is destroyed.

## Plan of Work

First, promote the current registry parser into public `qdmi::DeviceRegistry`
without changing source precedence, disabled-ID masking, manifest behavior,
typed configuration, or validation. Add a locked process-default registry and
free registration, ID-listing, and opening functions. Add `qdmi::DeviceManager`
as a by-value snapshot with `open` and `openAll`.

Second, add private `DeviceApi.hpp` and `DeviceState.hpp`. `DeviceApi` loads the
exact QDMI device symbols and is cached weakly by canonical library path and
prefix. A per-key generation token prevents a replacement library generation
from initializing until the old generation has finalized and unloaded. Use RAII
and `std::shared_ptr` construction without raw custom deletion. Each
`DeviceState` owns one allocated and initialized device session. Public child
objects share that state. An operation must compare every supplied site's state
identity with its own state and throw before calling native code when they do
not match.

Third, move the current object wrapper API into `qdmi/Device.hpp` and its direct
implementation. Preserve queue position and length, job retrieval by ID, custom
operation lists and values, byte and text programs, result decoding, child
devices, equality, and Python lifetime behavior. Remove the singleton driver and
C client only after the direct tests pass. Keep `Client.hpp` as a forwarding
include and move session configuration to a driver-independent header.

Fourth, bind `DeviceRegistry`, `DeviceManager`, and `OpenAllResult` in
`mqt.core.qdmi.driver`. Preserve the current `DeviceDefinition` keyword
constructor and `open_device` keyword overrides. A default manager snapshots the
default registry at construction; an explicit manager copies the supplied
registry. `open_all` returns `devices` and `errors` mappings by stable ID and
does not stop after one device fails. Keep Qiskit discovery lazy and keep Slurm,
PennyLane, and compiler factories on the default module functions.

Finally, update CMake exports, documentation, `CHANGELOG.md`, and
`UPGRADING.md`. Regenerate stubs through Nox. Do not edit generated files by
hand and do not add workflow warning suppressions.

## Concrete Steps

Run commands from the repository root. Build the direct implementation and its
focused tests first:

    cmake --preset release
    cmake --build --preset release --target \
      mqt-core-qdmi-test mqt-core-qdmi-manager-test \
      mqt-core-qdmi-registry-test
    ./build/release/test/qdmi/mqt-core-qdmi-test
    ./build/release/test/qdmi/manager/mqt-core-qdmi-manager-test
    ./build/release/test/qdmi/registry/mqt-core-qdmi-registry-test
    uv run --no-sync pytest test/python/qdmi test/python/plugins/qiskit \
      test/python/plugins/qdmi_pennylane test/python/test_mlir.py

Then run the complete supported validation:

    cmake --build --preset release
    ctest --preset release
    uvx nox -s stubs
    uvx nox -s tests
    uvx nox -s minimums
    uvx nox -s tests-3.14
    uvx nox -s minimums-3.14
    uvx nox -s qiskit
    uvx nox --non-interactive -s docs
    uvx nox -s docs -- -b linkcheck
    uvx nox -s lint
    git diff --check
    git status --short

Also configure and build once with the bundled decision-diagram and
superconducting devices disabled. The device-independent registry tests must
still configure, build, and pass.

## Validation and Acceptance

Registry construction and ID listing must not load native code. Runtime fallback
registration must not override a disabled ID. Explicit registries and managers
must remain isolated from process-default registrations. A manager snapshot must
not change after later registration or replacement.

Every open must create a fresh session. Compatible live sessions may share one
device-wide library initialization. A new generation must wait for the prior
finalization. Devices and derived objects must remain valid after manager and
parent-wrapper destruction. A site from another session must fail before the
native operation callback runs. `openAll` must return successes and failures by
ID in the same result.

Existing Qiskit, PennyLane, Slurm, and MLIR tests must demonstrate that later
default registrations remain visible and that exact-ID opening does not load
unrelated devices. Windows CI must compile test fixtures without relying on
class template argument deduction for `std::array`. C++ lint must pass without
the warning exclusions from the old PR.

Acceptance requires a clean diff against current `main`, regenerated stubs,
passing focused and full local checks, and passing exact-head GitHub checks. Old
PR checks do not count as validation of the rewritten branch.

## Idempotence and Recovery

All build and test commands are repeatable. Build output stays under `build/`.
Registry discovery reads configuration but does not mutate it. If a direct
object-model milestone fails, keep the old driver files until the equivalent
focused tests pass; remove them only in the subsequent subtractive step.

Before any remote update, fetch `origin/main` and PR #1901 again. If `main`
advanced, rebase the fresh local commit series and rerun affected checks. Push
the verified tip to `agent/qdmi-integration-redesign` only with
`--force-with-lease` against the refreshed old PR head.

## Artifacts and Notes

PR #1901 currently has one outdated but unresolved ownership discussion and two
current style discussions. The ownership reply must explain that the manager is
an immutable, non-singleton snapshot and that returned objects own their state.
The style replies must point to `.hpp` private headers and conventional return
types. Every public GitHub body must begin with `🤖 *AI text below* 🤖`.

The old PR failed Windows because a test used `std::array` class template
argument deduction without a portable declaration. It also failed C++ lint on
raw custom deletion, missing includes, and style warnings. The rewrite must use
explicit fixture types, direct includes, and RAII instead of suppressing those
diagnostics.

## Interfaces and Dependencies

The final C++ interface must provide:

    qdmi::DeviceRegistry()
    qdmi::DeviceRegistry(std::vector<qdmi::DeviceDefinition>)
    qdmi::DeviceRegistry::definitions() const
    qdmi::DeviceRegistry::deviceIds() const
    qdmi::DeviceRegistry::registerDevice(definition, replace = false)
    qdmi::DeviceRegistry::registerDeviceIfAbsent(definition)
    qdmi::DeviceManager()
    qdmi::DeviceManager(qdmi::DeviceRegistry)
    qdmi::DeviceManager::definitions() const
    qdmi::DeviceManager::deviceIds() const
    qdmi::DeviceManager::open(id, overrides = {}) const
    qdmi::DeviceManager::openAll(overrides = {}) const
    qdmi::registerDevice(definition, replace = false)
    qdmi::registerDeviceIfAbsent(definition)
    qdmi::registeredDeviceIds()
    qdmi::openDevice(id, overrides = {})

`OpenAllResult` contains `std::map<std::string, Device> devices` and
`std::map<std::string, std::string> errors`. Unknown IDs and load or session
errors preserve the current `open_device` exception behavior. `openAll` catches
each standard exception, records its message under that ID, and continues.

Python exposes `DeviceDefinition`, `DeviceRegistry`, `DeviceManager`,
`OpenAllResult`, `register_device`, `register_device_if_absent`,
`registered_device_ids`, and `open_device` from `mqt.core.qdmi.driver`. Device,
site, operation, job, and custom-property types remain in `mqt.core.qdmi`.

Revision note (2026-08-18): This plan replaces the historical PR #1901 plan. It
starts from current `main`, selects the hybrid default and explicit-instance
API, removes obsolete neutral-atom and TOML assumptions, and records the open
review and CI requirements.
