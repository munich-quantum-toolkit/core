# Unify QDMI device management

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current as the work changes.

This plan follows `.agent/PLANS.md`. The configurable QDMI device foundation is
already part of `main`; this change replaces its overlapping runtime layers with
one public object model.

## Purpose / Big Picture

Applications should manage QDMI devices through one sequence:

    configuration -> DeviceRegistry -> DeviceManager -> Device
                                                       |-> Site / Operation
                                                       `-> Job / child Device

`DeviceRegistry` discovers and combines device definitions without executing
device code. `DeviceManager` owns a registry snapshot and opens a fresh session
for a selected stable ID. Every returned object retains the library and session
state required by its QDMI handle.

This makes device discovery inspectable, isolates failures between devices, and
gives C++, Python, Qiskit, and the neutral-atom adapter the same lifecycle
model. The prior FoMaC and client-driver public APIs are removed as part of the
MQT Core v4 transition.

## Progress

- [x] (2026-07-15) Added the public `Device`, `Site`, `Operation`, and `Job`
  objects using QDMI's enum types and typed custom-property queries.
- [x] (2026-07-15) Added lazy opening, per-ID bulk-open results, child devices,
  and lifetime-safe derived objects.
- [x] (2026-07-15) Migrated the Python bindings, Qiskit integration, and
  neutral-atom adapter to `mqt.core.qdmi`.
- [x] (2026-07-27) Consolidated registration in `DeviceRegistry` and made
  `DeviceManager` an immutable snapshot.
- [x] (2026-07-27) Preserved `qdmi.json`, `[tool.qdmi]`, manifest discovery,
  disabled-ID masking, target metadata, and path-like Python arguments.
- [x] (2026-07-27) Serialized replacement library generations and covered
  initialization, finalization, cross-session handles, and object lifetimes.
- [x] (2026-07-27) Passed the focused native, Python, documentation, stub, and
  lint checks on the configuration branch.
- [x] (2026-07-30) Rebased only the redesign and documentation commits onto
      current `main`, which already contains the configurable-device and
      mandatory LLVM/MLIR changes.
- [x] (2026-07-30) Preserved the binary-safe submission and retrieval contract
      added to `main` after the original branch, including C++, Python, and
      stubs.
- [x] (2026-07-30) Passed the complete native build and 3,882 CTest cases, the
  Python 3.14 suite, stub generation, documentation, and full lint.
- [x] (2026-07-30) Integrated the optional bundled-device controls from #1965; a
      clean build with all bundled devices disabled retains and passes the 12
      device-independent registry tests.

## Surprises & Discoveries

- Replaying the original work beside the configurable-device implementation
  produced two parsers, two registries, and two stable-ID opening paths. The
  useful boundary is one mutable `DeviceRegistry` followed by an immutable
  `DeviceManager`.
- A QDMI library may permit only one live initialization while callers still
  need independent device sessions. A process-wide weak `DeviceApi` cache shares
  compatible live libraries without keeping them loaded indefinitely.
- Child devices, jobs, sites, and operations can outlive their manager or parent
  wrapper. Keeping the internal session state in the object graph makes those
  handles safe without a separate public session object.
- Current `main` builds LLVM/MLIR and QIR support unconditionally. The redesign
  must preserve the MLIR binding and validate with LLVM/MLIR 22 available.
- The post-branch binary-program work initially disappeared with FoMaC. A
  focused compile against the tests from `main` exposed the missing byte
  overload, which now belongs directly to the unified QDMI object model.
- Optional bundled devices require test dependencies to follow capabilities:
  registry tests run without devices, manager tests require only the
  superconducting device, and object-model tests require all three built-ins.

## Decision Log

- Decision: `DeviceRegistry` is the only mutable discovery and fallback
  registration boundary. Rationale: packages can supply a device definition
  without mixing configuration mutation into runtime management. Date/Author:
  2026-07-27, implementation review.
- Decision: `DeviceManager` owns an immutable registry snapshot. Rationale:
  opening sessions does not require singleton state or a second registration
  API. Date/Author: 2026-07-27, implementation review.
- Decision: disabled IDs remain reserved. Rationale: fallback registration must
  not undo an explicit higher-precedence disable. Date/Author: 2026-07-27,
  configuration integration review.
- Decision: cache `DeviceApi` by canonical library path and symbol prefix using
  weak ownership. Rationale: compatible sessions share one live initialization,
  and the library unloads after its last object is gone. Replacement waits for
  the prior generation to finish finalization. Date/Author: 2026-07-27,
  lifecycle review.
- Decision: retain runtime state directly in the device object graph. Rationale:
  public session ownership adds another layer but does not improve handle
  safety. Date/Author: 2026-07-27, API review.

## Context and Orientation

The public C++ interfaces are:

- `include/mqt-core/qdmi/DeviceRegistry.hpp`
- `include/mqt-core/qdmi/DeviceManager.hpp`
- `include/mqt-core/qdmi/Device.hpp`

The implementation is in `src/qdmi/`. Private `DeviceApi` owns the dynamic
library and exact QDMI function pointers; private `DeviceState` owns one device
session. Python bindings and stubs are in `bindings/qdmi/qdmi.cpp` and
`python/mqt/core/qdmi.pyi`.

Configuration remains in `DeviceRegistry.cpp` and `docs/qdmi/configuration.md`.
Qiskit integration is under `python/mqt/core/plugins/qiskit/`; the neutral-atom
adapter is under `src/na/qdmi/`.

## Plan of Work

1. Move configuration definitions and registration into the public QDMI object
   model while retaining all discovery and precedence behavior from `main`.
2. Open each stable ID through `DeviceManager`, overlaying per-open session
   parameters and isolating bulk-open failures by ID.
3. Keep the loaded library and session alive through the returned object graph;
   reject cross-session handles before invoking device code.
4. Bind the model directly in Python, migrate Qiskit and neutral-atom callers,
   and remove the superseded FoMaC and client-driver layers.
5. Update migration and API documentation, regenerate stubs, and validate the
   complete branch.

## Concrete Steps

From the repository root, with `MLIR_DIR` pointing to LLVM/MLIR 22:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target \
      mqt-core-qdmi-object-model-test \
      mqt-core-qdmi-manager-test \
      mqt-core-qdmi-registry-test \
      mqt-core-na-qdmi-test
    ./.agent/run.sh ctest --test-dir build/release --output-on-failure

Then validate generated and user-facing surfaces:

    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uvx nox -s tests-3.14
    ./.agent/run.sh uvx nox -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check

## Validation and Acceptance

Acceptance requires:

- registry construction does not initialize device code;
- configuration discovery, explicit definitions, fallback registration, and
  disabled-ID masking behave as documented;
- every open creates a fresh session while compatible live sessions share one
  library initialization;
- bulk opening isolates failures by stable ID;
- devices and derived objects remain valid after their manager is destroyed;
- Qiskit and neutral-atom integrations use `mqt.core.qdmi`;
- generated stubs match the bindings; and
- native and Python tests, documentation, lint, and `git diff --check` pass, or
  any environmental limitation is recorded.

## Idempotence and Recovery

Configuration and build commands are repeatable. Build outputs remain under
`build/` and agent caches under `.cache/`; neither is committed. Re-run CMake
after build-system changes. Regenerate stubs from the bindings rather than
editing generated signatures by hand.

## Outcomes & Retrospective

The reconstructed branch contains only the v4 device-management redesign on top
of current `main`; the already-merged configuration foundation is no longer
duplicated in its history or diff. The complete release build with LLVM/MLIR 22
passes all 3,882 CTest cases; two device job-ID cases are intentionally skipped
by their test fixtures. The Python 3.14 suite passes 397 tests with three
upstream-Qiskit skips. Stub generation, warning-as-error documentation, full
lint, and `git diff --check` also pass. A separate configuration with all three
bundled QDMI devices disabled builds and passes the 12 remaining registry tests.

## Artifacts and Interfaces

The principal interfaces are:

    qdmi::DeviceRegistry()
    qdmi::DeviceRegistry(std::vector<qdmi::DeviceDefinition>)
    qdmi::DeviceRegistry::registerDevice(definition, replace)
    qdmi::DeviceRegistry::registerDeviceIfAbsent(definition)
    qdmi::DeviceManager()
    qdmi::DeviceManager(qdmi::DeviceRegistry)
    qdmi::DeviceManager::open(id, sessionOverrides)
    qdmi::DeviceManager::openAll(sessionOverrides)

Python exposes the corresponding `DeviceDefinition`, `DeviceRegistry`,
`DeviceManager`, `OpenAllResult`, `SessionParameters`, `Device`, and `Job`
classes from `mqt.core.qdmi`.
