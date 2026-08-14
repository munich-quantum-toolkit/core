# Establish the QDMI Python namespace

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Python users must be able to distinguish QDMI entities from the layer that
registers and opens QDMI devices. After this change, users import entities from
`mqt.core.qdmi` and device-registry operations from `mqt.core.qdmi.driver`. MQT
Core v3 programs that import `mqt.core.fomac` continue to work, but the
compatibility module warns that it will be removed in MQT Core 4.0. The C++
FoMaC API does not change.

The result is visible by importing `Device` from `mqt.core.qdmi`, opening
`mqt.ddsim.default` through `mqt.core.qdmi.driver.open_device`, and running a
job. The compatibility module re-exports the same objects.

## Progress

- [x] (2026-08-13 00:56Z) Rename the native extension and create the `driver`
      submodule.
- [x] (2026-08-13 00:56Z) Add the v3 compatibility module and deprecation
      warnings.
- [x] (2026-08-13 00:56Z) Move MQT Python integrations and tests to the new
      namespace.
- [x] (2026-08-13 00:56Z) Generate recursive QDMI stubs and run focused Python
      tests.
- [x] (2026-08-13 01:32Z) Update user documentation and upgrade guidance; defer
      the changelog link until the pull request has a number.
- [x] (2026-08-13 03:08Z) Run packaging, documentation, lint, and aggregate
      tests.
- [x] (2026-08-13 11:30Z) Address review feedback, remove superseded design
      notes, and complete an adversarial implementation review.
- [x] (2026-08-13 12:00Z) Move the neutral-atom Python specialization to
      `mqt.core.na.qdmi`, retain its v3 alias, and add direct stable-ID Qiskit
      construction.

## Surprises & Discoveries

- Observation: The Python `Session` does not own a device registry. Explicit
  device opening delegates to `qdmi::Driver::get().openFresh`.
- Observation: Reusing one local ABI3 build across Nox interpreters can retain
  the device set from an earlier CMake configuration. Evidence: the reused
  Python 3.12 environment exposed only DDSIM, while the clean wheel exposed all
  five built-in device IDs and passed the same suite. CI builds each matrix job
  in a clean checkout and does not share this artifact.

## Decision Log

- Decision: Expose registry operations as functions in `mqt.core.qdmi.driver`.
  Rationale: The present API does not provide isolated registry instances.
  Date/Author: 2026-08-13, Lukas Burgholzer and Codex.
- Decision: Keep C++ FoMaC names and the `MQT::CoreFoMaC` target. Rationale:
  FoMaC remains the C++ wrapper library above the QDMI client interface. The
  change only corrects the public Python namespace. Date/Author: 2026-08-13,
  Lukas Burgholzer and Codex.
- Decision: Make `mqt.core.fomac` a Python module that aliases native objects.
  Rationale: Aliasing preserves exact type identity and v3 source compatibility
  without a second extension or wrapper hierarchy. Date/Author: 2026-08-13,
  Lukas Burgholzer and Codex.
- Decision: Publish the neutral-atom specialization as `mqt.core.na.qdmi` and
  keep `mqt.core.na.fomac` as a v3 alias. Rationale: This completes the Python
  namespace transition without duplicating native wrapper types. Date/Author:
  2026-08-13, Lukas Burgholzer and Codex.

## Outcomes & Retrospective

The native import, compatibility identity, device operations, and affected
Python integrations pass focused tests. The recursive stubs, editable build,
wheel build and isolated wheel install also pass. Documentation and all lint
hooks pass. Platform CI and independent review remain.

## Context and Orientation

QDMI defines a low-level device interface and a client interface. QDMI device
libraries implement the device interface. `src/qdmi/driver` loads these
libraries and implements the client interface for MQT Core. `src/fomac` and
`include/mqt-core/fomac` contain the C++ FoMaC wrappers that give QDMI handles
normal C++ ownership and methods.

Before this change, `bindings/fomac/fomac.cpp` exposed the wrappers as the
native module `mqt.core.fomac`. The same binding also exposes stable device
registration and opening functions backed by the singleton C++ driver.
`noxfile.py` runs `nanobind.stubgen` for installed native modules, and generated
stubs live in `python/mqt/core`.

## Plan of Work

Move the binding sources to `bindings/qdmi` and build the native module as
`mqt.core.qdmi`. Bind `Device`, `Job`, `ProgramFormat`, `CustomProperty`, and
the nested `Device.Site` and `Device.Operation` types in that module. Create the
`driver` submodule for `DeviceDefinition`, the legacy `Session`, and device
registration, discovery, and opening functions.

Replace `python/mqt/core/fomac.pyi` with `python/mqt/core/fomac.py`. The module
must emit `DeprecationWarning` with stack level two and then alias the public
objects from `mqt.core.qdmi` and `mqt.core.qdmi.driver`. The legacy `Session`
constructor must also warn. Update MQT's Qiskit, PennyLane, MLIR, and
neutral-atom integration points to import the new namespace. Do not make normal
integration use construct the deprecated session.

Regenerate recursive stubs through the repository Nox session. Update the QDMI
driver and configuration guides, affected examples, `UPGRADING.md`, and
`CHANGELOG.md`.

## Concrete Steps

Run all commands from the repository root through the worktree-local wrapper:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    CMAKE_ARGS=-DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
      ./.agent/run.sh uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    CMAKE_ARGS=-DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
      ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uv run --no-sync pytest test/python/qdmi -q
    ./.agent/run.sh uv run --no-sync pytest test/python/plugins/qiskit \
      test/python/plugins/qdmi_pennylane test/python/test_mlir.py -q
    ./.agent/run.sh uvx nox -s docs
    ./.agent/run.sh uvx nox -s lint

Stub generation must create `python/mqt/core/qdmi/__init__.pyi` and
`python/mqt/core/qdmi/driver.pyi`. The focused test runs must finish without a
failure or an unexpected deprecation warning.

## Validation and Acceptance

The namespace test must prove that QDMI entities live in `mqt.core.qdmi`, that
`Site` and `Operation` remain nested under `Device`, and that the compatibility
module exports the exact QDMI and driver objects. Constructing `driver.Session`
and importing `mqt.core.fomac` must each emit the documented warning.

The QDMI Python tests must still cover runtime registration, custom operation
queries, queue properties, job retrieval, device and job ownership, binary
programs, and execution. Qiskit, PennyLane, MLIR, and neutral-atom tests must
import the new module. A wheel and an editable installation must import on all
supported platforms. Generated-file, type, spelling, documentation, and lint
checks must pass.

## Idempotence and Recovery

Stub generation, formatting, tests, and documentation builds are repeatable. All
caches and build output remain inside this worktree. If a dependency download
fails, rerun the same command; do not change sources to hide a network failure.
If the upstream base advances before publication, rebase this bottom stack layer
first and rerun the affected checks before rebasing upper layers.

## Artifacts and Notes

Focused evidence at the current implementation point:

    310 passed in 6.18s
    177 passed in 23.90s
    Python 3.10: 571 passed, 6 skipped
    Python 3.11: 603 passed, 3 skipped
    Clean wheel on Python 3.12: 603 passed, 3 skipped
    Python 3.14: 603 passed, 3 skipped
    Sphinx build succeeded with warnings treated as errors
    All lint hooks passed
    Wheel built and imported in an isolated Python 3.12 environment

The first line covers the QDMI wrapper and namespace tests. The second covers
Qiskit, PennyLane, and MLIR integrations.

## Interfaces and Dependencies

The final Python interfaces include:

    from mqt.core.qdmi.driver import (
        open_device,
        register_device,
        register_device_if_absent,
        registered_device_ids,
    )
    register_device(definition)
    register_device_if_absent(definition)
    registered_device_ids()
    open_device(device_id, **device_session_overrides)

`mqt.core.qdmi.Device`, `mqt.core.qdmi.Job`, `mqt.core.qdmi.ProgramFormat`,
`mqt.core.qdmi.CustomProperty`, and the nested status enumerations remain native
objects. `driver.DeviceDefinition` and `driver.Session` remain native objects;
`driver.Session` is a deprecated v3 compatibility API. The C++ FoMaC headers,
namespace, library, and CMake target remain unchanged.

Revision note: Created after the native import and focused integration proof so
the plan records the verified singleton and ownership boundaries.
