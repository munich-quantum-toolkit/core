# Build a target environment from one QDMI payload

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Compiler users must not reconstruct payload facts from a format name. After this
change, one C++ or Python factory snapshots a QDMI device and an exact accepted
descriptor into a detached `TargetEnvironment`. The result contains both the
hardware target and the selected payload capabilities.

## Progress

- [x] (2026-08-24 06:25Z) Added C++ conversion for an open device and a stable
      device ID.
- [x] (2026-08-24 06:25Z) Added Python `TargetEnvironment.from_device` and
      `from_device_id` factories with session overrides.
- [x] (2026-08-24 06:25Z) Added exact-format, grouped-feature, QIR baseline, and
      rejection tests.
- [x] (2026-08-24 14:30Z) Passed all 10 focused C++ adapter tests and both
      focused Python factory tests on the final stack.
- [x] (2026-08-24 14:30Z) Passed the release build, all 4,073 configured CTest
      cases, and stub generation on the final stack.

## Surprises & Discoveries

- Observation: QDMI reports one constrained feature group as several records
  with the same feature ID and value. Evidence: each record contains one
  `constraint_id` and `constraint_value` pair.
- Observation: standard descriptor baselines are implicit. Evidence: QIR 2.1
  Adaptive guarantees five control-flow features even when the optional list is
  empty.

## Decision Log

- Decision: Require the selected descriptor to equal one device-supported value.
  Rationale: compilation must not claim a payload the device rejects.
  Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.
- Decision: Group records by feature ID and value and preserve constraints.
  Rationale: different values are alternatives, while constraints in one group
  are conjunctive. Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.
- Decision: Add the QIR Adaptive baseline in the adapter and keep optional-set
  completeness separate. Rationale: QDMI makes baseline facts implicit.
  Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.

## Outcomes & Retrospective

The implementation is complete and remains detached from a live QDMI session.
All 10 focused C++ adapter tests and both focused Python factory tests pass. The
complete stack passes the release build, all 4,073 configured CTest cases, stub
generation, documentation, and lint. The preceding atomic QDMI migration layer
also passes its complete validation set.

## Context and Orientation

`mlir::TargetEnvironment` combines a `CompilerTarget` with a validated
`PayloadSpecification`. Its definitions live in
`mlir/include/mlir/Compiler/TargetEnvironment.h`. The QDMI compatibility
boundary is `mlir/include/mlir/Compiler/QDMIAdapter.h` with its implementation
in `mlir/lib/Compiler/QDMIAdapter.cpp`. Python bindings are in
`bindings/mlir/register_mlir.cpp`.

## Plan of Work

Extend the existing QDMI adapter with factories for an open `qdmi::Device` and a
registered device ID. Validate the exact descriptor, convert its version and
encoding, group optional feature records, add the normative baseline, and
preserve whether optional metadata is complete. Snapshot the existing compiler
target and payload into one owning value. Bind both factories in Python,
regenerate `python/mqt/core/mlir.pyi`, and update compiler and DDSIM examples.

## Concrete Steps

Run from the repository root:

    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerQDMIAdapterTest.*'
    uvx nox -s stubs
    uvx nox -s tests-3.12 -- test/python/test_mlir.py \
      -k target_environment_from_device
    uvx nox --non-interactive -s docs
    uvx nox -s lint

## Validation and Acceptance

The factories reject a canonical descriptor that the device does not accept.
They preserve ID, canonical version, profile, and encoding. Records with the
same feature ID and value become one capability with all constraints. QIR 2.1
Adaptive contains each baseline capability once. A successful empty optional
query is known; `NOTSUPPORTED` remains unknown. The returned environment works
after the originating device is destroyed.

## Idempotence and Recovery

All commands are repeatable. Stub generation is the only authorized way to
change `python/mqt/core/mlir.pyi`. No step changes remote state.

## Artifacts and Notes

The adapter is stacked above the low-level QDMI 1.4 contract. It does not add a
second loader, registry, or payload inference path.

## Interfaces and Dependencies

The final C++ functions are
`targetEnvironmentFromDevice(const qdmi::Device&, const QDMI_Program_Format&)`
and
`targetEnvironmentFromDeviceId(std::string_view, const QDMI_Program_Format&)`.
Python exposes matching static factories. This layer requires the exact
descriptors and feature query from the preceding QDMI layer and
`TargetEnvironment` from the selected-payload compiler layer.

Plan update, 2026-08-24: Split this compiler adapter from the atomic QDMI
producer-and-consumer migration.
