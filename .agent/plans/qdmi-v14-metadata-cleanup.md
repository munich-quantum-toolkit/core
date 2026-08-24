# Adopt QDMI metadata removal

Status: implemented and locally validated; hosted CI remains the merge gate.

## Goal and scope

Remove Core's calibration-advisory accessor and the bundled devices' obsolete
pulse metadata after QDMI PRs #512 and #513 remove these properties. This is
Core 4.1 work, independent of driver replacement, program capabilities, and
multi-program jobs. It must not enter the Core 4.0 release.

## Decisions

Keep calibration-job submission and the calibration status: these are not the
removed advisory. Keep the existing program-format enum and job interfaces. Pin
the independent QDMI cleanup while developing; replace that pin with a released
QDMI 1.4 before publishing artifacts. Clients and devices must use matching
headers because the regular property values change.

The affected interfaces are in `include/mqt-core/qdmi/Client.hpp`,
`src/qdmi/Client.cpp`, and `bindings/qdmi/qdmi.cpp`. Bundled-device changes live
under `src/qdmi/devices/`. Remove only tests for the deleted API; preserve the
current optional-DDSIM build coverage and unrelated concurrency behavior.

## Validation

The release build and CTest passed (3,693 passed, one existing skip). All 249
QDMI Python tests passed with both bundled devices enabled. Stub generation,
lint, and C++ lint passed. Keep both bundled devices enabled for the full Python
suite; the stub-generation environment deliberately disables the SC device.
