# Adopt QDMI metadata removal

Status: rebased onto main and locally validated; hosted CI remains the merge
gate.

## Goal and scope

Remove Core's calibration-advisory accessor and the bundled devices' obsolete
pulse metadata after QDMI PRs #512 and #513 remove these properties. This
targets main for the next release, independent of driver replacement, program
capabilities, and multi-program jobs.

## Decisions

Keep calibration-job submission and the calibration status: these are not the
removed advisory. Keep the existing program-format enum and job interfaces. Pin
the independent QDMI cleanup while developing; replace that pin with a released
QDMI 1.4 before publishing artifacts. The removed property values remain
reserved, so surviving query IDs stay binary-compatible with released clients.

The affected interfaces are in `include/mqt-core/qdmi/Client.hpp`,
`src/qdmi/Client.cpp`, and `bindings/qdmi/qdmi.cpp`. Bundled-device changes live
under `src/qdmi/devices/`. Remove only tests for the deleted API; preserve the
current optional-DDSIM build coverage and unrelated concurrency behavior.

## Validation

QDMI #513 is pinned at `10b3b66936bf6de39f01530c7fbbc05a8a026c9c`. The release
build with Clang 23 and LLVM/MLIR 23 passed; CTest passed 3,580 tests with one
existing SC skip. The QDMI Python suite passed 298 tests with both bundled
devices enabled. Stub generation, lint, and C++ lint passed.

The default GCC release build hit duplicate symbols while linking the local LLVM
distribution. Using its matching Clang toolchain resolved the build. Python
tests require rebuilding with the SC device enabled after stub generation; the
cached stub-generation wheel omits that device.
