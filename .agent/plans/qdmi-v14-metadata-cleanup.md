# Adopt QDMI metadata and calibration removal

## Scope

Adopt QDMI #512, #513, and #551 on main. Remove the calibration advisory, pulse
metadata, calibration program format, and the C++ and Python calibration
submission helpers. Keep the calibration device status. IQM exposes calibration
submission through its own C extension in QDMI-on-IQM #266.

Keep surviving property and program-format IDs unchanged. Both providers retain
released MQT Core; no Core source build or LLVM additions are needed in their CI
matrices. Release notes and upgrade instructions belong in release prep.

QDMI #551 is pinned at `46422d82d793f6ef30b319a10ea6ed3cac27ae40`. Replace the
development pin with a released QDMI version before publishing.

## Validation

The Clang 23 release build passed with 3,578 native tests and one existing SC
skip. The QDMI Python suite passed 293 tests with both bundled devices enabled.
The 103 affected Qiskit serializer/backend tests passed. Stubs were regenerated.
Full repository lint and C++ lint cover the final diff. The provider drafts
retain released Core 4.0.0 and validate their own native and Python suites; IQM
also checks the installed C11 extension interface.
