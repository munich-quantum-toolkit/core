# Replaceable QDMI Client driver

Status: independent rebase validated locally.

## Scope and decisions

Core's Client wrappers load a validated implementation of the standard QDMI
Client ABI rather than link to Core's packaged driver. Driver selection is
process-wide; failed validation or allocation must leave retry possible. Owning
wrappers retain their originating session and the loaded function table.

This change depends only on QDMI #511. Keep the existing program-format enum,
single-program APIs, calibration submission, and current compiler and SDK
behavior. Multi-program adoption and payload capabilities are independent work.
The optional private discovery/configuration extension is in Core PR #2230.
Installed deployment is in Core PR #2231. Standardizing that extension belongs
to QDMI v2, not this Client ABI change.

Target Core 4.1 / QDMI 1.4, never Core 4.0. Development uses the isolated QDMI
driver branch; published artifacts require a released dependency version.

## Implementation boundary

The runtime is in `src/qdmi/Client.cpp` and `include/mqt-core/qdmi/Client.hpp`.
The packaged driver reports stable catalogue IDs through the standard property.
Bindings, SDK entry points, Slurm selection, and compiler device opening route
through the Client session; they must not call the packaged registry directly.
Existing compiler target inference still rejects unknown topology and gate sets.

## Validation

Validate ABI/symbol rejection, retry after failed allocation, process-wide
selection, session lifetime, malformed results, and packaged-driver loading.
Retain current optional-device builds and Slurm status semantics. Run
independent release build/CTest, QDMI and SDK Python suites, generated stubs,
repository lint, and C++ lint before publication. The release suite passed 3,869
tests with one existing skip; all 455 selected Python tests passed. Stub
generation, repository lint, and C++ lint passed.
