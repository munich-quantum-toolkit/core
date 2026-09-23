# Replaceable QDMI driver

Status: rebased onto main; validation in progress.

## Scope and decisions

Core's Client wrappers load a validated implementation of the standard QDMI
Client ABI rather than link to Core's packaged driver. Driver selection is
process-wide; failed validation or allocation must leave retry possible. Owning
wrappers retain their originating session and the loaded function table.

This change depends on QDMI #511 and preserves current compiler and SDK
behavior, including the merged calibration and pulse removals. The optional
private discovery/configuration extension is in Core PR #2230. Installed
deployment is in Core PR #2231. Standardizing that extension belongs to QDMI v2,
not this Client ABI change.

Development pins QDMI #511 by immutable commit. Published artifacts require a
released dependency version.

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
repository lint, and C++ lint before publication. Validation is being repeated
after rebasing onto current main.
