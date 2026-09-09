# QDMI driver opening performance

Status: complete. Both confirmed findings are implemented and validated.

## Scope and decisions

The default driver owns provider initialization and configuration forwarding.
Keep the existing process-lifetime provider cache and its loader-handle/prefix
identity. A module now owns its initialization mutex and provider map. The
global mutex protects only module lookup/insertion. Map entries are never
erased, so references remain stable after releasing that mutex. Each opener
holds a loader reference while waiting; failed initialization can be retried
without exposing a partial provider. Providers in the same module remain
serialized because they may share internal state.

Fresh opening still snapshots the registered definition while holding the
registry mutex. Move that owned session configuration into the by-value merge
helper. Parameter forwarding borrows NUL-terminated strings through optional
string views; all calls remain synchronous and path temporaries survive the
call. Preserve empty overrides, validation, and parent/child ownership.

The audit's unpromoted candidates remain deferred. No registry index, lazy
catalog/child opening, job pool, metadata cache, or additional loader cache is
part of this change. Related PRs #2229/#2230 alter the client/default-driver
boundary; these fixes apply to the current default driver independently.

## Validation

Both new regressions and the complete native driver binary pass: 106 tests. The
full QDMI CTest tree passed 478 tests with one existing skip. Repository lint
and changed-file C++ lint passed with zero findings. Focused probes confirm
unrelated opening completes before a gated initializer is released, and the
typed configuration path allocates one payload-sized snapshot instead of three.
See the [audit](../audits/qdmi-driver-performance-2026-09-08.md) for scope and
evidence.
