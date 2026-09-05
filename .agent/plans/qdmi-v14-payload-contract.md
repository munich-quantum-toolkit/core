# Independent QDMI program-capability prototype

Status: independent rebase; design decisions remain gated.

## Scope and dependencies

Core #2226 adopts the experimental format descriptors and optional execution
feature query from QDMI #508. It is a non-blocking Core 4.1 / QDMI 1.4
candidate, not Core 4.0 scope. Core #2365 and QDMI #523 must settle the contract
before implementation is merge-ready.

This runtime layer has no compiler-only #2219 ancestry, no driver replacement,
and no metadata-removal dependency. Native multi-program jobs were extracted to
Core #2362 and QDMI #509 with the existing program-format enum. This branch
retains single-program setters and unindexed results against QDMI #508. The
compiler/runtime integration layer remains in Core PR `#2227`.

## Preserved behavior and prototype boundaries

Retain optional shots, byte-exact binary transport, current DDSIM QCO-backed
simulation, session ownership, concurrent job behavior and target inference.
Unknown topology or gate sets still fail early; simulator controlled-operation
families and zero-arity global phase remain unchanged.

Format descriptors, optional feature records and text/result framing retain the
existing prototype semantics for evaluation. They are not a final answer to
format identity versus execution capabilities, supported versus native
operations, classical guarantees, opaque programs or provider-neutral verbatim
execution. Calibration status remains distinct from program vocabulary.

Keep mechanical SDK adaptations here because the same package must still import
and use the descriptor-valued runtime. Backend-owned serializers and decoders
remain part of that prototype. Core issues `#2363` and `#2364` track native SDK
batching separately. Do not replace concurrent single submissions with synthetic
aggregate jobs.

## Validation and release gate

Build independently against QDMI #508. Test descriptor validation, optional
feature metadata, text and binary submission/retrieval, optional shots, SDK
serialization/layout, asynchronous failure and concurrency. Preserve newer
mainline tests. Run stubs, repository lint and C++ lint. Check both bundled
devices and the compiler's existing device-to-target adapter.

Local validation passed 3,874 native tests with one existing skip and 399 Python
QDMI/SDK tests. Generated stubs, repository lint and C++ lint passed. Hosted CI
and contract design review remain separate gates.

Use the design trackers to record any contract change rather than silently
stabilizing one during a rebase. Published artifacts require released pins.
Preserve existing PR identity, attribution and review history; no archives or
automatic review requests.
