# QDMI-to-compiler program-capability adapter

Status: locally validated, design-gated prototype.

## Scope and decisions

Core PR `#2227` snapshots an open QDMI device and an exact accepted program
format into an owning compiler target environment. The runtime descriptors and
feature query belong to Core PR `#2226`; the detached compiler model belongs to
Core PR `#2219`. Neither foundation depends on this adapter or the other
foundation. A temporary integration base combines them only for testing.

The adapter preserves format identity and encoding, grouped optional features,
and the prototype's standard-format baseline. Unknown optional feature metadata
remains distinct from an empty complete list. Unknown topology or operation
support still fails during target inference; existing simulator control families
and zero-arity global phase are unchanged.

Bindings reuse one session-opening helper with configuration validation at the
Python boundary. The resulting environment remains valid after the device
session closes. Tests use explicit known topology in their fake provider.

## Design and release boundary

QDMI issue `#523` and Core issue `#2365` must settle program-capability
semantics before this prototype is merge-ready. It is a non-blocking Core 4.1
candidate, never a Core 4.0 dependency. Native multi-program jobs, driver
replacement, metadata removal, and compiler control-flow passes remain
independent. Retarget the adapter to the normal development branch after both
foundations land. Release artifacts require released dependencies.

## Validation

Run the release CTest suite, Python compiler/QDMI/SDK tests, generated stubs,
repository lint and C++ lint. Cover exact-format rejection, malformed grouped
features, optional metadata, QIR baselines, snapshot lifetime and error
translation. Exercise the documented DDSIM compilation/submission path.

Local validation passed 3,891 native tests with one existing skip and 454 Python
compiler/QDMI/SDK tests, including DDSIM bitcode submission. Generated stubs,
repository lint and C++ lint passed. Hosted CI and contract review remain
separate gates.

The source is in `mlir/Compiler/QDMIAdapter`, its unit tests and
`bindings/mlir/register_mlir.cpp`; canonical usage is documented in
`docs/mlir/target_compilation.md`.
