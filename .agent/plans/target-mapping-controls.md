# Shared compilation options

Status: complete.

## Scope and ownership

`CompilationOptions` groups timing, statistics, an optional compilation seed,
and native mapping controls in C++, Python, and `mqt-cc`. Existing calls with
timing/statistics flags remain supported. Core owns compilation; no Qiskit
compiler code is added.

## Decisions

An explicit seed overrides MQT pass-local seeds, including custom pipelines and
nested modules. Omission preserves existing defaults and pass settings. Passes
read a verified, scoped `mqt.compilation_seed` module attribute. MLIR captures
it in crash reproducers without process-global mutable state or a second
pipeline parser. The driver restores previous metadata on success or failure.
Low-level target builders also populate native MLIR pass options.

Mapping trials must be positive; omission retains the CPU-dependent default.
All-to-all placement ignores valid trials. Repeatability requires the same
build, input, target, seed, and explicit trial count. Execution sampling has a
separate seed. Layout selection and reporting are a separate API extension.

## Validation

The compiler Python suite passes 106 tests, including seed precedence, nested
pipelines, 64-bit values, instrumentation compatibility, source submission, and
failure cleanup. Native validation passes 227 compiler tests, 312 decomposition
tests, 32 metadata tests, and the three existing CLI CTests. The compiler tests
include CLI exit/diagnostic checks and device-independent seed forwarding.
Numerical retry tests preserve reconstructed unitaries and repeatable results.
Stub generation, repository lint, and full changed-file C++ lint against
`origin/main` pass. No changelog or upgrade-guide entries are included.
