# Shared compilation options

Status: complete.

## Scope and ownership

`CompilationOptions` groups timing, statistics, an optional compilation seed,
and native mapping controls in C++, Python, and `mqt-cc`. Compiler entry points
accept these settings only through this object. Core owns compilation.

## Decisions

An explicit seed overrides MQT pass-local seeds, including custom pipelines and
nested modules. Omission preserves existing defaults and pass settings. Passes
read a verified, scoped `mqt.compilation_seed` module attribute. MLIR captures
it in crash reproducers without process-global mutable state or a second
pipeline parser. The driver restores previous metadata on success or failure.
Low-level target builders accept only `MappingOptions`. Run their pass managers
with `runWithCompilationOptions` to apply the compilation seed and
instrumentation.

Bindings copy options before releasing the GIL. Source submission distinguishes
omitted options from supplied settings so already compiled payloads reject
compiler controls.

Mapping trials and refinement iterations must be positive. Trials retain the
CPU-dependent default; iterations default to one forward/backward round. Routing
lookahead defaults to 20 additional gates and permits zero. Its storage grows
with the gates present, even when the configured ceiling is `SIZE_MAX`.
All-to-all placement ignores valid mapping controls. Repeatability requires the
same build, input, target, seed, and explicit trial count. Execution sampling
has a separate seed. Layout selection and reporting are a separate API
extension.

## Validation

The compiler and QDMI Python suites pass 158 tests, including seed precedence,
nested pipelines, 64-bit values, instrumentation, source submission, and failure
cleanup. Native validation passes 228 compiler, 115 mapping, and 61
target-synthesis tests, plus three CLI CTests. This covers stored-seed overrides
and restoration, maximum lookahead, CLI exit/diagnostic checks, and
device-independent seed forwarding. The documented CLI command succeeds. Stub
generation, repository lint, and full changed-file C++ lint pass. No changelog
or upgrade-guide entries are included.
