# README and next-release documentation audit

Status: findings applied and locally validated. Current baseline: main
`fce58f02d`, with compiler prerequisite `1f7defd48` (PR #2506).

## Result

1. Replace the outdated feature list and buried example with six concrete
   capability groups and an executable compile-and-submit workflow.
2. Route readers from the documentation landing page to examples, guides, and
   APIs. Put MQSC before CDA/TUM and preserve the funding attribution.
3. Distinguish executable DDSIM devices from compilation-only SC models.
4. Correct the installed C++ reference scope and the OpenQASM exporter's
   classical-index support.

The PR #2519 rebase retains its API names and corrected compiler terminology.
The RUS demonstration is integrated into `benchmarks.md`, with MyST `math`
directives for displayed equations. README and documentation prose link to MQSC
by its short name; full legal names remain only in copyright notices. The
glossary expands MQSC and CDA and identifies both as developers of MQT Core. The
compiler guide is now `mlir/mqt_compiler_collection.md` and introduces the
Python, C++, and `mqt-cc` interfaces. All navigation uses the new page, and RUS
links target its benchmark section. The landing-page device note and the README
Shor implementation aside are removed.

## Contracts and evidence

- `bindings/mlir/register_mlir.cpp`, `mlir/lib/Compiler/Pipeline.cpp`, and
  `docs/mlir/target_compilation.md` define device-directed compilation and
  submission. The README uses those public APIs. The docs session extracts its
  sole Python block; MyST loads and executes that exact source.
- `mlir/bench/programs/QPE.cpp` and `src/bench/QPE.cpp` define the generated
  phase-gate benchmark and analytic distribution. Eight-bit standard and
  iterative programs use nine and two qubits before target optimization. QIR
  runtime outputs follow recording order; DDSIM normalizes them before returning
  QDMI shots and counts. Examples pass counts directly to evaluation.
  Exact-phase and sampled non-exact-phase checks exercise both the compiler and
  the result contract.
- `mlir/bench/programs/RepeatUntilSuccess.cpp`,
  `src/bench/RepeatUntilSuccess.cpp`, and the existing `docs/benchmarks.md`
  define the retry circuit and phase-sensitive parity reference. The tutorial
  checks its sampled distribution. It does not infer retry counts from parity or
  claim error correction.
- `src/qdmi/devices/sc/Device.cpp` rejects job creation and execution with
  `QDMI_ERROR_NOTSUPPORTED`. Its calibration and topology describe compiler
  targets. The README, QDMI index, and SC guide describe this scope.
- `docs/Doxyfile` generates the installed `include/mqt-core` reference.
  `mlir/include/mqt/Dialect/QIR` belongs to the source-tree interface.
  `docs/cpp_api.md` separates these scopes and compiles its displayed DD/CMake
  example against the installed wheel, then checks the four amplitudes. Its
  Ninja generator requires Ninja in the docs environment. Configure and build
  diagnostics are retained in notebook output so hosted failures are visible.
- `docs/mlir/OpenQASM.md` and the frontend/export implementation permit runtime
  classical-bit indices while requiring statically resolved exported qubit
  indices. The compiler overview now matches that distinction and retains the
  runtime-assertion limitation.

## Retained scope and limits

Reassessed the entry points and current compiler, benchmark, QDMI, QIR, and
build recipes alongside the earlier `documentation.md` audit. Existing DD
algebra checks, SDK examples, strict notebook execution, and generated HTML
navigation checks remain in place. Historical changelogs and generated API
descriptions were not individually re-audited. Template-owned installation and
contribution pages remain template-owned; installation already documents LLVM
23.1 setup. The docs dependency group supplies Ninja for the C++ example; no new
documentation framework is needed.

The adaptive examples depend on
[PR #2506](https://github.com/munich-quantum-toolkit/core/pull/2506).
This work targets the next release; new stable-page links become available with
that publication. No remote device, live calibration, Slurm cluster, or
non-Linux platform was exercised. Validation results are in the companion
execution plan.
