# Release wheel optimization

Status: complete. The selected release recipes pass hosted qualification with
trial LLVM 23.1.1 SDKs. Production activation requires the companion releases.

## Goal and scope

Build optimized release wheels with the existing architecture matrix and
deployment targets. Keep development SDK assertions enabled and native SDK tools
unchanged. The companion SDK repository owns generic library rebuilding and
profiling tools; Core owns representative training, wheel construction, repair,
and installed-package checks.

`scripts/prepare_release.py` builds an instrumented wheel, optionally rebuilds
its SDK archive dependencies, trains the installed package, verifies executed
counters, and writes the final CMake configuration. Profiles are specific to the
source, compiler, platform, and Python ABI. Missing SDKs, failed commands, and
unexecuted profiles stop the build before the final configuration is used.

## Decisions

- Linux uses the qualified prebuilt manylinux Clang 22.1.8 package, full Core
  LTO, native SDK dependency PGO, and BOLT. There is no compiler bootstrap.
- Both final macOS cohorts select native SDK libraries, ThinLTO for Core, and
  combined SDK/Core PGO. No matched candidate meets the confidence-supported
  adoption gate.
- Keep native SDK libraries unless both paired runtime cohorts meet the study's
  10% adoption gate without a confirmed workload regression above 3%.
- Split Linux and macOS wheel jobs by Python ABI to fit the five-hour budget.
  Each ABI trains a fresh profile. Windows retains its existing optimization
  settings.
- Restore two-level namespaces for macOS Python modules, matching upstream MLIR.
  Flat namespaces leave an unresolved private initializer alias in instrumented
  modules.
- Run numerical and CLI workloads after BOLT, stripping, and wheel repair. Check
  the installed CMake package with producer Clang and manylinux GCC; Python
  imports alone do not establish C++ compatibility.
- Apply final PGO and LTO through directory compile/link options. Global profile
  flags make unrelated CMake compiler probes collide with trained `main`
  functions under `-Werror`, producing false PIC failures and invalid TLS
  relocations during full LTO.
- Add device build-directory paths only for consumers using build RPATHs.
  Installed targets retain relative paths, and Linux wheel processing rejects
  absolute ELF runtime paths before stripping and repair.

## Validation

The focused profile-counter test in `test/python/test_release_preparation.py`
passes. Training passes against the current Core API with NumPy imports blocked.
The source distribution contains all release training and consumer sources.
Repository lint and whole-file C++ lint pass. The measured study recipes use
frozen Core revisions; those results are distinct from qualification of the
current integration. See the
[SDK study](https://github.com/munich-quantum-software/portable-mlir-toolchain/blob/codex/optimized-release-toolchain/experiments/RESULTS.md)
for runtime samples, artifact identities, resource measurements, and limits.

[All four Linux ABI jobs](https://github.com/munich-quantum-software/portable-mlir-toolchain/actions/runs/34721263840)
and
[both macOS ABI jobs](https://github.com/munich-quantum-software/portable-mlir-toolchain/actions/runs/34721264654)
pass at Core `1691559de4868923dacc3e565c7c00130d437773`. The actual cibuildwheel
hooks train fresh profiles, rebuild the recorded SDK dependencies, repair
wheels, and validate installed numerical, compiler, QIR, CLI, and DD behavior.
Stable-ABI jobs also pass their configured Python and C++ suites. Supplementary
C++ tests use the native release library settings; installed shared-wheel
consumers are checked separately with Clang and GCC on Linux and the recorded
Xcode on macOS.

All 13 binaries in each wheel have the expected architecture and deployment
requirements, with no build-directory runtime paths. Linux wheels retain
manylinux 2.28 compatibility, macOS binaries require 13.3, and SDK library
rebuilds retain the macOS 11.0 target. All six jobs finish within five hours on
the configured hosted runners. The report distinguishes complete job costs from
warm rebuilds and records the scope of memory measurements.

[All four Windows ABI jobs](https://github.com/munich-quantum-software/portable-mlir-toolchain/actions/runs/34709109479)
pass at Core `0fcd55068528aee5421965d66fda9c00f0955fc6`; later changes affect
Unix release preparation and runtime paths, leaving the Windows build behavior
unchanged. Native SDK builds and installed consumers pass on all five platforms
with assertions enabled and disabled.

The report retains runnable compiler-probe diagnostics, before/after
runtime-path checks, and Clang/GCC consumers tested with the original build tree
hidden. No production SDK release is part of this qualification. Companion PRs
and the assertion-free archives must land before the production release workflow
can use the selected recipes.
