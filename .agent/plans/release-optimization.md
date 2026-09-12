# Release wheel optimization

Status: in progress; final macOS selection and integrated hosted qualification
remain outstanding.

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
- The two macOS native-SDK cohorts select ThinLTO for Core and combined SDK/Core
  PGO. Qualify that recipe while the final matched-SDK trial finishes; its
  adoption gate remains pending.
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

## Work remaining

- [ ] Complete macOS final runtime comparisons and record the selected LTO and
      PGO scope. Both Linux architectures retain native SDK libraries after
      their completed paired comparisons.
- [ ] Qualify the actual cibuildwheel hooks with LLVM 23.1.1 trial SDK artifacts
      on both Linux architectures and Python ABIs, macOS ARM64, and Windows.
- [ ] Reconcile the companion PRs and study report with final measurements and
      hosted status. Do not publish an SDK release as part of qualification.

## Validation

The focused profile-counter test in `test/python/test_release_preparation.py`
passes. Training passes against the current Core API with NumPy imports blocked.
The source distribution contains all release training and consumer sources.
Repository lint and whole-file C++ lint pass. The measured study recipes use
frozen Core revisions; those results are distinct from qualification of the
current integration. See the
[SDK study](https://github.com/munich-quantum-software/portable-mlir-toolchain/blob/codex/optimized-release-toolchain/experiments/RESULTS.md)
for runtime samples, artifact identities, resource measurements, and limits.
