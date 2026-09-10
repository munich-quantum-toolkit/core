# Local Linux optimization experiments

Status: complete. The portable, CPU-tuning, host-toolchain, O2/O3, and all 24
PGO configurations are evaluated. Four quiet runtime cohorts, capped finalist
replays, installed-package checks, and source-build instructions are complete.

## Scope

Compare fixed LLVM/MLIR 23.1.0 and Core sources on local ARM64. Evaluate GCC and
downloaded Clang 23, SDK/Core LTO, patched mold/BFD/LLD, CPU tuning, compiler
PGO, and Core BOLT. Preserve numerical and installed-package contracts. No
macOS, hosted CI, compiler bootstrap, or release changes.

## Implementation

Reuse the SDK source builds and Core release training. Keep experimental
configuration in a local runner and separate build artifacts. Record exact
compiler/linker identities and executable hashes, compile/link options, resource
measurements, package sizes, and held-out runtime samples.

The portable matrix is GCC off/off, off/full, full/full and Clang off/off,
off/full, thin/thin, thin/full, full/full for SDK/Core LTO. Compare both
appropriate linkers before/after BOLT. Then test Core-only and SDK-plus-Core
native CPU tuning in manylinux, plus portable/native host builds for both
compilers. Compare O2/O3 and staged SDK/Core PGO on finalists.

Use regular C++/MLIR/Python tests and scalable compiler workloads for training.
BOLT profiles must come from the final binaries being rewritten. Keep evaluation
inputs separate. Measure twelve rotating fresh-process rounds without concurrent
builds.

## Remaining work

- [x] Verify prebuilt Clang and patched-linker provisioning.
- [x] Complete portable matrix and BOLT comparisons.
- [x] Complete CPU-tuning and host matrix.
- [x] Complete finalist optimization-level and PGO comparisons.
- [x] Validate packages and runner-limited stages; report measured
      recommendations.
- [x] Document reproducible portable and native Linux source builds.

## Decision record

Prefer portable Clang 23.1.0, LLD, O3, full SDK/Core LTO, benchmark-only
combined compiler PGO, and BOLT trained on Python tests plus scalable
benchmarks. This is also the recommendation for maximum measured local
performance: the host-native alternative is tied overall and regresses DD
simulation. Four LTO workers with one large link at a time passed the four-CPU,
16 GiB resource screen. Keep the host-native recipes and per-workload regression
tables for workload-specific choices. Release automation and other platforms
remain outside this phase.
