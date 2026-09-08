# Build and test performance

Status: complete; hosted CI remains pending.

## Outcome and scope

Implemented the findings in [the audit](../audits/build-test-profile.md),
including deterministic DDSIM lifecycle tests, complete QDMI discovery, smaller
sampling allocations, incremental builds, wheel size, and CI scheduling. QDMI
owns header generation in
[QDMI #537](https://github.com/Munich-Quantum-Software-Stack/QDMI/pull/537).
The shared workflows own caching and isolated wheel test jobs in
[Workflows #462](https://github.com/munich-quantum-toolkit/workflows/pull/462).
Core pins both changes.

## Decisions

- Preserve all test cases and normal C++ installation versioning.
- Share device implementation objects with tests to synchronize the real worker
  without exporting test hooks or compiling the implementation twice.
- Keep simulation integration cases and the actual one-second timeout check.
- Batch only QC/QCO IR binaries; retain per-case XML and check shuffled order.
- Build one Stable ABI wheel per platform, then resolve each test environment's
  dependencies independently. Separate checkouts prevent lockfile/build races.
- Preserve default source-building Nox behavior outside the explicit wheel path.

## Validation

The audit records local checks and measured limits. Hosted CI is pending; local
ARM64 wheel validation does not establish other platform results.
