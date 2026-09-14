# Release wheel optimization

Status: implementation cleanup complete. Package qualification and publication
of the dependencies remain.

## Scope and ownership

The SDK provides native libraries, assertion selection, BOLT tools, and native
library PGO rebuilding. Setup installs the requested SDK. Reusable workflows
provide runner and cache setup and split wheel jobs by ABI. Core owns training,
PGO preparation, BOLT processing, and installed-wheel checks.

Linux uses Clang 22.1.8, full Core LTO, combined SDK/Core PGO, and BOLT. macOS
uses Apple Clang, Core ThinLTO, and combined PGO. Windows keeps its existing
compiler and optimization settings. Preserve portable CPU targets, manylinux
2.28 compatibility, SDK macOS target 11.0, and Core macOS target 13.3.

## Work remaining

- [x] Remove candidate selection, benchmark evaluation, and experiment
      workflows; retain the selected training workloads and correctness checks.
- [x] Replace generic variant tooling with native library PGO rebuilding and
      keep BOLT processing in Core.
- [ ] Validate SDK installation, release preparation, repaired wheels, and CMake
      consumers; update all four PRs with signed commits.

## Evidence and release dependency

The
[completed study](https://github.com/munich-quantum-software/portable-mlir-toolchain/blob/592d4c6be117ea88dfa2cbfc44f695082fd278a8/experiments/RESULTS.md)
and its raw records remain in git history. Earlier qualification passed all ten
wheel jobs and native SDK installation on five platforms in both assertion
modes. Those results precede this cleanup and do not validate the revised code.

Core CD currently fails while downloading unpublished assertion-free SDK
archives. The existing 2026.09.11 SDK release is immutable. Publish the complete
2026.09.14 release after merging SDK PR #94, then merge setup-mlir PR #255 to
make its updated manifest available to the standalone Linux installer.
Publication and merges need human approval. Rerun Core CD after these
dependencies are available; do not add another optimization configuration.
