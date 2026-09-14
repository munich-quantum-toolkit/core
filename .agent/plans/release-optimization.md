# Release wheel optimization

Status: simplifying the release path and validating the revised builds.

- SDK #94 supplies native assertion-enabled and assertion-free libraries, plus
  standard BOLT tools on Linux. Prepare the 2026.09.15 changelog; publish
  through the existing workflow dispatch after review.
- Setup #255 selects assertions and uses the matching release manifest.
- Workflows #464 runs one cibuildwheel job per platform.
- Core #2476 owns SDK/Core PGO, MLIR test training, Linux BOLT, and wheel
  checks. It is stacked on #2545, which owns QDMI and benchmark exception
  boundaries.

Linux uses the manylinux container's matched Clang toolchain, full LTO, PGO, and
BOLT. macOS uses Apple Clang, ThinLTO, and PGO. Both SDK and Core target macOS
13.3. Keep the manylinux 2.28 baseline and normal Windows compiler.

Complete local and platform validation, push signed PR updates, and report
release dependencies separately from code failures.
