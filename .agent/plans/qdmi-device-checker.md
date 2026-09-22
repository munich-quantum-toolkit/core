# QDMI device checker

Status: implemented and validated locally.

## Goal and scope

Add `mqt-core-qdmi-check --device ID --timeout SECONDS` for a bounded check of
one registered device. The timeout defaults to 30 seconds. Exit codes are 0 for
an available device, 1 for a failed check, 2 for invalid arguments, and 124 for
a timeout. The checker does not submit jobs or depend on Slurm.

## Decisions

- Reuse the Slurm adapter's open-and-status behavior through a private helper;
  keep its license parser, exceptions, and public API unchanged.
- Run all provider code in a supervised child. Suppress child output and report
  fixed diagnostics so provider messages cannot expose credentials.
- Use the native executable from Python's console entry point, following the
  benchmark launcher. Keep QDMI and toolchain versions unchanged.
- Limit the executable to UNIX hosts. Keep the existing Windows runtime intact.
- Install the static executable beside the existing native runtime catalogues,
  with a small `bin` launcher. Shared builds and wheels install it in `bin`.
  This preserves driver discovery without duplicating provider installations.
- Retain the worker PID until process-group cleanup finishes. On Linux, kill the
  worker if its supervising checker dies, including before supervision setup.

## Validation before the main update

- The supported `release-no-mlir` preset configured and built successfully.
- All six focused adapter/CLI CTests passed on macOS and Linux. The existing
  driver suite passed 69 tests without MLIR and 98 in the full Linux lint build.
- Full native installation and an installed wheel passed checks for both bundled
  catalogue choices and the external session-device fixture. The CLI tests cover
  provider failures, crashes, initialization/cleanup timeouts, and descendants.
- All three focused Python launcher tests passed. An installed-wheel SIGTERM
  smoke test verified sanitized failure and worker cleanup.
- Full `uvx nox -s lint`, including type checking, and the complete Linux
  `uvx nox -s cpp-lint` session with clang-tidy 23 passed.
- The full strict documentation build passed with MLIR 23.1.0.
- The installed Linux checker passed forced-parent-death and worker-cleanup
  tests. Real Slurm validation belongs to the separate optional validator PR.

## Validation after the main update

The existing macOS `release-no-mlir` build passed all seven focused adapter/CLI
CTests and all 77 driver tests. Full `uvx nox -s lint` also passed.
