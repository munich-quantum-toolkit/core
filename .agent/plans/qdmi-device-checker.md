# QDMI device checker

Status: implemented; focused validation passed. Full lint remains blocked by
tool availability and network downloads.

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

## Validation

- The supported `release-no-mlir` preset configured and built successfully with
  `ENABLE_CACHE=OFF` because the host's ccache cannot load its libfmt dependency.
- All six focused adapter/CLI CTests and all 69 QDMI driver tests passed.
- Full native installation and an installed wheel passed checks for both bundled
  catalogue choices and the external session-device fixture. The CLI tests cover
  provider failures, crashes, initialization/cleanup timeouts, and descendants.
- All three focused Python launcher tests passed. An installed-wheel SIGTERM
  smoke test verified sanitized failure and worker cleanup.
- `SKIP=ty uvx nox -s lint` passed. Full lint is waiting for its uv/ty environment;
  a separate type-check download failed with a network timeout.
- `uvx nox -s cpp-lint -- origin/v4.1` requires clang-tidy 23; the host has version
  20, so this required check could not run. No toolchain was changed.
- A final wheel rebuild after the build-tree runtime staging refinement is
  waiting for the same pinned QDMI fetch; the previous wheel and installed tests
  passed. Real Slurm validation belongs to the separate optional validator PR.
