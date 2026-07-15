# MQT Core Agent Guide

This file contains repository-specific instructions for coding agents working on
MQT Core. The project-wide policy for AI-assisted contributions is
[`docs/ai_usage.md`](docs/ai_usage.md); follow it in addition to this guide.

## Repository Layout

- `include/mqt-core/` contains the public C++ headers; implementations live in
  `src/`.
- `bindings/` contains the nanobind-based Python bindings, and
  `python/mqt/core/` contains the Python package and generated type stubs.
- `mlir/` contains the MQT MLIR dialects, transformations, tools, and unit
  tests. Building it requires LLVM/MLIR 22.1 or newer.
- `test/` contains the C++ and Python tests. C++ tests generally mirror the
  corresponding component under `src/`.
- `docs/` contains the Sphinx and MyST documentation; `json/` contains schemas
  and data used by the project.
- `cmake/` and `CMakePresets.json` define the supported builds. Keep generated
  build output in `build/` and do not commit it.

## Working Principles

- Keep changes focused on the assigned task. Do not perform unrelated cleanup,
  broad reformatting, dependency upgrades, or refactors without explicit
  authorization.
- Preserve user changes and inspect the working tree before editing. Never
  discard or overwrite changes that are outside the task.
- Follow the patterns in neighboring files and prefer the smallest change that
  fully solves the problem.
- Add or update automated tests for every behavioral code change. During
  development, run the narrowest relevant test first, then the required lint
  checks before handoff.
- Update `CHANGELOG.md` and `UPGRADING.md` for user-facing, breaking, or
  otherwise noteworthy changes.
- Format changelog entries with the pull request reference and every
  contributing author, for example `([#123]) ([**@username**])`, and define the
  corresponding links at the bottom of `CHANGELOG.md`.
- Never commit credentials, tokens, private keys, personal data, or other
  secrets. Do not print secrets from the environment or GitHub Actions. Use
  documented environment variables and repository secrets instead.
- Do not edit files whose header says that they are generated from an external
  template. Propose those changes in the
  [MQT templates repository](https://github.com/munich-quantum-toolkit/templates)
  or let the templating workflow update them.

## Build and Test

### C++

- Configure a release build with `cmake --preset release`.
- Build it with `cmake --build --preset release`.
- Run all configured C++ tests with `ctest --preset release`.
- Run a component binary directly when iterating, for example
  `./build/release/test/ir/mqt-core-ir-test` or
  `./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test`.
- Use GoogleTest filters to narrow a binary further, for example
  `./build/release/test/ir/mqt-core-ir-test --gtest_filter='StandardOperation.*'`.
- Replace `release` with `debug` for a debug build. Consult `CMakePresets.json`
  for other supported configurations.

The C++ code targets C++20 and uses GoogleTest. Use Doxygen-style documentation,
`#pragma once` in headers, and existing project abstractions. Prefer C++20
standard-library facilities over custom equivalents. Within `mlir/`, prefer LLVM
types such as `llvm::SmallVector` and `llvm::function_ref` where appropriate.

### Python and Bindings

- Install build and test dependencies with
  `uv sync --inexact --only-group build --only-group test`.
- Install the package for fast local rebuilds with
  `uv sync --inexact --no-dev --no-build-isolation-package mqt-core`.
- Run the Python tests with `uv run --no-sync pytest`; pass a file or `-k`
  expression while iterating.
- Run the supported test sessions with `uvx nox -s tests` and
  `uvx nox -s minimums`. Python 3.14 variants are `tests-3.14` and
  `minimums-3.14`.
- If a file in `bindings/` is added or changed, regenerate type stubs with
  `uvx nox -s stubs`. Never edit generated `.pyi` files in `python/mqt/core/`
  manually.

Use Google-style Python docstrings. Prefer fixing diagnostics from `ruff` and
`ty` over suppressing them; document suppressions that are genuinely required.

### MLIR and Documentation

- Build the MLIR documentation with
  `cmake --build --preset release --target mlir-doc`.
- A real focused MLIR test binary is
  `./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`.
- Build the complete documentation with `uvx nox --non-interactive -s docs`.
- Check documentation links with `uvx nox -s docs -- -b linkcheck`.

## Generated Files and Validation

- Do not hand-edit generated stubs, rendered documentation, CMake-generated
  files, or template-managed files.
- Run `uvx nox -s lint` after each completed batch of changes. It runs the full
  `prek` hook set, including formatting, spelling, type, and metadata checks.
- Inspect the final diff and working-tree status. Report every check run and
  clearly distinguish passes, failures, and checks that could not be run.

## Git and GitHub Actions

- A coding agent may perform coding, Git, and GitHub workflow tasks that a human
  has explicitly delegated. Authorization is limited to that stated scope;
  request fresh authorization before taking an external action outside it.
- Scoped authorization to create or update public GitHub text permits posting
  within that scope without separate approval for each message. A human remains
  accountable and must review agent-assisted work before it is accepted or
  merged.
- Every public text body authored or edited by an agent—including issue and
  pull-request descriptions, comments, and reviews—must visibly include the
  exact disclosure `🤖 *AI text below* 🤖`. Titles are exempt.
- Never use an agent to work on an issue labeled `good first issue`, and never
  generate spam, repetitive reviews, or unreviewed contributions.
- Do not push, open or merge a pull request, post on GitHub, or otherwise change
  remote state unless the human has explicitly authorized that action.
- Every commit prepared with AI assistance must include the trailer
  `Assisted-by: [Model Name] via [Tool Name]`, using the actual model and tool.

## Handoff Checklist

- The diff is focused and follows neighboring code conventions.
- Behavioral changes have automated test coverage, and targeted tests pass.
- `uvx nox -s lint` passes.
- Binding changes have regenerated stubs.
- User-facing changes update `CHANGELOG.md` and `UPGRADING.md` when appropriate.
- Generated, template-managed, secret, and unrelated files are absent from the
  diff.
- AI assistance and validation results are reported transparently.
