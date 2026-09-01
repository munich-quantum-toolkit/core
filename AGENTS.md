# MQT Core Agent Guide

This file contains repository-specific instructions for coding agents working on
MQT Core. The project-wide policy for AI-assisted contributions is
[`docs/ai_usage.md`](docs/ai_usage.md); follow it in addition to this guide.

## Repository Layout

- `include/mqt-core/` contains the public C++ headers; implementations live in
  `src/`.
- `bindings/` contains the nanobind-based Python bindings, and
  `python/mqt/core/` contains the Python package and generated type stubs.
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
- Write code comments, documentation, tests, changelog entries, and public text
  for the final design. Never preserve prompts, review chronology, former names,
  or abandoned approaches unless they remain necessary user-facing context.
- Apply
  [Orwell's six rules for writing](https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/politics-and-the-english-language/)
  to every category of prose, including reasoning, descriptions, commit
  messages, documentation, docstrings, comments, test text, diagnostics, and
  handoffs:

  1. Do not use a familiar metaphor, simile, or other figure of speech.
  2. Use a short word when it has the same meaning as a long word.
  3. Remove every word that does not add meaning.
  4. Use active voice when possible.
  5. Use everyday English instead of a foreign phrase, scientific word, or
     jargon term when this does not reduce precision.
  6. Break a rule before it makes the text unclear, incorrect, or needlessly
     difficult to read.

- Apply the relevant principles of
  [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/): use
  short, direct sentences; give each sentence one main idea; use one term for
  one meaning; and use explicit nouns instead of vague pronouns. These are
  mandatory style rules, not a claim of formal ASD-STE100 compliance.
- Base terminology and phrasing on repository usage and established precedents
  in the quantum computing, LLVM, compiler, high-performance computing, and
  general computer science communities. Use the established term that most
  precisely matches the concept. If communities use different terms, explain the
  mapping once. Never invent synonyms for variety.
- Preserve the established capitalization of project and dependency names in
  prose.
- Add or update automated tests for every behavioral code change. During
  development, run the narrowest relevant test first, then the required lint
  checks before handoff.
- Add tests that protect intended behavior or reproduce a concrete regression.
  Never test provisional implementation choices that are not part of the
  supported contract.
- Place tests in the corresponding test tree, organized by the subsystem that
  owns the behavior. Prefer component unit tests for semantic contracts, and
  reserve subprocess tests for irreducible driver-level CLI behavior. Normal
  test targets and dependencies belong in the test build; avoid promoting an
  otherwise optional production tool into the default build solely for
  subprocess testing.
- Remove obsolete scaffolding and diagnostic suppressions before handoff. Keep a
  workaround or suppression only when it is still necessary, scope it as
  narrowly as possible, and document the technical reason.
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

- Configure a release build with `./.agent/run.sh cmake --preset release`.
- Build it with `./.agent/run.sh cmake --build --preset release`.
- Run all configured C++ tests with `./.agent/run.sh ctest --preset release`.
- Run a component binary directly when iterating, for example
  `./build/release/test/ir/mqt-core-ir-test` or
  `./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test`.
- Use GoogleTest filters to narrow a binary further, for example
  `./build/release/test/ir/mqt-core-ir-test --gtest_filter='StandardOperation.*'`.
- Replace `release` with `debug` for a debug build. Consult `CMakePresets.json`
  for other supported configurations.

The C++ code targets C++20 and uses GoogleTest. Use Doxygen-style documentation,
`#pragma once` in headers, and existing project abstractions. Prefer C++20
standard-library facilities over custom equivalents.

### Python and Bindings

- Install build and test dependencies with
  `./.agent/run.sh uv sync --inexact --only-group build --only-group test`.
- Install the package for fast local rebuilds with
  `./.agent/run.sh uv sync --inexact --no-dev --no-build-isolation-package mqt-core`.
- Run the Python tests with `./.agent/run.sh uv run --no-sync pytest`; pass a
  file or `-k` expression while iterating.
- Run the supported test sessions with `./.agent/run.sh uvx nox -s tests` and
  `./.agent/run.sh uvx nox -s minimums`. Python 3.14 variants are `tests-3.14`
  and `minimums-3.14`.
- For finite-shot tests, choose shot counts and tolerances with a sufficiently
  low false-failure probability; avoid placing expected values on tolerance
  boundaries.
- If a file in `bindings/` is added or changed, regenerate type stubs with
  `./.agent/run.sh uvx nox -s stubs`. Never edit generated `.pyi` files in
  `python/mqt/core/` manually.

Use Google-style Python docstrings. Prefer fixing diagnostics from `ruff` and
`ty` over suppressing them; document suppressions that are genuinely required.

### Worktree-Local Tool Caches

- Run cache-producing commands through `.agent/run.sh`. It derives the
  repository root from its own location, so it works from any directory in the
  worktree and exports worktree-local cache paths before executing the requested
  command. In addition to the download cache, this localizes `uv` tool
  environments, tool binaries, and managed Python installations. It also
  supplies a local XDG cache root and `PREK_HOME` so other cache-aware
  development tools stay within the worktree.
- The wrapper configures `uv` and `uvx` to use `.cache/uv`. The CMake presets
  configure `ccache` and `sccache` to use `.cache/ccache` and `.cache/sccache`,
  respectively. These paths are ignored by Git. Outside agent-driven work,
  contributors remain free to use their preferred cache configuration.
- Do not redirect these tools to a user-level or shared cache outside the
  worktree. In particular, do not work around sandbox failures by requesting
  access to a cache under a home directory.
- Use the CMake presets for configuration, builds, and tests so the compiler
  cache environment is applied consistently. The compiler caches are capped at 4
  GiB per worktree and clean up automatically as they reach that limit.
- If invoking `ccache` or `sccache` outside a CMake preset, set `CCACHE_DIR` or
  `SCCACHE_DIR` to the corresponding repository-local path first.
- After a significant batch of work, and only once no `uv`, build, or compiler
  cache process is running, run `./.agent/clean-caches.sh`. This clears only the
  current worktree's local caches. Do not remove cache contents manually while
  another process may be using them.

### Documentation

- Build the complete documentation with
  `./.agent/run.sh uvx nox --non-interactive -s docs`.
- Check documentation links with
  `./.agent/run.sh uvx nox -s docs -- -b linkcheck`.

## Generated Files and Validation

- Do not hand-edit generated stubs, rendered documentation, CMake-generated
  files, or template-managed files.
- Run `./.agent/run.sh uvx nox -s lint` after each completed batch of changes.
  It runs the full `prek` hook set, including formatting, spelling, type, and
  metadata checks.
- Inspect the final diff and working-tree status. Report every check run and
  clearly distinguish passes, failures, and checks that could not be run.

## ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as
described in [`.agent/PLANS.md`](.agent/PLANS.md)) from design to
implementation. Keep one ExecPlan per independently implemented task and store
it under `.agent/plans/<task-slug>.md`; the plan is a living record of that
task's decisions and progress.

## Spec Audits

When a subsystem accumulates tests that pin implementation choices instead of
the supported contract, audit it with a SpecAudit (as described in
[`.agent/AUDITS.md`](.agent/AUDITS.md)). Keep one audit per audited scope under
`.agent/audits/<scope-slug>.md`. An audit produces ranked verdicts with executed
evidence and stops there; a human decides which verdicts to apply, and each one
lands as its own pull request.

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
- Review findings should focus on substantive correctness, contracts,
  maintainability, tests, documentation, licensing, and validation rather than
  optional process metadata.

## Handoff Checklist

- The diff is focused and follows neighboring code conventions.
- Behavioral changes have automated test coverage, and targeted tests pass.
- `./.agent/run.sh uvx nox -s lint` passes.
- Binding changes have regenerated stubs.
- User-facing changes update `CHANGELOG.md` and `UPGRADING.md` when appropriate.
- Generated, template-managed, secret, and unrelated files are absent from the
  diff.
- AI assistance and validation results are reported transparently.
