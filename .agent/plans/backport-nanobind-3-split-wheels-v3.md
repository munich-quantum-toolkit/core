# Backport nanobind 3 split-mode wheels to v3.x

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core v3.x currently uses nanobind 2 and publishes wheels tied to several
CPython versions. This backport adopts the reviewed nanobind 3 split-mode design
from pull request #2209 without restoring the MLIR and QIR stacks that v3.x has
removed. After the change, one `cp311-abi3` wheel supports GIL-enabled CPython
3.11 and newer, and one `cp315-abi3t` wheel supports free-threaded CPython 3.15
and newer. Installing either wheel also installs `nanobind-backend`, which owns
the interpreter-specific runtime.

The change also protects process-wide DD, IR, and QDMI state from concurrent
access on free-threaded Python. The focused ownership and concurrency tests
demonstrate these changes. A release build, the C++ and Python test suites, stub
generation, and repository lint provide the final acceptance evidence.

## Progress

- [x] (2026-09-01 19:10Z) Read `AGENTS.md`, `docs/ai_usage.md`, and
  `.agent/PLANS.md` and inspected the clean v3.x tip.
- [x] (2026-09-01 19:12Z) Inspected pull request #2209 and identified squash
      commit `3c699b36548cd0fa9e17bfa59c63dd8cd57170be` as the reviewed source
      of truth.
- [x] (2026-09-01 19:13Z) Created a separate worktree from the refreshed
  `origin/v3.x` tip and applied the squash commit without committing it.
- [x] (2026-09-01 19:20Z) Resolved the source, test, configuration,
      release-note, and delete/modify conflicts against v3.x.
- [x] (2026-09-01 19:27Z) Regenerated `uv.lock` from the resolved v3.x
      dependency graph, selected nanobind 3.0.1 because 3.0.0 is yanked, and
      held the new CMake dependency to the 4.4.2 release used by #2209.
- [x] (2026-09-01 20:05Z) Built the package and passed 18 focused DD ownership
      tests, the full Python suite with 575 passes and three documented skips,
      and an installed-extension import check.
- [x] (2026-09-01 20:31Z) Configured and built the release preset, passed the
      focused IR and QDMI concurrency tests, and passed all 1,478 configured C++
      tests with two contract-defined skips.
- [x] (2026-09-01 20:43Z) Built a `cp311-abi3` wheel, validated its contents and
      Stable ABI, and imported every extension from a clean Python 3.14
      environment with `nanobind-backend` installed as a dependency.
- [x] (2026-09-01 20:50Z) Regenerated the nanobind 3 stubs, passed repository
      lint and the lock check, and passed the staged whitespace check. The two
      generated stub changes match the follow-up on main.
- [x] (2026-09-01 20:58Z) Reproduced the existing v3 C++ CI job with cpp-linter
      1.13.0 and a preinstalled clang-tidy 22 binary. The first run found two
      unused includes and two incomplete designated initializers; the final run
      reported zero findings.
- [x] (2026-09-01 21:00Z) Passed the Python 3.11 minimum-dependency session with
      578 tests. Attempted the complete documentation build; Sphinx reached
      notebook execution but the host lacks the Graphviz `dot` executable.
- [x] (2026-09-01 21:02Z) Inspected the complete staged diff with independent
      C++ and packaging reviews. Both reviews found no remaining substantive
      defects, and no unrelated or generated build files are staged.

## Surprises & Discoveries

- Observation: Pull request #2209 contains 23 branch commits, including
  provisional fixes and three merges from `main`, but GitHub merged it as one
  squash commit. Evidence: commit `3c699b365` has parent `f672f53fc`, and its
  tree matches the final pull request head. The squash diff is safer than
  replaying the branch history.
- Observation: v3.x removed MLIR and QIR after version 3.9.2. Evidence: the
  target tip deletes `mlir/`, `bindings/mlir/`, `src/qir/`, the MLIR stub, and
  related scripts. The source pull request still touched MLIR packaging and
  release-wheel setup.
- Observation: v3.x builds `mqt.core.na` where the source pull request's reduced
  wheel test imports `mqt.core.mlir`. The reduced v3.x test must import
  `mqt.core.dd`, `mqt.core.ir`, `mqt.core.na`, and `mqt.core.qdmi`.
- Observation: v3.x retains the `fomac::Session` compatibility API around QDMI.
  The source concurrency test used the later `qdmi::Session` name and would not
  compile unchanged.
- Observation: Python 3.11 already selects Sphinx 8.2.3 on v3.x. The Sphinx 9
  floor in the source pull request comes from the main dependency graph, not
  from nanobind split mode.
- Observation: v3.x predates the symbol-localization code that pull request
  #2209 extends. Applying only textual conflict hunks would omit required macOS
  exception RTTI exports and ELF symbol and section handling.
- Observation: nanobind 3.0.0 was yanked after pull request #2209 merged because
  its metadata mistakenly declared Python 3.9 compatibility. Evidence: uv now
  rejects 3.0.0 for the compatible `~=3.0.0` requirement and identifies 3.0.1 as
  the next valid release.
- Observation: `docs/tooling.md` still describes the former Python 3.12 stable
  ABI and per-version free-threaded wheels. Its header says that the file comes
  from the MQT templates repository and must not be edited directly. The
  backport therefore documents the new wheel policy in `CHANGELOG.md` and
  `UPGRADING.md`; the template text needs a separate upstream change.
- Observation: the current repository guide requests `uvx nox -s cpp-lint`, but
  the v3.x `noxfile.py` does not define that session. A preinstalled clang-tidy
  22 binary was available outside `PATH`, so the existing v3 reusable CI job
  could be reproduced directly with cpp-linter 1.13.0. This validation does not
  add LLVM or MLIR to the project. The first run found four diagnostics that the
  source pull request's older lint run did not report.
- Observation: the documentation build reaches notebook execution but this host
  has no Graphviz `dot` executable. The failure is environmental and occurs in
  unchanged `docs/dd_package.md`; lint and all code tests still pass.

## Decision Log

- Decision: Use squash commit `3c699b365` as the backport source of truth.
  Rationale: the squash contains the reviewed final state, while the branch
  history contains provisional implementations and merges unrelated to v3.x.
  Date/Author: 2026-09-01 / Codex.
- Decision: Keep MLIR and QIR absent and omit LLVM setup from the release-wheel
  workflow. Rationale: restoring removed stacks would undo v3.x pull request
  #2314 and is not needed for nanobind 3. Date/Author: 2026-09-01 / Codex.
- Decision: Port the complete final `cmake/AddMQTPythonBinding.cmake` behavior.
  Rationale: split mode requires module-name tracking, macOS initializer and
  exception RTTI exports, ELF symbol hiding and section collection, Windows
  export control, and Windows `abi3t` detection. Date/Author: 2026-09-01 /
  Codex.
- Decision: Preserve v3.x interfaces and dependency floors outside the
  migration. Rationale: the backport must retain the FoMaC compatibility API,
  optional-shot behavior, neutral-atom module, `eval` lint rule, Breathe docs
  dependency, and effective Sphinx 8.2.3 floor. Date/Author: 2026-09-01 / Codex.
- Decision: Keep pull request #2209's Python 3.11 floor and remove Python 3.10
  compatibility code. Rationale: the reduced split-wheel matrix depends on this
  explicit part of the requested backport. Date/Author: 2026-09-01 / Codex.
- Decision: Regenerate the lockfile instead of taking the source lockfile.
  Rationale: `uv.lock` must represent the v3.x dependency graph, which still
  differs from main. Date/Author: 2026-09-01 / Codex.
- Decision: Use nanobind 3.0.1 across build metadata, CI, and the lockfile.
  Rationale: 3.0.0 is yanked and no longer resolves through the pull request's
  compatible-release constraint. MQT Core already adopted 3.0.1 as the direct
  patch update after #2209. Date/Author: 2026-09-01 / Codex.
- Decision: Do not edit the stale stable-ABI description in `docs/tooling.md`.
  Rationale: its generated-file header and repository policy require changes in
  the MQT templates repository. The branch-local upgrade guide provides the
  release-specific migration instructions. Date/Author: 2026-09-01 / Codex.

## Outcomes & Retrospective

The backport is complete. The resolved diff preserves v3.x-only interfaces and
keeps removed production stacks absent. The release build, full C++ and Python
suites, Python 3.11 minimum-dependency suite, focused ownership and concurrency
tests, generated stubs, wheel build, wheel-content check, Stable ABI audit,
clean wheel import test, repository lint, direct v3 C++ CI lint, lock check, and
staged diff checks pass. The only incomplete validation is the documentation
build: Sphinx cannot execute an unchanged DD notebook because Graphviz `dot` is
not installed on the host.

## Context and Orientation

`pyproject.toml` owns the Python floor, build requirements, scikit-build-core
wheel ABI, cibuildwheel matrix, runtime dependencies, and development dependency
groups. `uv.lock` is generated from that file. The classic wheel requests the
CPython 3.11 stable ABI through `wheel.py-api = "cp311"`. An override requests
`cp315t` when the interpreter is free-threaded CPython 3.15 or newer.

Split mode keeps a small binding frontend in each MQT Core extension and loads
the matching runtime from the `nanobind-backend` Python package. All v3.x
extensions pass through `cmake/AddMQTPythonBinding.cmake`; therefore the shared
helper selects split mode once for `mqt.core.dd`, `mqt.core.ir`, `mqt.core.na`,
and `mqt.core.qdmi`. The ordinary C++ project keeps its CMake 3.24 floor. Python
package builds require CMake 4.4.1 so Windows free-threaded detection is
correct.

`bindings/dd/` exports decision-diagram vectors and matrices to NumPy.
`include/mqt-core/ir/Register.hpp`,
`include/mqt-core/ir/operations/Expression.hpp`, `src/dd/Edge.cpp`, and
`src/ir/operations/Expression.cpp` contain process-wide scratch or registry
state reachable from independent Python objects. These paths use thread-local,
atomic, or mutex-protected state after the backport.

`include/mqt-core/qdmi/driver/Driver.hpp` and `src/qdmi/driver/Driver.cpp` own
the process-wide device catalog, sessions, and per-device jobs. The backport
keeps provider calls and object destruction outside collection locks. Tests in
`test/qdmi/driver/` exercise concurrent registration, persistent opens, and job
destruction. The v3.x public compatibility surface still calls device sessions
through `fomac::Session`.

The target branch removed its MLIR and QIR stacks. The backport must not restore
`bindings/patterns.txt`, `python/mqt/core/mlir.pyi`, or
`scripts/qiskit_c_api_adopt.py`, and `.github/workflows/cd.yml` must not gain
LLVM setup. Generated stubs in `python/mqt/core/` must be changed only by the
stub-generation session.

## Plan of Work

Apply the final squash diff without committing it. Resolve conflicts by
translating the final behavior to the target branch instead of choosing whole
source or target files. Preserve v3.x code that landed after the pull request's
base, including optional QDMI shot counts and removal of MLIR and QIR.

In `pyproject.toml`, require Python 3.11, nanobind 3.0.1, and
`nanobind-backend>=1.0`. Select `cp311` for classic stable-ABI wheels and
`cp315t` for free-threaded Python 3.15. Skip 3.14t and use dependency-free
import tests for Windows ARM64 and Python 3.15 artifacts. Retain v3.x
dependencies and replace only Python-version conditions made obsolete by the new
floor.

Port the complete binding helper needed by nanobind split mode. Apply the
nanobind 3 API updates, direct DD vector ownership, exception-safe capsules, and
shared-state synchronization. Add the ownership and concurrency tests from the
source pull request, translated to v3.x names where required.

Update `CHANGELOG.md` and `UPGRADING.md` under the unreleased v3.x section.
Record pull request #2209 and both original human contributors. Regenerate
`uv.lock`. Run focused tests first, then the complete build, test, stub, and
lint checks. Inspect generated stub changes and the final diff before handoff.

## Milestones

The first milestone establishes the correct target and produces a conflict-free
source tree. Refresh `origin/v3.x`, create a separate worktree, apply squash
commit `3c699b365` without committing it, and resolve each conflict against the
current v3.x interfaces. At the end, MLIR and QIR remain absent, all unmerged
paths are resolved, and the worktree contains only the intended backport.

The second milestone adapts packaging and runtime behavior. Update the Python
floor, split-wheel metadata, binding helper, nanobind APIs, shared-state locks,
tests, release notes, and v3.x lockfile. Generate stubs instead of editing them
by hand. At the end, a local package build imports all four v3.x extensions and
the focused DD, IR, and QDMI tests pass.

The third milestone proves the integrated backport. Configure and build the
release preset, run all C++ and Python tests, build and inspect a wheel, install
that wheel in a clean environment, and run all available lint and diff checks.
Acceptance requires a `cp311-abi3` wheel with the runtime backend dependency, no
Stable ABI findings, successful imports, passing suites with only documented
skips, and a focused final diff with no generated build output.

## Concrete Steps

Run all commands from the repository root of the separate v3.x worktree.

Regenerate and check the lockfile with:

    uv lock
    uv lock --check

Configure and build the release preset, then run the focused C++ tests:

    cmake --preset release
    cmake --build --preset release
    ./build/release/test/ir/mqt-core-ir-test --gtest_filter='SymbolicVariableTest.ConcurrentRegistration'
    ./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test --gtest_filter='DeviceRegistrationTest.Concurrent*'

Run all configured C++ tests after the focused tests pass:

    ctest --preset release

Install the package through the repository's no-build-isolation workflow and run
the focused and complete Python tests:

    uv sync --locked --only-group dev
    uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    uv run --no-sync pytest test/python/dd/test_vector_dds.py test/python/dd/test_matrix_dds.py
    uv run --no-sync pytest

Regenerate stubs and run required checks:

    uvx nox -s stubs
    uvx nox -s lint
    git diff --cached --check

The v3.x Nox file does not define `cpp-lint`. Reproduce the existing C++ linter
CI job with a preinstalled clang-tidy 22 binary and cpp-linter 1.13.0. These
commands use a local analysis tool; they do not add an LLVM or MLIR dependency
to MQT Core:

    env CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ .venv/bin/cmake -B build/cpp-lint --preset lint -DPython_EXECUTABLE=.venv/bin/python
    env CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ CCACHE_DIR=.cache/ccache CCACHE_TEMPDIR=.cache/ccache/tmp .venv/bin/cmake --build build/cpp-lint
    env GITHUB_OUTPUT=/private/tmp/mqt-core-cpp-linter-output uvx --from cpp-linter==1.13.0 cpp-linter --style= --tidy-checks= --version=/opt/homebrew/opt/llvm/bin '--ignore=build|!build/mlir/**|**/include|include|vendor/**' --thread-comments=false --step-summary=false --database=build/cpp-lint --extra-arg=-std=c++20 --files-changed-only=true --lines-changed-only=false --diff-base=origin/v3.x --file-annotations=false --jobs=0 --verbosity=info

Build and inspect the wheel, then test it in a clean environment:

    uv build --wheel --out-dir /private/tmp/mqt-core-backport-wheel
    uvx check-wheel-contents /private/tmp/mqt-core-backport-wheel/*.whl
    uvx abi3audit --strict /private/tmp/mqt-core-backport-wheel/*.whl
    uv venv --python 3.14 /private/tmp/mqt-core-backport-wheel-venv
    uv pip install --python /private/tmp/mqt-core-backport-wheel-venv/bin/python /private/tmp/mqt-core-backport-wheel/*.whl
    /private/tmp/mqt-core-backport-wheel-venv/bin/python -c 'import nanobind_backend, mqt.core.dd, mqt.core.ir, mqt.core.na, mqt.core.qdmi'

Inspect `git status --short`, `git diff --stat`, and the complete diff.
Generated build output must remain under `build/` and must not enter the change.

## Validation and Acceptance

The metadata must reject Python 3.10 and require `nanobind-backend>=1.0`. The
resolved wheel matrix must contain classic `cp311-abi3` and free-threaded
`cp315-abi3t` builds, skip 3.14t, and not restore 3.13t. Reduced wheel tests
must import `mqt.core.dd`, `mqt.core.ir`, `mqt.core.na`, and `mqt.core.qdmi`.

The focused DD tests must prove that vector and matrix arrays remain valid and
writable after their source DD objects are destroyed. The symbolic test must
complete concurrent registration without corrupting names or hashes. The QDMI
tests must show that concurrent registration inserts one definition, concurrent
persistent opens return one device, and concurrent job frees destroy every job.

The release build and all configured C++ tests must pass. The complete Python
test suite must pass with only documented skips. Stub generation must either
produce the expected nanobind 3 changes or leave the generated files unchanged.
The direct cpp-linter CI reproduction, `uvx nox -s lint`, `uv lock --check`, and
`git diff --cached --check` must pass. If an environmental limit prevents a
check, this plan and the final handoff must record the command and exact
failure.

The final diff must not contain restored MLIR or QIR files, generated build
output, credentials, unrelated dependency upgrades, or changes from another
worktree.

## Idempotence and Recovery

Lock generation, configuration, builds, tests, stub generation, and lint are
safe to repeat. Build output stays under `build/`. If configuration becomes
stale after dependency changes, rerun the release preset before building.

The source squash was applied without creating a commit. Before the final
commit, all conflict stages must be resolved and `git diff --cached --check`
must pass. Because this work lives in a separate worktree created for the task,
conflict recovery cannot overwrite unrelated user changes in another worktree.

## Artifacts and Notes

The source pull request reported 4,143 passing CTest cases with one
contract-defined skip, 727 passing Python tests with three skips, 18 DD vector
and matrix tests, 95 QDMI driver tests, and 20 symbolic tests. These results
establish the intended main behavior but do not replace v3.x validation.

The initial application conflicted in CI, release notes, the binding helper, Nox
and Python configuration, QDMI tests, and the lockfile. It also produced
delete/modify conflicts for files removed with the MLIR and QIR stacks. The
resolved backport keeps those deleted files absent and regenerates the lockfile.

## Interfaces and Dependencies

`cmake/AddMQTPythonBinding.cmake` must call `nanobind_add_module` with
`FREE_THREADED`, `NB_SUPPRESS_WARNINGS`, and `BACKEND_MODULE nanobind_backend`.
On Windows `Py_TARGET_ABI3T` must set `NB_ABI` to the interpreter major and
minor version followed by `t`. macOS must export each module initializer and, on
x86-64, nanobind's four exception RTTI symbols. ELF builds must hide
static-library symbols and collect unused sections in optimized configurations.

`pyproject.toml` must require `nanobind~=3.0.1` for builds, CMake 4.4.1 in the
Python build group, and `nanobind-backend>=1.0` at runtime. The project Python
floor is 3.11. Classic wheels use `cp311`; free-threaded Python 3.15 wheels use
`cp315t` and therefore receive the `abi3t` tag.

The QDMI driver must serialize mutations of its owned job map, registered
definitions, opened devices, client catalog, and session map. It must not hold
collection locks while loading provider libraries or destroying provider-owned
objects. The IR registry must lock both writes and reads, implicit register
counters must use relaxed atomic increments, and per-thread DD scratch state
must not be shared across independent Python calls.
