# Backport the gate-based QDMI and selected library improvements to v3.x

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core 3.9.0 should make gate-based QDMI devices usable through PennyLane
without adopting the v4 compiler architecture. After this work, a Python 3.11 or
newer installation can install the optional `pennylane` extra, construct the
built-in DD-based simulator as `qp.device("mqt.ddsim.default", ...)`, and run
finite-shot PennyLane programs through the same QDMI path used by external
devices. The release also receives a small set of independent fixes and bindings
that are valuable on the maintained v3 line and do not require LLVM/MLIR 22.

The observable end-to-end result is the executable PennyLane MaxCut notebook and
the DDSIM tests under `test/python/plugins/qdmi_pennylane/`. Python 3.10
continues to install and test MQT Core without PennyLane.

## Progress

- [x] (2026-08-09 08:00Z) Refreshed `main` and `v3.x`, inspected the 3.8.0
  backport precedent, and inventoried changes merged after 3.8.0.
- [x] (2026-08-09 08:20Z) Selected the v3-compatible functionality groups and
  recorded explicit exclusions.
- [x] (2026-08-09 09:10Z) Backported typed QDMI device configuration and stable
  device-ID enumeration.
- [x] (2026-08-09 09:35Z) Backported the independent DD serialization bindings
      and ZX multi-controlled-X complexity improvement.
- [x] (2026-08-09 10:05Z) Ported the PennyLane QDMI device, tests, packaging,
  and executable notebook.
- [x] (2026-08-09 10:50Z) Regenerated stubs and `uv.lock`; completed the native
      QDMI, DD, and ZX tests, the Python 3.10, minimums 3.12, and 3.14
      boundaries, and both forced and cached documentation builds.
- [ ] Publish the signed, functionality-scoped commit series as a pull request
  targeting `v3.x` and monitor its checks.

## Surprises & Discoveries

- Observation: The v3.8.0 branch already contains the configurable QDMI driver,
  binary-safe FoMaC transport, stable device registration, and composable
  bundled-device build needed by the PennyLane adapter. Evidence: the merge of
  the 3.8.0 backport is an ancestor of the current `v3.x` tip.
- Observation: Changes merged after 3.8.0 split cleanly into v3-compatible
  library work and compiler work that requires the v4 LLVM/MLIR 22 design.
  Evidence: the selected changes affect the existing QDMI driver, DD, ZX, QIR,
  and Python plugin surfaces, while the excluded compiler-target and typed
  OpenQASM frontend changes depend on new v4 MLIR dialect contracts.
- Observation: The QIR classical-result ordering fix from #1979 modifies the
  `QCToQIR` conversion, which does not exist on v3.x. Evidence: applying the
  upstream change produces modify/delete conflicts for every implementation and
  test file because those paths were introduced only with the v4 compiler
  collection.

## Decision Log

- Decision: Backport the typed QDMI configuration transport before the PennyLane
  device. Rationale: the public PennyLane constructor maps device descriptions
  into this typed transport, and carrying that contract avoids a v3-only adapter
  API. Date/Author: 2026-08-09 / Codex.
- Decision: Backport registered device-ID enumeration before the PennyLane
  device. Rationale: PennyLane entry-point discovery and stable QDMI device IDs
  should use the same load-free enumeration API as `main`. Date/Author:
  2026-08-09 / Codex.
- Decision: Include the DD serialization API and ZX multi-controlled-X
  complexity improvement as separate commits. Rationale: each is an
  independently tested v3-compatible API or complexity improvement suitable for
  a minor release. Date/Author: 2026-08-09 / Codex.
- Decision: Exclude the LLVM/MLIR 22 compiler-target series, the typed OpenQASM
  frontend and emitter, QCO mapping fixes, specialized neutral-atom and
  superconducting runtime configuration, and routine dependency churn.
  Rationale: those changes either rely on the v4 compiler architecture, are
  outside this gate-based integration, or do not justify expanding the backport.
  Date/Author: 2026-08-09 / Codex.
- Decision: Exclude #1979 after testing patch applicability. Rationale: v3.x has
  no `QCToQIR` conversion to fix, and importing that subsystem would violate the
  v3 compiler boundary. Date/Author: 2026-08-09 / Codex.
- Decision: Keep Python 3.10 as a base-package boundary and enable PennyLane on
  Python 3.11 through 3.14. Rationale: this matches the upstream plugin and
  prevents the optional dependency from constraining existing v3 deployments.
  Date/Author: 2026-08-09 / Codex.

## Outcomes & Retrospective

The backport is split into independently reviewable commits for the two QDMI
prerequisites, DD serialization, the ZX construction, and the PennyLane device.
The QDMI driver suites passed 127 tests in total; the focused DD and ZX suites
passed 7 tests; the Python DD bindings passed 16 tests; and the PennyLane suite
passed 33 tests on Python 3.14, 10 tests with minimum dependencies on Python
3.12, and its optional-dependency boundary on Python 3.10. Stub generation and
both documentation builds completed successfully. The v3 documentation build
retains its existing C++ and optional-MLIR warning baseline; the PennyLane
notebook executes successfully. Publication and remote CI remain in progress.

## Context and Orientation

`v3.x` is the maintained MQT Core 3 release line. It retains optional LLVM/MLIR
21 support, unlike `main`, which contains the v4 compiler collection and
requires LLVM/MLIR 22. A backport therefore ports user-visible behavior against
existing v3 interfaces rather than merging the complete `main` tree.

The QDMI driver lives under `include/mqt-core/qdmi/driver/` and
`src/qdmi/driver/`. FoMaC exposes it to Python through
`bindings/fomac/fomac.cpp` and `python/mqt/core/fomac.pyi`. The optional
PennyLane integration belongs under `python/mqt/core/plugins/pennylane/` and
uses the stable QDMI device registry to open a device, preprocess a PennyLane
program, serialize it as a device-advertised OpenQASM format, submit finite-shot
jobs, and reconstruct PennyLane measurement results.

The built-in DD-based simulator is the credential-free integration oracle. Its
QDMI implementation is under `src/qdmi/devices/dd/`, while PennyLane tests are
under `test/python/plugins/qdmi_pennylane/`. The executable MyST notebook
`docs/qdmi/pennylane_device.md` documents the same local path and is generated
only through the repository's documentation Nox session.

## Plan of Work

Apply each selected upstream pull request as a separate functionality-scoped
commit. Resolve differences against v3 contracts instead of importing v4-only
surrounding code. First add typed QDMI configuration transport and stable-ID
enumeration, regenerate FoMaC stubs, and run the driver and focused Python
tests. Next port the three independent library changes and run their native or
Python regression suites. Finally add the PennyLane package, optional
dependency, tests, and notebook, preserving the `import pennylane as qp`
convention and the Python 3.10 skip boundary.

Do not copy the `main` lockfile. Regenerate `uv.lock` from the final v3
`pyproject.toml` so it continues to represent the optional LLVM/MLIR 21 build
and v3 dependency graph. Preserve the `[Unreleased]` release notes above the
existing 3.8.0 sections and define links for the upstream pull requests and the
new backport pull request.

## Concrete Steps

From the repository root, apply and validate the QDMI prerequisites:

    git show --first-parent <upstream-merge> --stat
    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target mqt-core-qdmi-driver-test
    ./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test
    ./.agent/run.sh uv run --no-sync pytest test/python/fomac/test_fomac.py

Run the focused DD and ZX validation after their respective commits:

    ./.agent/run.sh cmake --build --preset release --target mqt-core-dd-test mqt-core-zx-test
    ./build/release/test/dd/mqt-core-dd-test
    ./build/release/test/zx/mqt-core-zx-test
    ./.agent/run.sh uv run --no-sync pytest test/python/dd/test_matrix_dds.py test/python/dd/test_vector_dds.py

After the PennyLane port, regenerate derived artifacts and exercise every
supported Python boundary:

    ./.agent/run.sh uv lock
    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uvx nox -s tests-3.10
    ./.agent/run.sh uvx nox -s tests-3.14
    ./.agent/run.sh uvx nox -s minimums-3.12 -- test/python/plugins/qdmi_pennylane/test_ddsim.py
    ./.agent/run.sh uvx nox --non-interactive -s docs -- -D nb_execution_mode=force
    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check origin/v3.x...HEAD

Python 3.11 through 3.13 focused sessions should also run when those local
interpreters are available. The full release build and available C++ tests
provide the non-Python integration boundary.

## Validation and Acceptance

Acceptance requires QDMI driver tests to prove that typed configuration and
registered-ID enumeration preserve ordering and do not eagerly load devices. DD
serialization must round-trip terminal and non-terminal vector and matrix DDs in
text and binary forms. The ZX tests must prove unitary equivalence,
dirty-workspace restoration, and a quadratic resource bound.

On Python 3.11 or newer, `qp.device("mqt.ddsim.default", wires=2)` must execute
finite-shot Bell-state and QAOA programs, including samples, counts,
probabilities, expectations, variances, Hamiltonians, shot vectors, sequential
batches, and parameter-shift gradients. OpenQASM 3 must be preferred whenever
advertised; OpenQASM 2 is only a format-negotiation fallback. Python 3.10 must
import and test the base package while skipping PennyLane modules cleanly.

The documentation is accepted only when generated through `nox -s docs`, with
the first run forcing notebook execution. The complete lint session, lockfile
check, commit-signature audit, and `git diff --check` must pass before
publication.

## Idempotence and Recovery

All validation commands are repeatable. Builds, Nox environments, and caches
remain worktree-local. If an upstream patch conflicts, abort only that
in-progress application or resolve it against the current v3 file; never reset
the worktree or discard unrelated changes. A failed lockfile update can be rerun
after correcting `pyproject.toml` without changing committed source.

## Artifacts and Notes

The authoritative upstream functionality is represented by MQT Core pull
requests #1967, #1972, #1983, #1984, and #2005. The 3.8.0 precedent is pull
request #1966. Their changes are ported independently so reviewers can inspect
or revert each capability without coupling it to the rest of the release series.

## Interfaces and Dependencies

The final Python API must provide `mqt.core.plugins.pennylane.QDMIDevice`, its
converted-program record and focused exceptions, and the `mqt.ddsim.default`
PennyLane entry point. `QDMIDevice` must consume stable `device_id`, finite
`shots`, and optional FoMaC `session_parameters` and `job_parameters` mappings.

The `pennylane` optional extra must require a compatible PennyLane release only
on Python 3.11 or newer. The base MQT Core installation and Python 3.10 test
environment must not resolve or import PennyLane. No direct Amazon Braket SDK or
device-specific dependency belongs in MQT Core.

Revision note: Created the plan after comparing the live `main` and `v3.x` heads
and the exact upstream pull requests. The initial scope deliberately separates
v3-compatible library work from the v4-only compiler series. Updated the scope
after proving that #1979's `QCToQIR` implementation does not exist on v3.x.

Publication note: The selected changes were published as pull request #2024.
Local validation passed for the QDMI driver, DD and ZX suites, Python 3.10 and
3.14 boundaries, the Python 3.12 minimums environment, generated stubs, forced
and cached documentation builds, and the complete lint session. The first CI
revision passed the complete test matrix and documentation build. Its C++ lint
job identified four Clang 22 missing-field-initializer diagnostics in newly
backported QDMI test aggregates; the remediation explicitly initializes each
device definition and constructs partial session configurations through normal
assignment before rerunning the driver and lint suites.
