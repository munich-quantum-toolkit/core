# Backport compatible QDMI, device, and maintenance changes to v3.x

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core 3 already exposes the QDMI Python namespace through the first namespace
backport. This combined backport completes the compatible QDMI work that has
since merged into `main`, restores the previously omitted runtime device
configuration, and carries compatible test, documentation, packaging,
dependency, and CI maintenance in one pull request.

Users can inspect device queues, retrieve provider jobs, query custom operation
lists, configure the bundled neutral-atom and superconducting devices at
runtime, select reusable IQM models, and select a registered device from a Slurm
license environment. Administrators also get a two-node Slurm fixture and a
concise tutorial that demonstrate cluster-wide admission without presenting
license metadata as authorization. The documentation build emits `llms.txt` and
`llms-full.txt` through Sphinx-LLM.

## Progress

- [x] (2026-08-14) Compared each first-parent `main` change merged after pull
  request #2024 with the current `v3.x` architecture.
- [x] (2026-08-14) Ported queue telemetry, job retrieval, custom operation
  lists, the Slurm selector, and the Slurm fixture.
- [x] (2026-08-14) Ported runtime-configurable neutral-atom and superconducting
  devices and the reusable IQM Garnet and Emerald models.
- [x] (2026-08-14) Ported compatible Python, packaging, dependency, CI, test,
  documentation, and cleanup changes, including Sphinx-LLM.
- [x] (2026-08-14) Added the optional-component wheel target needed by the
  faster stub build and regenerated the recursive stubs without a tracked
  result.
- [x] (2026-08-14) Built and tested the complete non-MLIR native and Python
  surfaces and built the documentation with Sphinx-LLM.
- [x] (2026-08-14) Completed the final lint pass and adversarial aggregate
  review. Corrected the v3 workflow target, documentation roles, QDMI
  terminology, optional-component wheel target, and unity-build collisions.
- [ ] Publish one combined pull request against `v3.x` and verify its exact
  revision, metadata, signatures, assignment, labels, and replacement CI.

## Surprises & Discoveries

- Observation: Pull request #2113 already contains the compatibility surface
  required by the later QDMI and Slurm layers. Evidence: `v3.x` exports
  `mqt.core.qdmi`, `mqt.core.qdmi.driver`, and the deprecated v3 aliases.
- Observation: The QDMI features form a dependency chain even though they merged
  as separate pull requests on `main`. Queue telemetry updates the QDMI revision
  and low-level driver contract; retrieval and custom operation lists build on
  that contract; the Slurm adapter builds on the namespace.
- Observation: The v3 Python package still listed each native target directly.
  Disabling the NA and SC providers for stub generation therefore requested
  targets that CMake intentionally did not create. Evidence: the first stub
  build failed on `mqt-core-qdmi-na-device`. The conditional `mqt-core-wheel`
  target from the namespace work fixes this without changing wheel contents.
- Observation: Enabling unity builds exposed the same DD-binding and
  neutral-atom anonymous-namespace collisions fixed on `main`. The applicable
  source fixes from pull request #2047 are required even though its newer MLIR
  documentation-build path is not present on v3.
- Observation: Sphinx-LLM entered `main` as part of pull request #1989, while
  pull request #2046 only added the minimum version. Both pieces are required
  for a functional backport.

## Decision Log

- Decision: Use one pull request with separate signed commits for coherent
  upstream functions. Rationale: this saves stable-branch CI capacity while
  preserving provenance, review boundaries, and possible future reverts.
  Date/Author: 2026-08-14 / Codex.
- Decision: Include non-breaking maintenance when it applies to v3. This covers
  CPython 3.15 preparation, standard dynamic package metadata, nanobind 2.14,
  faster stubs, Sphinx-LLM, unity builds, the reduced CI matrix, hook and action
  updates, stronger ZX tests, optional-dependency cleanup, and dead private
  typing-code removal. Date/Author: 2026-08-14 / Codex.
- Decision: Exclude the compiler-target, QCO/QTensor, newer QIR-runtime, and
  OpenQASM semantic series. Their source paths do not exist on v3 or depend on
  the LLVM/MLIR 22 architecture. Also exclude the explicitly breaking ZX and
  CMake-helper removals. Date/Author: 2026-08-14 / Codex.
- Decision: Treat lock-only maintenance as superseded by the final lock
  generated from this aggregate dependency set. Rationale: replaying an older
  lock snapshot would replace newer, intentionally resolved metadata.
  Date/Author: 2026-08-14 / Codex.
- Decision: Keep one local Slurm license with an optional count of one as the
  selector grammar. Rationale: this is the reviewed `main` contract; Slurm
  licenses are admission controls and mutable environment metadata, not access
  credentials or allocation attestation. Date/Author: 2026-08-14 / Codex.

## Outcomes & Retrospective

The aggregate non-MLIR release build succeeds with LLVM 21.1.8 `llc`. All 1,526
CTest cases pass; two provider-specification job-ID cases are intentionally
skipped. The complete Python suite passes 531 tests with five documented skips.
Recursive stub generation completes without a tracked stub change.

The Sphinx build succeeds with the existing v3 warning baseline and produces
non-empty `llms.txt` and `llms-full.txt` outputs. The privileged Slurm fixture
remains a GitHub Actions validation because it requires Docker, systemd, cgroup
v2, and two Slurm compute containers.

Final lint and the adversarial review pass. Publication and exact-head CI are
pending.

## Context and Orientation

QDMI is the low-level device interface. Device providers implement its device
side. `src/qdmi/driver` loads registered provider libraries and implements the
QDMI client interface. `src/fomac` and `include/mqt-core/fomac` provide owning
C++ wrappers, while `bindings/qdmi` exposes the Python namespace. The current v3
Python API uses `mqt.core.qdmi` for device objects and `mqt.core.qdmi.driver`
for registration and opening.

Queue telemetry consists of an optional device queue length and an optional job
queue position. Job retrieval opens an existing provider job by its stable job
identifier. A custom operation list is a provider-defined QDMI property whose
payload is an array of operation handles. FoMaC converts those handles to the
same owning `Operation` objects used by the standard operation property.

The NA and SC providers load strict JSON models into session-owned state. Their
configuration can come from persistent device definitions, explicit session
parameters, environment-selected files, or bundled defaults. The IQM Garnet and
Emerald files are reusable SC models registered under stable IDs.

The Slurm adapter reads `SLURM_JOB_LICENSES`, accepts one registered local
device ID with an omitted count or count one, opens a fresh device session, and
accepts device states `IDLE` and `BUSY`. The environment variable is mutable.
The adapter therefore selects a device but does not authenticate the user or
prove an allocation. Provider credentials or operating-system isolation remain
the authorization boundary.

## Plan of Work

Adapt the implementation and tests from pull request #2010 to the v3 driver and
the QDMI Python namespace. Update the bundled QDMI revision to the version that
defines queue and retrieval interfaces. Preserve optional-provider loading so
old provider libraries remain usable and report unsupported retrieval rather
than failing to load.

Add the high-level retrieval and custom-operation APIs from pull requests #2008
and #2042. Keep returned devices, jobs, sites, and operations owning their
provider sessions. Return `None` or `std::nullopt` for unsupported optional
properties, reject malformed byte counts, and preserve provider errors.

Adapt pull requests #2025 and #2043 to the v3 namespace. Keep the
mechanism-specific `fomac::slurm::openDeviceFromLicense()` C++ API and
`mqt.core.qdmi.slurm.open_device_from_license()` Python API. Add the Slurm 25.11
fixture, wheel-based container installation, DDSIM and SC license scenarios, and
the trust-boundary tutorial.

Port the runtime configuration from pull requests #1974, #1980, and #1992. Then
apply each compatible maintenance change independently, retain v3's optional
LLVM/MLIR 21 contract, and omit main-only architecture work. Resolve the
combined changelog semantically and do not carry obsolete intermediate plans or
superseded APIs.

## Concrete Steps

Run cache-producing commands through the worktree-local wrapper.

Regenerate recursive stubs after all binding changes:

    ./.agent/run.sh uvx nox -s stubs

Configure, build, and test the non-MLIR release tree with LLVM 21.1.8 `llc` on
`PATH`:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./.agent/run.sh ctest --preset release

Install the Python package and run the Python suite:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    ./.agent/run.sh uv run --no-sync pytest test/python -q

Run documentation and repository validation:

    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check origin/v3.x...HEAD

The privileged Slurm workflow runs in GitHub Actions.

## Validation and Acceptance

The driver tests prove absent and present queue values, a fresh status query for
each job queue-position request, supported and unsupported retrieval, correct
job ownership, custom operation handles, malformed custom payloads, and
operation-property queries. Python tests prove equivalent `None`, error, and
lifetime behavior through `mqt.core.qdmi`.

Runtime-configuration tests prove strict model parsing, source precedence,
independent and concurrent sessions, supported topology and calibration data,
and stable model registration. Packaging must build with all device targets and
with the NA and SC providers disabled for stub generation.

The selector tests accept only `<registered-device-id>` and
`<registered-device-id>:1`. They reject a missing value, whitespace, unknown
IDs, remote licenses, malformed or non-unit counts, and AND or OR expressions.
The adapter opens a fresh device in `IDLE` or `BUSY` state and rejects all other
states with the ID and status in the diagnostic.

The privileged fixture must run two held DDSIM Bell jobs on separate nodes,
leave a third pending for the DDSIM license while CPUs remain available, run an
SC job independently, release one DDSIM job, admit the pending job, and validate
256 samples containing only `00` and `11`. It also proves that the selector
rejects compound and non-unit expressions. The container imports the built wheel
without `PYTHONPATH`, and the packaged `slurmd` unit reports `Delegate=yes`.

The documentation build must load `sphinx_llm.txt` and create non-empty
`llms.txt` and `llms-full.txt` files.

## Idempotence and Recovery

Builds, stub generation, tests, and documentation commands are repeatable. A
failed source adaptation is repaired in its logical commit rather than by
discarding unrelated work. If `v3.x` advances before publication, rebase the
complete sequence, resolve changelog conflicts semantically, regenerate stubs,
and rerun affected tests. Never modify another task worktree.

## Artifacts and Notes

The combined series backports functionality from pull requests `#1974`, `#1980`,
`#1989`, `#1992`, `#2008`, `#2010`, `#2011`, `#2019`, `#2020`, `#2021`, `#2023`,
`#2025`, `#2027`, `#2029`, `#2042`, `#2043`, `#2046`, `#2047`, `#2059`, `#2065`,
`#2073`, `#2074`, `#2075`, `#2083`, and `#2108`. Pull request `#2113` is the
authoritative v3 prefix.

The lock-only pull request #2033 is represented by the final `uv lock` result.
Compiler and QIR changes are excluded because the corresponding main source tree
is not present on v3. Pull requests #2082 and #2106 are explicitly breaking and
remain main-only.

### Interfaces and Dependencies

The public additions are:

- optional C++ and Python device queue-length and job queue-position queries;
- `fomac::Device::retrieveJobById(std::string_view)` and
  `mqt.core.qdmi.Device.retrieve_job_by_id(str)`;
- `fomac::Device::queryCustomOperations(fomac::CustomProperty)` and
  `mqt.core.qdmi.Device.query_custom_operations(CustomProperty)`;
- `fomac::slurm::openDeviceFromLicense()` and
  `mqt.core.qdmi.slurm.open_device_from_license()`;
- runtime-configurable NA and SC providers and stable IQM model registrations.

The bundled QDMI revision exposes the matching queue and retrieval enums and
functions. The Slurm fixture requires Slurm 25.11 or newer, Ubuntu 26.04,
systemd, Munge, cgroup v2, and Docker Compose. Runtime device authorization is
provider-specific and remains outside the Slurm selector.

Revision note: expanded the original QDMI and Slurm plan after the user asked
for one comprehensive non-breaking backport, runtime device configuration,
compatible CI and dependency updates, and Sphinx-LLM.
