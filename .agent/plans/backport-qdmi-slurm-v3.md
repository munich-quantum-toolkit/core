# Backport the remaining QDMI and Slurm functionality to v3.x

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core 3 already exposes the QDMI Python namespace. This backport completes
the compatible QDMI feature set that merged into `main` after the previous
combined v3 backport. Users can inspect device queues, retrieve provider jobs,
query provider-specific operation lists, and select a registered device from a
Slurm license environment. Administrators also get a two-node Slurm fixture and
a concise tutorial that demonstrate cluster-wide admission without presenting
license metadata as authorization.

## Progress

- [x] (2026-08-14 18:20Z) Compared every first-parent `main` change merged
  after pull request #2024 with the current `v3.x` architecture.
- [x] (2026-08-14 18:25Z) Selected pull requests #2010, #2008, #2042, #2025,
  and #2043 as one coherent, v3-compatible functionality range.
- [ ] Port queue telemetry and job retrieval while preserving the v3 QDMI
  driver and Python namespace.
- [ ] Port generic custom-operation list queries.
- [ ] Port the Slurm license selector and its system fixture and tutorial.
- [ ] Regenerate stubs, validate the aggregate range, and complete an
  adversarial review.
- [ ] Publish one combined pull request against `v3.x` and verify its exact
  revision, metadata, signatures, and replacement CI.

## Surprises & Discoveries

- Observation: The first namespace backport, pull request #2113, already
  contains the compatibility surface required by the later Slurm layer.
  Evidence: the current `v3.x` tip exports `mqt.core.qdmi`,
  `mqt.core.qdmi.driver`, and the deprecated v3 aliases.
- Observation: The QDMI features form a dependency chain even though they
  merged as separate pull requests on `main`. Queue telemetry updates the QDMI
  revision and low-level driver contract; job retrieval and custom operation
  lists build on that contract; the Slurm adapter builds on the namespace.

## Decision Log

- Decision: Use one pull request with separate signed commits for each upstream
  functionality. Rationale: this saves stable-branch CI capacity while keeping
  provenance, review boundaries, and possible future reverts clear.
  Date/Author: 2026-08-14 / Codex.
- Decision: Exclude compiler and QIR changes, breaking removals, dependency
  churn, and CI-only optimizations merged after #2024. Rationale: those changes
  either depend on the LLVM/MLIR 22 architecture, conflict with the v3
  compatibility promise, or do not justify widening a functional backport.
  Date/Author: 2026-08-14 / Codex.
- Decision: Keep one local Slurm license with an optional count of one as the
  selector grammar. Rationale: this is the reviewed `main` contract; Slurm
  licenses are admission controls and mutable environment metadata, not access
  credentials or allocation attestation.
  Date/Author: 2026-08-14 / Codex.

## Outcomes & Retrospective

Implementation and final validation are pending.

## Context and Orientation

QDMI is the low-level device interface. Device providers implement its device
side. `src/qdmi/driver` loads registered provider libraries and implements the
QDMI client interface. `src/fomac` and `include/mqt-core/fomac` provide owning
C++ wrappers, while `bindings/qdmi` exposes the Python namespace. The current
v3 Python API uses `mqt.core.qdmi` for device objects and
`mqt.core.qdmi.driver` for registration and opening.

Queue telemetry consists of an optional device queue length and an optional job
queue position. Job retrieval opens an existing provider job by its stable job
identifier. A custom operation list is a provider-defined QDMI property whose
payload is an array of operation handles; FoMaC converts those handles to the
same owning `Operation` objects used by the standard operation property.

The Slurm adapter reads `SLURM_JOB_LICENSES`, accepts one registered local
device ID with an omitted count or count one, opens a fresh device session, and
accepts device states `IDLE` and `BUSY`. The environment variable is mutable.
The adapter therefore selects a device but does not authenticate the user or
prove an allocation. Provider credentials or operating-system isolation remain
the authorization boundary.

## Plan of Work

First adapt the implementation and tests from #2010 to the v3 driver and the
new QDMI Python namespace. Update the bundled QDMI revision to the version that
defines the queue and retrieval client interfaces. Preserve optional-provider
loading so old provider libraries remain usable and report unsupported
retrieval rather than failing to load.

Next add the high-level retrieval and custom-operation APIs from #2008 and
#2042. Keep returned devices, jobs, sites, and operations owning their provider
sessions. Return `None` or `std::nullopt` for unsupported optional properties,
reject malformed byte counts, and preserve provider error codes.

Then adapt #2025 and #2043 directly to the v3 namespace established by #2113.
Keep the mechanism-specific `fomac::slurm::openDeviceFromLicense()` C++ API and
`mqt.core.qdmi.slurm.open_device_from_license()` Python API. Add the modern
Slurm 25.11 fixture, wheel-based container installation, DDSIM and SC license
scenarios, and the trust-boundary tutorial.

Keep one signed commit per upstream pull request. Resolve the combined
`CHANGELOG.md` additions semantically and do not carry obsolete intermediate
plans or superseded API names from development history.

## Concrete Steps

Run all commands from the repository root through the worktree-local wrapper
when they create caches or build artifacts.

Apply the five upstream changes in dependency order and inspect each staged
range before committing. Regenerate recursive stubs after all binding changes:

    ./.agent/run.sh uvx nox -s stubs

Configure and build the native release tree, then run the focused driver,
FoMaC, DDSIM, and Slurm-selector tests:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./.agent/run.sh ctest --preset release

Install the Python package and run the QDMI tests:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    ./.agent/run.sh uv run --no-sync pytest test/python/qdmi -q

Run strict documentation and repository validation:

    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check origin/v3.x...HEAD

The privileged Slurm workflow is executed by GitHub Actions because it requires
Docker, systemd, cgroup v2, and two Slurm compute containers.

## Validation and Acceptance

The driver tests must prove absent and present queue values, a fresh status
query for each job queue-position request, supported and unsupported retrieval,
correct job ownership, custom operation handles, malformed custom payloads, and
operation-property queries. Python tests must prove equivalent `None`, error,
and lifetime behavior through `mqt.core.qdmi`.

The selector tests must accept only `<registered-device-id>` and
`<registered-device-id>:1`. They must reject a missing value, whitespace,
unknown IDs, remote licenses, malformed or non-unit counts, and AND or OR
expressions. It must open a fresh device in `IDLE` or `BUSY` state and reject
all other states with the ID and status in the diagnostic.

The privileged fixture must run two held DDSIM Bell jobs on separate nodes,
leave a third pending for the DDSIM license while CPUs remain available, run an
SC job independently, release one DDSIM job, admit the pending job, and validate
256 samples containing only `00` and `11`. It must also prove that the selector
rejects compound and non-unit expressions. The container must import the built
wheel without `PYTHONPATH`, and the packaged `slurmd` unit must report
`Delegate=yes`.

## Idempotence and Recovery

Builds, stub generation, tests, and documentation commands are repeatable. A
failed source adaptation is repaired in the current logical commit rather than
discarding unrelated work. If `v3.x` advances before publication, rebase the
complete sequence, resolve changelog conflicts semantically, regenerate stubs,
and rerun the affected tests. Never modify another task worktree.

## Artifacts and Notes

The selected upstream commits are the squash commits for #2010, #2008, #2042,
#2025, and #2043 on `main`. Pull request #2113 is the authoritative v3 prefix.
Final exact heads, test counts, and any environment-limited checks will be added
here after validation.

## Interfaces and Dependencies

The final public additions are:

- optional C++ and Python device queue-length queries and job queue-position
  queries;
- `fomac::Device::retrieveJobById(std::string_view)` and
  `mqt.core.qdmi.Device.retrieve_job_by_id(str)`;
- `fomac::Device::queryCustomOperations(fomac::CustomProperty)` and
  `mqt.core.qdmi.Device.query_custom_operations(CustomProperty)`;
- `fomac::slurm::openDeviceFromLicense()` and
  `mqt.core.qdmi.slurm.open_device_from_license()`.

The bundled QDMI revision must expose the matching queue and retrieval enums and
functions. The Slurm fixture requires Slurm 25.11 or newer, Ubuntu 26.04,
systemd, Munge, cgroup v2, and Docker Compose. Runtime device authorization is
provider-specific and remains outside the Slurm selector.

Revision note: created this plan after the first namespace backport merged and
after classifying every `main` change merged since the previous combined v3
backport.
