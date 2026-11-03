# Shared HPCQC playground for QDMI workflows

Status: implementation and local validation complete. Docker Desktop on Mac
hardware remains untested.

## Supported deployment

`examples/slurm` provides login submission, scalable classical and quantum
access nodes, and persistent SlurmDBD/MariaDB accounting. Every QAN can compile
circuits and reach all configured devices through QDMI. One environment and
catalogue serve all implementations. Shared application files connect workflow
stages. Partitions place jobs; licenses limit concurrent device access; device
services authorize requests.

Single-device examples derive a catalogue ID from `SLURM_JOB_LICENSES`,
requesting one unit of one local license. Applications using several devices
pass explicit IDs to independent driver sessions. Scheduler policy stays in the
playground, outside the installed MQT Core library. The released C++/Python
Slurm adapters and proposed native checker are removed; the PR describes the API
migration.

The monitor probes status in a bounded, quiet child process and updates
license-only reservations. Its credentials remain separate from workload
credentials. Failed probes close admission; systemd cleans descendants when a
monitor exits. Availability is a snapshot for the monitor identity. Controller
communication failures can leave previous scheduler state active and require
administrator attention. QDMI imports resolve lazily inside the worker's
exception boundary in the playground's Python runtime.

## Installation and portability

Docker builds the checked-out MQT Core source with the official setup-mlir
installer, using the private package version 4.1.0. Ordinary uv requirements
select additional PyPI, Git, or local-source packages. Wheels include device
libraries and manifests; all workloads use those installed packages. A separate
native installation mode and catalogue-generation callback are unnecessary. MQT
Core supplies Qiskit and PennyLane extras. Applications add further requirements
when needed, such as IQM's estimator dependencies.

The completed environment is copied into the runtime image; compilers and the
LLVM SDK stay in builder stages. Linux Docker Engine and macOS Docker Desktop
use the same Compose settings. Privileged containers, the host cgroup namespace,
and the cgroup-v2 mount follow Docker Desktop's documented systemd setup;
Enhanced Container Isolation is incompatible with this access.

## Documentation and device CI

`docs/qdmi/slurm.md` is the canonical guide for the playground and existing
clusters. Its topology shows the shared filesystem and the QDMI interface
between QANs and individual device implementations. IQM and Braket link to that
guide. The scientific application remains separate; the guide demonstrates
dependencies between quantum measurement and classical propagation jobs.

Device repositories skip authenticated jobs for upstream PRs from forks while
retaining ordinary unit and Python checks. Contributors can run credentialed CI
inside their own forks using their own repository secrets; those secrets do not
reach upstream PR jobs. There is no reviewed-SHA dispatch workflow or
Dependabot-specific branch. Secret requirements remain explicit for trusted
runs. IQM quantum-computer identifiers and aliases remain distinct from
site-defined Slurm license names.

## Validation

All PR bases are current: Core `974e4fdc`, IQM `d6b57cc7`, Braket `95719f9d`.
The simplified runner/monitor suite passes 15 cases. The source-built Core
cluster passes with three classical nodes and two QANs: DDSIM/SC execution,
license selection and contention, failed-probe admission and recovery,
environment export, partition placement, and persistent accounting. The full run
took 148 seconds and installed MQT Core 4.1.0.

Authenticated package-only workloads passed on Emerald (73 seconds) and SV1 (340
seconds), including verification of the installed device libraries. Full lint
passes in all three repositories. Strict Core documentation and generated site
links pass; all five device notebooks executed. IQM's normal documentation gate
retains 19 existing reference/rendering warnings; Braket's strict build is
clean. Chrome inspection found no layout, console, or network errors at desktop
and mobile widths. Hosted status must be reported for the published heads,
separately from local checks.

A cache-preserving requirements change reran the shared environment stage in 1.9
seconds (5.64 seconds for the build) without recompiling Braket or its SDK. The
resulting image passes dependency checks and reports MQT Core 4.1.0.
