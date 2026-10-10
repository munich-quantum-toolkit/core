# Shared Slurm playground for QDMI workflows

Status: complete; the shared cluster is implemented and validated locally.

## Scope and decisions

`examples/slurm` provides a login node, scalable classical compute nodes,
dedicated IQM and Braket quantum access nodes, and persistent slurmdbd/MariaDB
accounting. Partitions route allocations; licenses limit concurrent access.
Neither grants remote device authorization. All nodes share one installed
software environment and catalogue; users and monitors authenticate separately.

Use standard Slurm accounting and job dependencies. The application belongs in
its own repository: measurement jobs use a QAN and device license, then
classical propagation runs in a dependent compute job. Tightly coupled workflows
require an allocation spanning their resources, rather than nested submissions
to another partition. Docker supplies hosts for this disposable cgroup-v2
playground.

Accounting stores job metadata and license TRES, not environments or scripts.
Private generated database credentials and Munge keys stay outside images.
Database and controller state survive normal restarts; deleting volumes resets
history. Operational monitoring accepts IDLE/BUSY for the monitor identity. It
cannot establish user authorization or predict queueing at a remote device.

## Validation

The release build and all 597 QDMI CTest cases completed with one existing skip
and no failures. The installed Python 3.15 command, discovery, and Slurm tests
passed, as did the 20 runner/monitor cases and full-file C++ lint. The complete
Sphinx build executed all 18 notebooks and passed internal-link checks.

`test/slurm/run_integration.py --nodes 3` passed with the freshly rebased wheel:
login environment, QAN placement, DDSIM/SC execution, license contention,
failed-probe admission, recovery, environment export, and accounting after
service recreation. Repository lint passed. Native/wheel authenticated device
workloads are maintained and run in the respective device repositories.

## Test boundaries

Tests cover real DDSIM/SC execution, capacity contention, failed-probe
admission, recovery, and job/daemon environment separation. The device
repositories test actual authenticated executions and selected native/wheel
libraries. The manual multivendor harness was removed; simultaneous cross-vendor
qualification belongs with the scientific application. Duplicate Bell
assertions, wrapper scripts, synthetic job-script checker scenarios, and
redundant state matrices are omitted. Runner unit tests cover resource cleanup
and invalid external inputs.

## Dependency boundary

Device package dependencies track current MQT Core main. Their Slurm CI must
build this integration PR until its checker and Slurm Python API reach main; a
main-only wheel cannot yet run the shared deployment.
