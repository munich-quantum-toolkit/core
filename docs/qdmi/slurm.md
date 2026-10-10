# Use QDMI devices with Slurm

Slurm can limit concurrent access to a quantum device through a cluster-wide
license. Use a QDMI device ID as the license name, then open that device in the
job with MQT Core. The job can compile and submit workloads through the same
QDMI interface it uses outside Slurm.

Slurm schedules jobs; the device implementation authenticates users and submits
quantum work. A license does not grant device access or reserve capacity at a
remote service. Device availability and queues can change after admission.

## Run a job

Install MQT Core and the required QDMI device implementation in the workload
environment. Make its catalogue, libraries, and credentials available on the
compute nodes. See [device configuration](configuration.md) and the device
implementation's installation guide.

An administrator registers each device ID in `slurm.conf`, for example:

```ini
Licenses=mqt.sc.default:1,amazon.braket.sv1:2,iqm.emerald:1
```

The counts limit simultaneous Slurm allocations. Choose counts appropriate for
the device and the site's access policy.

Request one device with `--licenses=ID` or `--licenses=ID:1`:

```bash
#!/bin/bash
#SBATCH --licenses=amazon.braket.sv1
#SBATCH --time=00:05:00
set -eu

source /shared/quantum/.venv/bin/activate
export MQT_CORE_QDMI_CONFIG_FILE=/shared/quantum/devices.json
export AWS_PROFILE=research
mqt-core-qdmi-check --device amazon.braket.sv1 --timeout 10
srun python workload.py
```

In `workload.py`, select the allocated device:

```python
from mqt.core.qdmi import slurm

device = slurm.open_device_from_license()
```

Pass `device` to the appropriate Qiskit or PennyLane adapter. MQT Core accepts
one local license with a unit count and requires the device to report `IDLE` or
`BUSY`. Compound license expressions and remote licenses are unsupported.

The optional [availability command](driver.md#probe-device-availability) gives a
quick indication that the device is operational. Run it after activating the
workload environment and setting credentials. It does not reserve the device.
The workload still opens the device and handles submission errors normally.

Slurm exports the submission environment by default. Use ordinary environment
variables or Slurm's `--export` option for job-specific settings. Variables set
inside a batch script are inherited by its subsequent `srun` steps.

## Hold jobs while a device is unavailable

A check inside a job runs after Slurm has allocated resources. To leave jobs
pending while a device is unavailable, an administrator must update scheduler
state independently of the jobs.

Slurm supports
[license-only reservations](https://slurm.schedmd.com/reservations.html) for
unavailable shared resources. Reserve the device's entire configured license
count for an administrative account. New jobs requiring it remain pending with
reason `Licenses`; running jobs continue, and other devices remain usable.
Remove the reservation after a successful health check.

The
[availability monitor example](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
performs one such update. It blocks the licenses before invoking the bounded
QDMI checker and removes the block only on success. Run it periodically from a
trusted administrative host with Slurm clients, MQT Core, the device
implementation, and site-owned credentials. Initialize the blocks before
admitting workloads. The controller does not need device libraries.

Use a site account that can assess the shared device's operational state. One
user's expired credentials must not determine availability for every user.
Conversely, a successful site check does not verify each user's authorization.
The monitor is a snapshot: device status can change between checks, and a
stopped monitor leaves its last scheduler state in place. Supervise it and alert
on stale updates. Keep the optional job check for the user's environment.

This integration uses licenses and Slurm's environment export. Device libraries
run in application or checker processes; no Slurm plugin is needed. The
[QRMI integration paper](https://arxiv.org/abs/2607.19591) describes a separate
acquire/execute/release lifecycle for services that issue allocation tokens. The
QDMI device implementations used here do not require that lifecycle.

## Configure the cluster

Use matching Slurm versions across the cluster. This integration requires Slurm
25.11 or newer.

| Location               | Software and configuration                           |
| ---------------------- | ---------------------------------------------------- |
| Login/submission nodes | Slurm clients and access to the workload environment |
| Controller             | `slurmctld`, scheduling policy, and license counts   |
| Compute nodes          | `slurmd`, cgroup v2, and the workload environment    |
| Accounting service     | `slurmdbd` when persistent accounting is needed      |

Static local licenses do not require an accounting database. The controller does
not need device libraries or SDKs. Use consistent numeric user/group IDs and
readable catalogue/library paths across compute nodes. A shared versioned
environment and identical per-node installations are both suitable.

Keep scheduler authentication, such as Munge, separate from device credentials.
For CPU and allocated-memory constraints, use memory-consuming selection with
cgroup enforcement:

```ini
# slurm.conf
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU_Memory
```

```ini
# cgroup.conf
CgroupPlugin=cgroup/v2
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
```

Set node resources, memory defaults, partitions, accounts, and limits for the
site. See the
[Slurm administration guide](https://slurm.schedmd.com/quickstart_admin.html)
and [cgroup configuration](https://slurm.schedmd.com/cgroup.conf.html).

`SLURM_JOB_LICENSES` is mutable within a process. MQT Core uses it for device
selection, not as proof of allocation or authorization. Device services and
operating-system permissions must enforce access independently.

## Try the Docker cluster

The reusable
[Docker Slurm setup](https://github.com/munich-quantum-toolkit/core/tree/main/docker/slurm)
supports local demonstrations and integration tests with a configurable number
of compute containers. Follow its README to build the workload image, start the
cluster, submit jobs, and remove it.

Use rootful Docker on a disposable Linux cgroup-v2 host. The containers run
systemd and require privileged access to the host cgroup hierarchy. Jobs run as
an unprivileged user. This setup is intended for development and demonstrations,
not a production security boundary.
