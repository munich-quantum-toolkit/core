# Use QDMI devices with Slurm

Slurm limits concurrent access to a quantum device through a cluster-wide
license. A QDMI device ID names the license; the job opens that device through
MQT Core's driver. The same environment can contain implementations from several
vendors, and independent sessions can submit work to their devices concurrently.

An administrative monitor checks each device's operational status and keeps its
license reserved while it is unavailable. Jobs then wait in Slurm instead of
occupying compute nodes during an outage. Device implementations authenticate
users and submit quantum work; licenses do not grant device access or reserve
capacity at a remote service.

## Run a job

Install MQT Core and the required device implementations in one workload
environment. Make its catalogue, libraries, and credentials available on compute
nodes. See [device configuration](configuration.md) and each device
implementation's installation guide.

An administrator registers the device IDs and concurrency limits in
`slurm.conf`, for example:

```ini
Licenses=mqt.sc.default:1,amazon.braket.sv1:2,iqm.emerald.mock:1
```

Request one device with `--licenses=ID` or `--licenses=ID:1`:

```bash
#!/bin/bash
#SBATCH --licenses=amazon.braket.sv1
#SBATCH --time=00:05:00
set -eu

source /shared/quantum/bin/activate
export MQT_CORE_QDMI_CONFIG_FILE=/shared/quantum/catalogue.json
export AWS_PROFILE=research
srun python workload.py
```

In `workload.py`, open the allocated device:

```python
from mqt.core.qdmi import slurm

device = slurm.open_device_from_license()
```

Pass `device` to the Qiskit or PennyLane adapter. This convenience function
accepts one local license with a unit count and requires the device to report
`IDLE` or `BUSY`. It initializes only the selected device, so an unrelated
device's initialization does not delay the job. Applications using several
devices can request their licenses explicitly and open separate
[built-in driver sessions](driver.md).

Slurm exports the submission environment by default. Use environment variables
or `--export` for job settings; variables set in a batch script reach its
subsequent `srun` steps. IQM and AWS authentication settings can coexist in the
same environment. The workload must handle authentication and submission
failures even after a successful site availability check.

## Keep unavailable devices out of allocations

The
[cluster example](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
includes an availability monitor for every configured device license. Each
monitor creates a Slurm
[license-only reservation](https://slurm.schedmd.com/reservations.html) for the
full license count, then invokes the
[QDMI availability command](driver.md#probe-device-availability). It removes the
reservation only after a successful check. New jobs otherwise remain pending
with reason `Licenses`; running jobs and unrelated devices continue.

Run monitors on an administrative host with Slurm clients, the shared QDMI
environment, and site-owned credentials. Monitors use separate sessions to probe
their selected devices in that environment. They run outside Slurm daemons.
Initialize the reservations before opening the queue; the example's systemd
units enforce this ordering, bound probes, restart failed monitors, and close
admission when a monitor stops.

A monitor's credentials must represent site access. One user's expired token
must not determine cluster-wide availability, and a successful site check does
not verify each user's authorization. Availability is a snapshot: device status
can change after admission. Alert on controller errors, which prevent the
monitor from updating reservations.

This integration uses Slurm licenses, reservations, and environment export; it
requires no Slurm plugin. The
[QRMI integration paper](https://arxiv.org/abs/2607.19591) also describes an
acquire/execute/release lifecycle for services that issue allocation tokens. The
QDMI device implementations used here do not need that lifecycle.

## Configure the cluster

Use matching Slurm versions, at least 25.11, across the cluster.

| Location               | Software and configuration                           |
| ---------------------- | ---------------------------------------------------- |
| Login/submission nodes | Slurm clients and access to the workload environment |
| Controller             | `slurmctld`, scheduling policy, and license counts   |
| Administrative host    | Availability monitors and site-owned credentials     |
| Compute nodes          | `slurmd`, cgroup v2, and the workload environment    |
| Accounting service     | `slurmdbd` when persistent accounting is needed      |

Static local licenses do not require an accounting database. The controller
needs no device libraries unless it also hosts the monitors. Use consistent
numeric user/group IDs and readable catalogue/library paths across nodes. A
shared versioned environment or identical per-node installations both work.

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

`SLURM_JOB_LICENSES` is mutable within a process. MQT Core uses it for
selection, not proof of allocation or authorization. Device services and
operating-system permissions enforce access independently.

The
[example cluster](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
puts these components together for demonstrations and integration tests. Docker
Compose supplies its hosts; the jobs, QDMI configuration, and Slurm scheduling
follow the same interfaces as other clusters.
