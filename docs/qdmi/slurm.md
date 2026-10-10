# Use QDMI devices with Slurm

Slurm schedules access to a QDMI device through a cluster-wide license named
after its device ID. MQT Core's QDMI driver opens the allocated device. One
administrator-maintained environment can serve several device implementations,
each with its own authentication and independent sessions.

An availability monitor reserves a device's licenses while it is unavailable.
Jobs then wait for the device without occupying compute nodes. Licenses control
admission; they do not authorize access or reserve capacity at a remote service.

## Submit a workload

The administrator provides the quantum software environment and device catalogue
on the login and compute nodes. Log in, then save this script as `job.sh` and
submit it with `sbatch job.sh`:

```bash
#!/bin/bash
#SBATCH --licenses=mqt.ddsim.default
#SBATCH --time=00:05:00
set -eu

srun python workload.py
```

For example, `workload.py` can execute a Bell circuit on DDSIM:

```python
from mqt.core.qdmi import ProgramFormat, slurm

device = slurm.open_device_from_license()
program = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
job = device.submit_job(program, ProgramFormat.QASM2, 1024)
if not job.wait(60):
    raise TimeoutError("The quantum job did not finish within 60 seconds")
print(job.get_counts())
```

For a remote device, select the site's quantum access partition as well as its
license, for example
`sbatch --partition=iqm --licenses=iqm.emerald.mock job.sh`. Partitions choose
where the classical part of a quantum workload runs; licenses are cluster-wide.
See the {doc}`workflow playground <slurm_cluster>` for a login node, classical
nodes, quantum access nodes, and persistent accounting.

The same `device` can be passed to a {doc}`Qiskit <qdmi_backend>` or
{doc}`PennyLane <pennylane_device>` adapter. The Slurm convenience function
accepts one local license, `ID` or `ID:1`, and requires status `IDLE` or `BUSY`.
It opens only that device. An application using several devices requests their
licenses together and opens separate {doc}`built-in driver sessions <driver>`.

Slurm
[exports the submission environment by default](https://slurm.schedmd.com/sbatch.html#OPT_export).
Users authenticate through the device implementation's usual mechanism before
submission. Credential files must be accessible to the job on compute nodes. IQM
and AWS settings can coexist; jobs do not need separate Python environments. An
availability check uses site credentials and does not establish a user's
authorization. Workloads must handle authentication and execution failures.

## Provide the shared environment

Install MQT Core and the required device implementations in one versioned,
administrator-owned environment. Expose its executables through the site's
default `PATH` or a software module. Where several environments are offered,
users select the appropriate module before `sbatch` or load it in their batch
script; the module configures paths without each user maintaining a virtual
environment. See, for example, the
[GWDG module guide](https://docs.hpc.gwdg.de/software_stacks/module_basics/index.html).

Installed wheels supply device manifests. Put shared, non-secret settings in
`/etc/mqt-core/qdmi.json` on the submission, compute, and monitoring hosts. The
built-in driver discovers this file automatically; see {doc}`configuration` for
its format and precedence. Keep authentication out of the shared catalogue: jobs
use their users' credentials, while administrative monitors use site
credentials. Environment variables set for a Slurm daemon do not configure
users' jobs.

Use matching software paths and numeric user/group IDs across nodes. A shared
filesystem or identical per-node installations both work. The
{doc}`example cluster <slurm_cluster>` supplies this environment and includes
login submission, device partitions, availability checks, and accounting.

## Schedule available devices

Register device IDs and concurrency limits in `slurm.conf`, for example:

```ini
Licenses=mqt.ddsim.default:2,iqm.emerald.mock:1,amazon.braket.sv1:2
```

The example runs one administrative monitor per device license. Before probing,
it creates a
[license-only reservation](https://slurm.schedmd.com/reservations.html) for the
full license count. It removes that reservation only after the
[availability command](driver.md#probe-device-availability) succeeds. Failed or
timed-out checks leave new jobs pending with reason `Licenses`; running jobs and
other devices continue.

Run monitors outside Slurm daemons, with the shared QDMI environment and
site-owned credentials, with one monitor per device ID across the cluster.
Initialize reservations before opening the queue. The example's systemd units
enforce this order, bound checks, restart failed monitors, and close admission
when a monitor stops. Investigate controller communication errors: failed
reservation updates can leave the last scheduler state in effect. Availability
is a snapshot, so a device can fail after allocation.

Slurm licenses, reservations, and environment export provide this integration;
no Slurm plugin is required. `SLURM_JOB_LICENSES` is mutable process data: MQT
Core uses it for selection, while device services and operating-system
permissions enforce access.

## Configure compute resources

Use matching Slurm versions, at least 25.11, across the cluster. Submission
nodes need Slurm clients; compute nodes need `slurmd` and the workload
environment. The controller runs `slurmctld` and needs device libraries only if
it also hosts the monitors. Static local licenses do not require `slurmdbd`; add
accounting when the site needs persistent records.

For CPU and allocated-memory constraints, configure:

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
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
```

Use cgroup v2 hosts. Slurm's `autodetect` default selects the host's cgroup
implementation; cgroup v1 is deprecated. Set node resources, memory defaults,
partitions, accounts, and limits for the site. Keep Slurm authentication, such
as Munge, separate from device credentials. See the
[Slurm administration guide](https://slurm.schedmd.com/quickstart_admin.html)
and [cgroup configuration](https://slurm.schedmd.com/cgroup.conf.html).
