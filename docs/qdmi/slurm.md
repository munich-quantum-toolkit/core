# Use QDMI devices with Slurm

This example uses Slurm 25.11 or newer on Ubuntu 26.04. It has one controller,
two compute nodes, and two CPUs on each compute node.

A Slurm license controls admission to a cluster-wide resource. In this setup,
the license name is a stable QDMI device ID. A license does not show provider
availability. It does not show the device queue. The QDMI provider supplies that
information when its interface supports it.

## Understand the control boundaries

This example is suitable for admission and accounting tests on a cooperative
cluster. It does not make a Slurm license an access-control credential. The
controls are independent:

- Slurm admits jobs and accounts for the configured license count.
- The MQT Core adapter uses the license environment to select a registered QDMI
  device.
- The QDMI provider reports device availability and queue data.
- The provider or the operating system authorizes access to the device.

`SLURM_JOB_LICENSES` is process-mutable. A job can change it before it calls
`slurm.open_device_from_license()`. Thus, the function does not prove that Slurm
allocated the named license. It does not authenticate the user. It does not
authorize access. A lookup through another Slurm interface would not make MQT
Core an access-control boundary because a program can also call
`driver.open_device(device_id)` directly.

## Install the software

Install the same MQT Core package on each compute node. The package contains the
MQT Core QDMI interface and the bundled QDMI devices. You can use a shared
software environment or install the same wheel on each node.

Install Slurm, Munge, and systemd. Start Munge before Slurm. Use the same Munge
key on all nodes. Keep this key outside the QDMI device configuration.

Use the unified cgroup v2 hierarchy. Add these settings to `slurm.conf`:

```ini
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU
```

Use the cgroup plugin to constrain processors and memory. For example, use this
`cgroup.conf`:

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
```

This fixture has no local device file. For strict isolation of a local device,
configure it as a Slurm GRES with a `File=` entry. Also set
`ConstrainDevices=yes` for the cgroup task plugin. Slurm can then restrict the
device files that a job can open. See the [Slurm GRES configuration]
documentation. A remote QPU has no local device file, so the provider must
enforce its authorization.

Run `slurmd -C` on each compute node and use its output for the `NodeName`
record. The MQT Core test uses two CPUs on `node1` and `node2`.

## Register the devices

MQT Core installs persistent definitions for `mqt.ddsim.default` and
`mqt.sc.default`. You do not have to add another registry file for these two
devices. You can verify the stable IDs before you configure Slurm:

```console
python -c "from mqt.core.qdmi import driver; print(*driver.registered_device_ids(), sep='\n')"
```

For an external provider, install its shared library and QDMI manifest. You can
also add one trusted system registry file at `/etc/mqt-core/qdmi.json`. The `id`
field in that file is the stable device ID. Use the same ID as the local Slurm
license name. Do not put short-lived access tokens in this file. Use the
authentication method that the provider documents for batch jobs.

The Slurm adapter does not supply credentials. IQM can use its configured token
source. Amazon Braket uses the AWS credential provider chain, such as an
instance role, workload identity, or AWS profile. A job can also export provider
configuration. Use provider-scoped credentials with minimum permissions. Do not
store access tokens or AWS access keys in a persistent device definition.

Add the licenses to `slurm.conf` on the controller:

```ini
Licenses=mqt.ddsim.default:2,mqt.sc.default:1
```

The DDSIM count is two. Therefore, Slurm can admit two jobs that each request
one DDSIM license. These jobs can run on different nodes. The count is a local
cluster policy. It is not a DDSIM property and it is not a per-node count.

Restart `slurmctld` after you first add the licenses. Reconfigure the compute
nodes as required by your Slurm installation. Then inspect the configured
resources:

```console
scontrol show lic
```

The initial report must contain these values:

```text
LicenseName=mqt.ddsim.default Total=2 Used=0 Free=2 Remote=no
LicenseName=mqt.sc.default Total=1 Used=0 Free=1 Remote=no
```

## Submit a DDSIM job

Save this program as `bell.py` in a location that all compute nodes can read:

```python
from mqt.core.qdmi import ProgramFormat, slurm

program = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

device = slurm.open_device_from_license()
job = device.submit_job(program, ProgramFormat.QASM2, num_shots=256)
if not job.wait(60):
    raise RuntimeError("DDSIM did not finish within 60 seconds")

counts = job.get_counts()
if sum(counts.values()) != 256 or not set(counts) <= {"00", "11"}:
    raise RuntimeError(f"Invalid Bell results: {counts}")
print(counts)
```

Save this batch script as `bell.sbatch`:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --licenses=mqt.ddsim.default:1
#SBATCH --output=bell-%j.out

set -euo pipefail
python bell.py
```

The adapter reads `SLURM_JOB_LICENSES` and selects the persistent device
definition with the same ID. It requires one unambiguous QDMI device license. It
opens a fresh device session and checks the device status. The function accepts
`IDLE` and `BUSY`. This check is not authorization. The provider can still
reject a later submission or put the quantum task in its device queue.

Submit the job with this command:

```console
sbatch bell.sbatch
```

## Check concurrent jobs

For a scheduling test, add a sufficiently long classical post-processing step
after the Python command. For example, add `sleep 120` to `bell.sbatch`. Then
submit two jobs:

```console
sbatch --nodelist=node1 bell.sbatch
sbatch --nodelist=node2 bell.sbatch
```

Both jobs can run because two DDSIM licenses exist. Each job uses one CPU. One
CPU remains free on each node. Submit a third DDSIM job. Slurm keeps it pending
until one DDSIM license becomes free.

Use these commands to inspect the state:

```console
squeue --format="%.18i %.9T %.20R %.12N %.20L"
scontrol show lic mqt.ddsim.default
scontrol show node node1
scontrol show node node2
```

The third job must have state `PENDING` and reason `Licenses`. The license
report must show `Total=2 Used=2 Free=0`. The node records must still show one
free CPU on each node. A job that requests `mqt.sc.default:1` can use one of
these CPUs because it uses a different license.

When one DDSIM job ends, Slurm returns its license. The pending job can then
start without a change to the device registry or `slurm.conf`.

## Diagnose a failure

First, run `scontrol show job <job-id>`. Check `JobState`, `Reason`, `Licenses`,
and `NodeList`. Use `scontrol show lic` to compare the total, used, and free
counts. Use `scontrol show node` to check CPU allocation.

If a node is down, check `systemctl status munge slurmd` and the Slurm journal
on that node. Check that `/sys/fs/cgroup/cgroup.controllers` exists. Check that
all nodes use the same Munge key and the same `slurm.conf`.

If MQT Core cannot select a device, print `SLURM_JOB_LICENSES` inside the batch
job and list the registered QDMI IDs. Use this value only to diagnose selection.
It is not proof of the Slurm allocation. The license name and stable ID must
match exactly. Do not add a generic device license. Do not use a Slurm OR
license expression for device selection because the environment does not
identify a single selected device in that case.

[Slurm GRES configuration]: https://slurm.schedmd.com/gres.conf.html
