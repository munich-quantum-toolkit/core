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

## Optional site defaults with SPANK

Use MQT Core's SPANK module when a device license should select default
catalogue paths or credential references. Jobs can override these defaults
through their environment. No plugin is needed when jobs already supply their
configuration.

Build the standalone module against the cluster's Slurm development headers:

```console
cmake -S spank -B build/spank -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build/spank
cmake --install build/spank
```

The build requires Linux, CMake, and a C++20 compiler. It does not need LLVM or
device SDKs. Install the module and its `plugstack.conf` entry on compute nodes.
If login or submission hosts use the same plugstack configuration, install the
module there too. Rebuild it when changing Slurm major versions.
`MQT_CORE_SPANK_INSTALL_DIR` selects the module installation directory.

Load the module once, with the permitted device IDs and non-secret defaults:

```ini
required /usr/local/lib/slurm/mqt-core-qdmi-spank.so licenses=amazon.braket.sv1,iqm.emerald qdmi_config_file=/etc/mqt-core/qdmi.json reference=AWS_PROFILE:amazon.braket.sv1:quantum reference=IQM_TOKENS_FILE:iqm.emerald:/shared/iqm/tokens.json
```

`qdmi_config_file=PATH` supplies `MQT_CORE_QDMI_CONFIG_FILE`. Each
`reference=ENV:ID,ID:DEFAULT` supplies an environment variable for the listed
device IDs. Use paths, profile names, and other non-secret references; keep
tokens and passwords out of `plugstack.conf`.

Defaults apply only to an exact configured `ID` or `ID:1` license expression.
Other jobs pass through unchanged. Values already present in the job environment
take precedence. The module never reads credential files or loads a device
implementation, and it does not copy credentials from Slurm daemons. Invalid
settings fail the job without draining the compute node.

The module is GPL-3.0-or-later and distributed in the source checkout,
separately from MQT Core's MIT-licensed runtime, wheels, and source packages.

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
