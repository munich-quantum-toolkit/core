# Hybrid quantum-classical workflows with Slurm

Use Slurm to coordinate scientific workflows that combine classical computation
with quantum devices. Submit from a login node, compile and offload circuits on
{term}`quantum access nodes <quantum access node>` (QANs), and share inputs and
results with classical compute jobs. Every QAN uses
**QDMI as its device interface**. One software environment can connect DDSIM,
IQM, Amazon Braket, and other QDMI device implementations.

```{figure} ../_static/slurm-playground.svg
:alt: Login, classical, and quantum access nodes share application files and results. QANs connect to individual devices through one QDMI interface. Slurm schedules jobs and records accounting data.

A shared filesystem connects workflow stages; QDMI connects quantum devices.
```

| Setting            | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| **Partition**      | Place applications on classical or quantum access nodes. |
| **Device license** | Limit concurrent allocations for a QDMI device.          |
| **Credentials**    | Authenticate with the device service.                    |

The playground names each Slurm license after its QDMI device ID. Licenses
coordinate allocations within the cluster; they neither authorize device access
nor reserve capacity at a remote service. An availability monitor keeps new jobs
pending while a device is unavailable.

Already have a cluster? Start with [job submission](#submit-your-first-job) or
[site configuration](#deploy-on-an-existing-cluster). Otherwise, the
[playground sources](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
provide a cluster to experiment with:

| Host                     | Role                                                             |
| ------------------------ | ---------------------------------------------------------------- |
| `login`                  | User shell, Slurm clients, and shared software environment       |
| `controller`             | Scheduling and device availability monitors                      |
| `node` (two by default)  | Classical computation in the `compute` partition                 |
| `qan` (two by default)   | Compilation and access to all devices in the `quantum` partition |
| `accounting`, `database` | Persistent job and resource accounting with SlurmDBD and MariaDB |

## Start the cluster and log in

Use Docker Engine on Linux or Docker Desktop on macOS, with Docker Compose.
Docker builds MQT Core from your checkout, downloads LLVM/MLIR with
[setup-mlir](https://github.com/munich-quantum-software/setup-mlir), and
installs one workload environment with uv. Apple Silicon uses native Linux
ARM64. Build tools and source trees stay outside the runtime image; later builds
reuse caches.

The containers need a rootful Linux Docker engine with cgroup v2 and privileged
access for systemd and Slurm resource isolation. The Compose settings follow
[Docker Desktop's systemd setup](https://docs.docker.com/desktop/release-notes/#430).
Disable Enhanced Container Isolation and allocate at least four CPUs and 12 GiB
of RAM to Docker for small demonstrations. Use trusted development workloads.

From the MQT Core checkout:

```bash
sh examples/slurm/prepare.sh
docker compose -f examples/slurm/compose.yml up --build -d --wait
docker compose -f examples/slurm/compose.yml exec --user 10000:10000 login bash --login
```

Jobs run as `mqt-dev` (UID/GID 10000). Each classical node advertises two CPUs
and 2 GiB of scheduled memory; each QAN advertises four CPUs and 8 GiB. All
containers share the Docker engine's resources: scaling nodes does not add
physical capacity. Cgroup v2 enforces allocations.

The shared environment at `/opt/venv` is on `PATH`; jobs need no activation or
package installation. `/jobs` holds shared application files and results, backed
by `build/slurm/jobs` on the host. The MQT Core checkout is mounted read-only at
`/workspace`.

## Submit your first job

In the login shell, inspect partitions and licenses:

```bash
sinfo
scontrol show licenses
```

Save these two files in `/jobs`, then run `sbatch --wait job.sh`:

```{code-block} bash
:caption: job.sh

#!/bin/bash
#SBATCH --partition=quantum
#SBATCH --licenses=mqt.ddsim.default
#SBATCH --time=00:05:00
set -eu

srun python workload.py
```

```{code-block} python
:caption: workload.py

import os

from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi import builtin_driver as qdmi_driver

device_id = os.environ["SLURM_JOB_LICENSES"].removesuffix(":1")
device = qdmi_driver.open_device(device_id)
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

The Bell-circuit counts appear in `slurm-JOBID.out`. This example requests
**one unit of one local device license** and reads its ID from
[`SLURM_JOB_LICENSES`](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_JOB_LICENSES).
Applications using several devices request their licenses together, accept
explicit device IDs, and open independent driver sessions. Each session
initializes only its selected device.

Pass the same `device` to a {doc}`Qiskit <qdmi_backend>` or
{doc}`PennyLane <pennylane_device>` adapter. C++ applications use
{cpp-api:func}`qdmi::builtin_driver::openDevice`.

(multiple-device-implementations)=

## Connect quantum devices

Add implementations to `MQT_CORE_SLURM_PACKAGES` using
[uv requirements](https://docs.astral.sh/uv/pip/dependencies/). For example,
save this as `devices.txt` to build both implementations from their sources:

```{code-block} text
:caption: devices.txt

iqm-qdmi @ git+https://github.com/iqm-finland/QDMI-on-IQM.git@main
amazon-braket-qdmi @ git+https://github.com/munich-quantum-software/amazon-braket-qdmi-device.git@main
```

Use a full commit SHA instead of `main` to fix the revision. For published
packages, use names such as `iqm-qdmi` with optional version constraints. For a
local checkout, set `MQT_CORE_SLURM_WORKLOAD` to its absolute parent directory
and use a path such as `/workload/QDMI-on-IQM` in the requirements. Source
packages are compiled inside Docker for its architecture. MQT Core's Qiskit and
PennyLane extras are already installed; add any application-specific
dependencies to the same list.

Configure device license counts and accounting records in
`build/slurm/slurm.conf` before starting the updated cluster:

```ini
Licenses=mqt.ddsim.default:2,mqt.sc.default:1,iqm.emerald.mock:1,amazon.braket.sv1:2
AccountingStorageTRES=license/mqt.ddsim.default,license/mqt.sc.default,license/iqm.emerald.mock,license/amazon.braket.sv1
```

Installed packages supply device manifests. Shared endpoints and presets belong
in `build/slurm/qdmi.json`, mounted at `/etc/mqt-core/qdmi.json`; see
{doc}`configuration` for the format and precedence. Keep credentials outside
this catalogue and the image. Rebuild with the selected packages:

```bash
export MQT_CORE_SLURM_PACKAGES="$(cat devices.txt)"
docker compose -f examples/slurm/compose.yml up --build -d --wait
```

All implementations share `/opt/venv`, and every QAN can reach them. Adding a
device requires its implementation, catalogue entry, license, and credentials;
it does not require a new partition or node type. Classical nodes use an
internal network; QANs, the login node, and the monitoring controller also have
outbound connectivity.

### Supply credentials

Monitors use site credentials independently of jobs. Put their references in the
private `build/slurm/qdmi.env` file, using systemd `EnvironmentFile` syntax:

```ini
IQM_TOKENS_FILE=/run/credentials/iqm.json
AWS_SHARED_CREDENTIALS_FILE=/run/credentials/aws
AWS_PROFILE=site-monitor
```

Mount those files read-only into the controller through a Compose overlay. Users
authenticate on the login node through each device implementation's usual
mechanism. Slurm
[exports the submission environment by default](https://slurm.schedmd.com/sbatch.html#OPT_export);
credential files must also be readable by the user on the relevant QAN. IQM and
AWS settings can coexist. A daemon's environment does not configure user jobs.

Request the selected device's license when submitting a compatible application:

```bash
sbatch --partition=quantum --licenses=iqm.emerald.mock iqm-job.sh
squeue --me
```

## Admit jobs while devices are operational

A license count limits concurrent Slurm allocations, not the number of circuits
an application can submit. Choose counts according to site policy and service
capacity. The playground runs one administrative monitor per device. It probes
QDMI in a bounded child process and updates a
[license-only reservation](https://slurm.schedmd.com/reservations.html):

| Probe result       | Scheduling behavior                                           |
| ------------------ | ------------------------------------------------------------- |
| `IDLE` or `BUSY`   | Release the reservation; jobs can acquire the device license  |
| Failure or timeout | Keep all units reserved; new jobs wait with reason `Licenses` |
| Monitor stopped    | Reserve all units until monitoring resumes successfully       |

`BUSY` means operational; the remote service may queue a submitted job. Waiting
jobs occupy no compute nodes. Reservations affect new allocations, while running
jobs and other device licenses continue unaffected.

The controller reserves licenses before opening the partitions. Each
`qdmi-availability@ID:COUNT.service` checks its device every 30 seconds with a
ten-second probe timeout. Systemd restarts failed monitors and closes admission
when a monitor stops. Probe output is suppressed because device diagnostics can
contain credentials; service failures appear in the journal. From an
administrator shell on the controller, exercise admission with:

```bash
systemctl stop qdmi-availability@mqt.ddsim.default:2.service
systemctl start qdmi-availability@mqt.ddsim.default:2.service
```

Open that shell with
`docker compose -f examples/slurm/compose.yml exec controller bash`. The monitor
maintains `qdmi-unavailable-ID`; do not create a conflicting reservation.
Inspect both `Free` and `Reserved` in `scontrol show licenses`, since reserved
tokens can still appear in `Free`.

:::{important}
Availability is a snapshot for the monitor's identity, not user authorization.
Applications must handle expired credentials, device failures, and remote
queueing. Controller communication failures require administrator attention: a
failed reservation update can leave the last scheduler state in effect.
:::

## Chain quantum and classical stages

A quantum-assisted AFQMC application can measure a trial state on a QAN, save
results to `/jobs`, and propagate classical walkers on compute nodes. Submit its
two batch scripts from the login shell:

```bash
measurement=$(sbatch --parsable --partition=quantum --licenses=iqm.emerald.mock measure.sh)
sbatch --partition=compute --dependency=afterok:"$measurement" propagate.sh
```

The scripts choose their resources and agree on an output path. The classical
stage starts only after successful measurement and holds no quantum license.
Scientific code, validation, and dependencies belong with the application.

For tightly coupled iterations, use a
[heterogeneous allocation](https://slurm.schedmd.com/heterogeneous_jobs.html)
that includes both node types. Licenses belong to its first component and are
retained for the allocation. An `srun` step cannot move an ordinary compute job
into a QAN partition.

## Inspect and operate the cluster

From the login shell, inspect completed jobs and allocated license resources:

```bash
sacct -X --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES%80
```

SlurmDBD records these in MariaDB under the `research` account. Database and
controller state persist in named volumes. Job scripts and environments are not
archived; applications should save software versions, input digests, and remote
job IDs alongside results, without credentials.

Scale nodes from the host:

```bash
docker compose -f examples/slurm/compose.yml up -d --wait --scale node=4 --scale qan=3
```

The cluster accepts up to 128 registered nodes. Drain nodes and wait for their
jobs before scaling down. Remove stopped dynamic nodes with
`scontrol delete NodeName=...` when retaining controller state.

`docker compose -f examples/slurm/compose.yml down` stops the cluster and keeps
accounting. To discard it, including accounting history:

```bash
docker compose -f examples/slurm/compose.yml down --volumes
rm -r build/slurm
```

Images and build caches remain. For independent clusters, choose a Compose
`--project-name`, set `MQT_CORE_SLURM_RUNTIME` to an absolute directory, and
pass that directory to `prepare.sh`.

## Deploy on an existing cluster

Provide one versioned, administrator-owned environment on login and quantum
access nodes. Expose its executables through the default `PATH` or a software
module, and maintain the catalogue in `/etc/mqt-core/qdmi.json`. Shared files
and matching numeric user/group IDs make scripts, inputs, and credential files
available across nodes. Where several environments exist, users select a module
before `sbatch` or load it in their batch script; see the
[GWDG module guide](https://docs.hpc.gwdg.de/software_stacks/module_basics/index.html).

Use matching Slurm versions across hosts; the playground uses Slurm 25.11 or
newer. Login nodes need Slurm clients, compute nodes need `slurmd`, and a
monitor host needs the device libraries and site credentials. Run monitors
outside Slurm daemons and initialize reservations before opening queues.
SlurmDBD provides persistent accounting but is not required for static local
licenses.

:::{dropdown} CPU and memory isolation with cgroup v2
For allocation-based CPU and memory limits, configure:

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

`autodetect` selects the host's cgroup implementation; cgroup v1 is deprecated.
Set resources, partitions, accounts, and limits for the site. Keep scheduler
authentication, such as Munge, separate from device credentials. See the
[Slurm administration guide](https://slurm.schedmd.com/quickstart_admin.html)
and [cgroup configuration](https://slurm.schedmd.com/cgroup.conf.html).
:::

## Integration tests

`test/slurm/run_integration.py` exercises login submission, partition placement,
license contention, unavailable-device admission and recovery, environment
export, and accounting with DDSIM and SC.

The IQM and Braket repositories install their device packages into the same
cluster and execute authenticated Emerald Resonance mock and SV1 workloads. They
supply credentials through a Compose overlay. These tests can incur service
charges.
