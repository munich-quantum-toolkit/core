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
| **HRES**           | Set device capacity, eligible nodes, and availability.   |
| **Credentials**    | Authenticate with the device service.                    |

The playground uses Slurm hierarchical resources (HRES), named after QDMI device
IDs. HRES controls placement and concurrent allocations. Its Lua plugin queries
QDMI availability before admitting jobs. Applications own their QDMI sessions
and credentials; HRES does not reserve capacity at a remote service. The
existing `--licenses` submission syntax and `license/ID` accounting apply.

:::{important}
This prototype builds Slurm 26.11.0rc1 development revision
`ba178ef837c4ba799fb3de72bcdaf30bc2b1616d` with `hres/lua`. Its callbacks and
configuration may change before release. The two QANs have explicit node records
and HRES membership; no Slurm source patches are needed. See the
[pinned HRES documentation](https://github.com/SchedMD/slurm/blob/ba178ef837c4ba799fb3de72bcdaf30bc2b1616d/doc/html/hres.shtml).
:::

Already have a cluster? Start with [job submission](#submit-your-first-job) or
[site configuration](#deploy-on-an-existing-cluster). Otherwise, the
[playground sources](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
provide a cluster to experiment with:

| Host                     | Role                                                             |
| ------------------------ | ---------------------------------------------------------------- |
| `login`                  | User shell, Slurm clients, and shared software environment       |
| `controller`             | Scheduling and HRES availability callbacks                       |
| `node` (two by default)  | Classical computation in the `compute` partition                 |
| `qan1`, `qan2`           | Compilation and access to all devices in the `quantum` partition |
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

Each systemd container consumes host inotify instances. If startup reports
`Failed to create control group inotify object: Too many open files`, stop
unused playground projects or ask the host administrator to raise
`fs.inotify.max_user_instances`. Running multiple full clusters can exhaust a
host limit of 128 before any Slurm job starts.

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

Configure each device in `build/slurm/resources.yaml`. For example:

```yaml
- resource: iqm.emerald.mock
  mode: MODE_1
  plugin:
    type: hres/lua
    config:
      - name: path
        value: /etc/slurm/hres_qdmi.lua
    maintenance_cycle: 30
    maintenance_timeout: 15
  layers:
    - layer_name: quantum
      nodes: ["qan[1-2]"]
      count: 1
```

Use one explicit block per device; the pinned Slurm YAML parser does not support
anchors. List only the nodes that can access the device in its layer. Both
playground QANs can access every configured device; classical nodes belong to
none of these layers. Extend the accounting list in `build/slurm/slurm.conf`:

```ini
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
device requires its implementation, catalogue entry, HRES block, and
credentials; it does not require a new partition or node type. Classical nodes
use an internal network; QANs, the login node, and the HRES controller also have
outbound connectivity.

### Supply credentials

The controller probe uses site credentials independently of jobs. Put their
references in the private `build/slurm/qdmi.env` file, using systemd
`EnvironmentFile` syntax:

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

An HRES count limits concurrent Slurm allocations, not submitted circuits.
Choose it according to site policy. The Lua callback starts with admission
closed, then probes the selected QDMI device in a separate process:

| Probe result                           | Scheduling behavior                              |
| -------------------------------------- | ------------------------------------------------ |
| `IDLE` or `BUSY`                       | Enable the layer for new allocations             |
| Initialization/status error or timeout | Disable the layer; jobs wait with `HRESDisabled` |
| Capacity exhausted                     | Keep jobs pending without allocating nodes       |

The default probe runs every 30 seconds and has a ten-second deadline, followed
by forced termination after two seconds. Slurm's `maintenance_timeout` only
reports slow callbacks; the separate process deadline enforces the bound.
Startup and reconfiguration require a new successful probe. Running jobs keep
their allocations when a device becomes unavailable; unrelated devices remain
usable. An application must handle execution failure and remote queuing.

`BUSY` means operational, not immediately available. Admission reflects the
probe identity and the last successful observation, not a user's authorization.
Secrets belong in the private environment/credential files, never in HRES
`config` or callback payloads, which Slurm can pass to job users. Probe output
and core dumps are disabled because device diagnostics can contain credentials.

Inspect `DisableLayer`, `Nodes`, `Used`, and `Free` in `scontrol show licenses`.
From an administrator shell on the controller, a manual stop is:

```bash
scontrol update HRESName=mqt.ddsim.default DisableHRES=true
scontrol update HRESName=mqt.ddsim.default DisableHRES=false
```

This administrative flag is independent of the probe-controlled layer flag. For
an automatic outage/recovery test, temporarily disable the device in the shared
QDMI catalogue, then restore it; the integration test exercises this path.

## Chain quantum and classical stages

Submit separate jobs for classical preparation, quantum sampling, and classical
analysis. Each stage releases its allocation before the next starts; only the
quantum stage requests an HRES unit. For example, with application scripts
`prepare.sh`, `measure.sh`, and `analyze.sh`:

```bash
preparation=$(sbatch --parsable --partition=compute prepare.sh)
measurement=$(sbatch --parsable --partition=quantum --licenses=iqm.emerald.mock \
  --dependency=afterok:"$preparation" measure.sh)
sbatch --partition=compute --dependency=afterok:"$measurement" analyze.sh
```

Stages exchange files through `/jobs`; `afterok` requires the predecessor to
succeed. The quantum application compiles for the same open QDMI device used for
submission. Separate jobs avoid holding classical resources during compilation
or remote queuing. A
[heterogeneous allocation](https://slurm.schedmd.com/heterogeneous_jobs.html)
acquires its components together and can suit tightly coupled applications, but
retains resources across these stages. An `srun` step cannot move an ordinary
compute job into the quantum partition.

## Inspect and operate the cluster

From the login shell, inspect completed jobs and allocated license resources:

```bash
sacct -X --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES%80
```

SlurmDBD records these in MariaDB under the `research` account. Database and
controller state persist in named volumes. Job scripts and environments are not
archived; applications should save software versions, input digests, and remote
job IDs alongside results, without credentials.

Scale classical nodes from the host:

```bash
docker compose -f examples/slurm/compose.yml up -d --wait --scale node=4
```

The two QANs are fixed. To add a QAN, define its Compose service and hostname,
then add it to `slurm.conf` and the applicable resource layers. This keeps HRES
membership declarative across controller restarts and reconfiguration.

The cluster accepts up to 128 nodes. Drain classical nodes and wait for their
jobs before scaling down. Remove stopped dynamic nodes with
`scontrol delete NodeName=...` when retaining controller state. Run
`scontrol reconfigure` after changing the shared Slurm configuration.

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

Use the pinned Slurm build on every host. Login nodes need its clients; compute
nodes need `slurmd`; the controller needs Lua, the QDMI libraries, and probe
credentials. Install the Lua script and probe at their configured paths. Replace
the playground node names with the site's eligible nodes. Review the chosen
Slurm revision before deployment; this development build is not a production
deployment recommendation.

SlurmDBD records resource allocations; it does not measure remote device time or
billing. A QAN allocation may still wait in a provider queue. Remote reservation
or QRMI integration would require a separate provider/site contract; this
prototype provides local admission only.

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
HRES capacity and node eligibility, unavailable-device admission and recovery,
environment export, and accounting with DDSIM and SC.

`test/slurm/check_hres.lua` checks callback state transitions and the bounded
probe command with Lua 5.4. CI runs both checks. Live provider tests are
separate and require explicit credentials and authorization.
