# A playground for hybrid quantum-classical workflows

Use this cluster to prototype scientific workflows that combine classical
computation with QDMI devices. Submit jobs from a login node, place quantum
stages on dedicated {term}`quantum access nodes <quantum access node>` (QANs),
and inspect their resource use and outcomes through Slurm accounting. The same
software environment supports DDSIM, IQM, and Amazon Braket.

The
[cluster sources](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
provide these hosts:

| Host                     | Role                                                       |
| ------------------------ | ---------------------------------------------------------- |
| `login`                  | User shell, Slurm clients, and shared software environment |
| `controller`             | Scheduling and device availability monitors                |
| `node` (two by default)  | Classical computation in the default `compute` partition   |
| `qan-iqm`, `qan-braket`  | Quantum access through the `iqm` and `braket` partitions   |
| `accounting`, `database` | Slurm accounting and persistent MariaDB storage            |

Each compute node and QAN supplies two CPUs and 2 GiB of scheduled memory. Jobs
run as `mqt-test` (UID/GID 10000), with cgroup v2 enforcing allocations. One
Python 3.15 environment at `/opt/venv` contains MQT Core and the installed
device implementations. The administrator maintains the catalogue; users supply
their credentials. See {doc}`slurm` for deployment on an existing cluster.

## Start the cluster and log in

Docker Compose supplies the hosts. Use rootful Docker on a disposable Linux host
with cgroup v2: the containers run systemd with privileged access to the host
cgroup hierarchy. This is a development playground, not a security boundary for
untrusted users.

From an MQT Core checkout with the {doc}`build prerequisites <../installation>`
installed:

```console
uv build --python 3.15 --wheel --out-dir dist -Ccmake.define.DEPLOY=ON
sh examples/slurm/prepare.sh
docker compose -f examples/slurm/compose.yml up --build -d --wait --scale node=2
docker compose -f examples/slurm/compose.yml exec --user 10000:10000 login bash --login
```

`dist` must contain exactly one MQT Core wheel. The cluster installs the wheels
into its common environment and puts its executables on `PATH`; jobs need no
activation or package installation. The following commands run in the login
shell:

```console
sinfo
scontrol show licenses
sbatch --wait --licenses=mqt.ddsim.default --time=00:05:00 \
  --output=/jobs/slurm-%j.out \
  --wrap='srun python /workspace/test/slurm/bell_job.py'
sacct -X --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES%80
```

This executes a Bell circuit through DDSIM in the `compute` partition. Results
appear in `/jobs`, shared by all nodes and backed by `build/slurm/jobs` on the
host. Put application inputs, scripts, and outputs there, or add a shared
application mount. The MQT Core checkout is available read-only at `/workspace`.

(multiple-device-implementations)=

## Connect quantum devices

Build compatible Python 3.15 device wheels and place them alongside MQT Core in
`dist` before building the cluster image:

```text
dist/
  mqt_core-....whl
  iqm_qdmi-....whl
  amazon_braket_qdmi-....whl
```

Subdirectories are accepted. Use the device implementations' current `main`
branches while QDMI 1.4 and MQT Core 4.1 are unreleased. All implementations and
Qiskit adapters share `/opt/venv`; MQT Core's PennyLane adapter is included too.
Installed wheels supply device manifests. Targeted driver sessions open only
their selected device, so sessions for several implementations can coexist.

Before startup, configure concurrency limits and their accounting records in
`build/slurm/slurm.conf`:

```ini
Licenses=mqt.ddsim.default:2,mqt.sc.default:1,iqm.emerald.mock:1,amazon.braket.sv1:2
AccountingStorageTRES=license/mqt.ddsim.default,license/mqt.sc.default,license/iqm.emerald.mock,license/amazon.braket.sv1
```

Put shared endpoints and device presets in `build/slurm/qdmi.json`, mounted at
`/etc/mqt-core/qdmi.json`. These settings extend installed manifests; see
{doc}`configuration`. Keep credentials outside this catalogue and the image.

Monitors use site credentials, independently of job credentials. Put their
references in the private `build/slurm/qdmi.env` file, using systemd
`EnvironmentFile` syntax, for example:

```ini
IQM_TOKENS_FILE=/run/credentials/iqm.json
AWS_SHARED_CREDENTIALS_FILE=/run/credentials/aws
AWS_PROFILE=site-monitor
```

Mount those credential files read-only into the controller through a Compose
overlay. Users authenticate on the login node through each implementation's
usual mechanism. Slurm exports their environment to jobs; credential files must
also be readable by the user on the relevant QAN. Environment variables set for
a daemon do not configure user jobs.

Submit IQM jobs with `--partition=iqm --licenses=iqm.emerald.mock` and Braket
jobs with `--partition=braket --licenses=amazon.braket.sv1`. Classical nodes use
an internal network; QANs, the login node, and the monitoring controller also
have outbound connectivity. Partitions route jobs to the appropriate nodes,
while licenses limit concurrency across the cluster. Neither restricts which
remote API a process may call: apply site egress policies and device
authorization when access must be restricted to a particular QAN.

## Chain quantum and classical stages

A quantum-assisted AFQMC application can measure a trial state on a QAN, save
its results to `/jobs`, and propagate classical walkers on compute nodes. From
the login shell, submit the application's two batch scripts:

```console
measurement=$(sbatch --parsable --partition=iqm --licenses=iqm.emerald.mock measure.sh)
sbatch --partition=compute --dependency=afterok:"$measurement" propagate.sh
```

The scripts choose their CPU and memory requirements and agree on an output
path. The classical stage starts only after successful measurement and holds no
quantum license. Application code, scientific validation, and additional
scientific dependencies belong with the application.

For tightly coupled iterations, use a
[heterogeneous allocation](https://slurm.schedmd.com/heterogeneous_jobs.html)
that includes both node types. Slurm permits licenses on its first component and
retains them for the allocation. An `srun` step stays within its allocation; it
cannot move an ordinary compute job into a QAN partition.

## Inspect and operate the cluster

`slurmdbd` records jobs in MariaDB, including user, account, partition, state,
exit code, elapsed time, and allocated license TRES. The `research` account and
`mqt-test` association are created at startup. Database and controller state
persist in named volumes. `sacct` can therefore inspect completed jobs after
service restarts. Job environments and scripts are not archived. Applications
should save software versions, input digests, and remote job IDs alongside
results, without credentials.

The controller reserves every configured device license before opening the
partitions. Each `qdmi-availability@ID:COUNT.service` checks its device every 30
seconds, with a ten-second probe timeout, and releases the reservation after
success. Stopping a monitor closes its license; failed checks keep new jobs
pending without interrupting running work. From an administrator shell on the
controller:

```console
systemctl stop qdmi-availability@mqt.ddsim.default:2.service
systemctl start qdmi-availability@mqt.ddsim.default:2.service
```

Enter that shell from the host with
`docker compose -f examples/slurm/compose.yml exec controller bash`. The monitor
maintains the `qdmi-unavailable-ID` reservation; do not create a conflicting
reservation. Inspect both `Free` and `Reserved` in `scontrol show licenses`,
because reserved tokens can still appear in `Free`. A successful check is a
snapshot for the monitor's identity, not a promise of user authorization or
prompt execution at the remote service.

Scale classical computation from the host:

```console
docker compose -f examples/slurm/compose.yml up -d --wait --scale node=4
```

The cluster accepts up to 128 registered nodes. Drain nodes and wait for their
jobs before scaling down. Slurm retains stopped dynamic nodes until they are
removed with `scontrol delete NodeName=...` or the cluster is reset.

`docker compose -f examples/slurm/compose.yml down` stops the cluster while
preserving accounting. To discard the cluster, including its accounting history:

```console
docker compose -f examples/slurm/compose.yml down --volumes
rm -r build/slurm
```

Images and build caches remain. For independent clusters, choose a Compose
`--project-name`, set `MQT_CORE_SLURM_RUNTIME` to an absolute directory, and
pass that directory to `prepare.sh`.

## Integration tests

`test/slurm/run_integration.py` tests login submission, partition placement,
license contention, unavailable-device admission and recovery, job environment
export, and accounting using the bundled DDSIM and SC implementations.

The IQM and Braket repositories run their own authenticated simulator workloads
on their QANs, in native and wheel installation modes. They supply credentials
and an optional native build through a Compose overlay and setup script. Native
libraries are compiled independently of the MQT Core image layer, and Python
adapters share the common environment in either mode. IQM uses the Emerald
Resonance mock and Braket uses SV1; these tests can incur service charges.
