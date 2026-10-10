# Run the example Slurm cluster

The
[cluster sources](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
provide a Slurm controller and a configurable number of compute nodes for QDMI
workloads. Each compute node has two CPUs and 2 GiB of scheduled memory. Jobs
run as `mqt-test` (UID/GID 10000), and cgroup v2 enforces their allocations.

One Python 3.15 environment at `/opt/venv` contains MQT Core and the installed
device implementations. The administrator maintains the shared catalogue; users
submit jobs with their own credentials. Availability monitors keep jobs pending
while a requested device is unavailable. See {doc}`slurm` for these interfaces
and their use on an existing cluster.

## Start the cluster

Docker Compose supplies the hosts. Use rootful Docker on a disposable Linux host
with cgroup v2. The containers run systemd with privileged access to the host
cgroup hierarchy.

From an MQT Core checkout with the {doc}`build prerequisites <../installation>`
installed, build a Python 3.15 wheel and prepare the cluster:

```console
uv build --python 3.15 --wheel --out-dir dist -Ccmake.define.DEPLOY=ON
sh examples/slurm/prepare.sh
docker compose -f examples/slurm/compose.yml up --build -d --wait --scale node=2
```

`dist` must contain exactly one MQT Core wheel. The image uses uv 0.13.0 to
install all wheels into the shared environment, which is on every node's `PATH`.
No environment activation or package installation is needed in jobs.

The controller starts with the compute partition closed. It reserves every
configured device license, starts the monitors, then opens the partition. Each
monitor releases its reservation after a successful check. The cluster can
therefore start while a device is unavailable.

Run the DDSIM workload and inspect the cluster:

```console
docker compose -f examples/slurm/compose.yml exec --user 10000:10000 controller \
  sbatch --wait --licenses=mqt.ddsim.default --time=00:05:00 \
  --output=/jobs/slurm-%j.out \
  --wrap='srun python /workspace/test/slurm/bell_job.py'
docker compose -f examples/slurm/compose.yml exec controller sinfo
docker compose -f examples/slurm/compose.yml exec controller scontrol show licenses
```

Job results are written to `/jobs`, mounted from `build/slurm/jobs` on the host.
The integration workload checks its Bell measurement counts and writes a JSON
result. For an application of your own, use the batch script and Python example
in {doc}`slurm`.

## Multiple device implementations

Build compatible Python 3.15 wheels from the current MQT Core, QDMI-on-IQM, and
Amazon-Braket-QDMI `main` branches. Put them together in `dist` before building
the cluster image:

```text
dist/
  mqt_core-....whl
  iqm_qdmi-....whl
  amazon_braket_qdmi-....whl
```

Subdirectories under `dist` are also accepted. All device implementations and
their Qiskit adapters share `/opt/venv`; MQT Core's PennyLane adapter is
included too. Installed wheels supply device manifests. Each targeted driver
session opens only its selected device, and sessions for several devices can
coexist in one process.

Before starting the cluster, add concurrency limits to `build/slurm/slurm.conf`,
for example:

```ini
Licenses=mqt.ddsim.default:2,iqm.emerald.mock:1,amazon.braket.sv1:2
```

Put shared device settings in `build/slurm/qdmi.json`, mounted read-only at
`/etc/mqt-core/qdmi.json` on all nodes. Its definitions extend the installed
manifests. Configure endpoints and device presets here; keep user and monitor
credentials separate. See {doc}`configuration` and each device implementation's
documentation for its settings.

Place monitor credential references in `build/slurm/qdmi.env`, using systemd
`EnvironmentFile` syntax, for example:

```ini
IQM_TOKENS_FILE=/run/credentials/iqm.json
AWS_SHARED_CREDENTIALS_FILE=/run/credentials/aws
AWS_PROFILE=site-monitor
```

Mount those files read-only into the controller with a Compose overlay. Keep
`qdmi.env` and the credential files private to the administrator. They are used
only by monitors; users authenticate independently before submission. Credential
files used by jobs must be readable by their users on compute nodes. Credentials
stay outside the image and shared catalogue.

## Operate the cluster

The partition accepts up to 128 nodes. Add nodes with:

```console
docker compose -f examples/slurm/compose.yml up -d --wait --scale node=4
```

Drain nodes and wait for their jobs before scaling down. Slurm retains stopped
nodes until they are removed with `scontrol delete NodeName=...` or the cluster
is recreated.

Each `qdmi-availability@ID:COUNT.service` checks its device every 30 seconds,
with a ten-second probe timeout. Systemd restarts failed monitors and cleans up
their child processes. Stopping a monitor closes its license until the monitor
is restarted. On the controller, for example:

```console
systemctl stop qdmi-availability@mqt.ddsim.default:2.service
systemctl start qdmi-availability@mqt.ddsim.default:2.service
```

Use `docker compose -f examples/slurm/compose.yml exec controller` to run these
commands from the host. The monitor owns the `qdmi-unavailable-ID` reservation;
do not create conflicting reservations. Inspect both `Free` and `Reserved` in
`scontrol show licenses`, because reserved tokens can still appear in `Free`.

Stop the cluster and remove its runtime files with:

```console
docker compose -f examples/slurm/compose.yml down --volumes
rm -r build/slurm
```

Images and the build cache remain. For independent clusters, choose a Compose
`--project-name`, set `MQT_CORE_SLURM_RUNTIME` to an absolute directory, and
pass that directory to `prepare.sh`.

## Integration tests

`test/slurm/run_integration.py` uses the cluster with a test overlay. Device
repositories supply workloads, credentials, and an optional native installation
through `MQT_CORE_SLURM_WORKLOAD`, `MQT_CORE_SLURM_SETUP_SCRIPT`,
`PROVIDER_RUNTIME_COMPONENT`, and `PROVIDER_INSTALL_MODE` (`native` or `wheel`).
Their Python adapters use the same environment in either mode. Device libraries
are built independently so an MQT Core wheel change can reuse that compilation.

`test/slurm/multivendor.py` submits DDSIM, IQM Emerald Resonance mock, and
Braket SV1 workloads concurrently in one process and checks that an IQM outage
leaves the other devices usable. These tests make small simulator requests and
can incur service charges; they do not submit to quantum hardware.
