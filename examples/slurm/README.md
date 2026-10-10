# A Slurm cluster for QDMI workloads

This example runs quantum workloads through MQT Core's QDMI driver on a Slurm
cluster. One shared Python 3.15 environment contains MQT Core and the device
implementations. Jobs can use different devices concurrently, with separate
licenses and credentials. An availability monitor keeps new jobs pending while
their device is unavailable.

The cluster has one controller and a configurable number of compute nodes. Each
node provides two CPUs and 2 GiB of scheduled memory. Slurm enforces allocations
with cgroup v2; jobs run as the unprivileged user `mqt-test` (UID/GID 10000).
The [Slurm guide](../../docs/qdmi/slurm.md) explains the scheduling and QDMI
interfaces used here.

## Start the cluster

Docker Compose supplies the hosts for this example. Use rootful Docker on a
disposable Linux host with cgroup v2. The containers run systemd and need
privileged access to the host cgroup hierarchy.

From an MQT Core checkout, build a Python 3.15 wheel and prepare the cluster:

```console
uv build --python 3.15 --wheel --out-dir dist -Ccmake.define.DEPLOY=ON
sh examples/slurm/prepare.sh
docker compose -f examples/slurm/compose.yml up --build -d --wait --scale node=2
```

`dist` must contain exactly one MQT Core wheel. The image uses uv 0.13.0 to
install the wheels into `/opt/venv`, which is on every node's `PATH`.

The controller starts with its compute partition closed. It reserves every
configured device license, starts the availability monitors, then opens the
partition. Each monitor releases its reservation after a successful probe. This
ordering also allows the cluster to start while a device is unavailable.

Submit a job and inspect the cluster:

```console
docker compose -f examples/slurm/compose.yml exec --user 10000:10000 controller \
  srun --licenses=mqt.sc.default python3 /workspace/test/slurm/sc_job.py
docker compose -f examples/slurm/compose.yml exec controller sinfo
docker compose -f examples/slurm/compose.yml exec controller scontrol show licenses
```

The shared `/jobs` directory is `build/slurm/jobs` on the host. Slurm registers
each compute node under its unique hostname. Add nodes with:

```console
docker compose -f examples/slurm/compose.yml up -d --wait --scale node=4
```

The partition accepts up to 128 nodes. Drain nodes and wait for their jobs
before scaling down. Slurm retains stopped nodes until they are removed with
`scontrol delete NodeName=...` or the cluster is recreated.

## Multiple device implementations

Build compatible wheels from the current MQT Core, QDMI-on-IQM, and
Amazon-Braket-QDMI `main` branches. Put them together in `dist`:

```text
dist/
  mqt_core-....whl
  iqm_qdmi-....whl
  amazon_braket_qdmi-....whl
```

All wheels share `/opt/venv`; subdirectories under `dist` are also accepted. The
Qiskit and PennyLane adapters are installed with MQT Core. The driver discovers
the installed device catalogues, and each targeted session initializes only its
selected device. Sessions for IQM and Braket can therefore coexist in one
process without initializing unrelated devices.

Before starting the cluster, add the device license counts to
`build/slurm/slurm.conf`, for example:

```ini
Licenses=mqt.ddsim.default:2,mqt.sc.default:1,iqm.emerald.mock:1,amazon.braket.sv1:2
```

Place the site's shared configuration in `build/slurm/qdmi.env`, using systemd
`EnvironmentFile` syntax. The file is read only by the administrative monitors.
For example:

```ini
MQT_CORE_QDMI_CONFIG_FILE=/jobs/catalogue.json
IQM_TOKENS_FILE=/run/credentials/iqm.json
AWS_SHARED_CREDENTIALS_FILE=/run/credentials/aws
AWS_PROFILE=site-monitor
```

Mount these credential files read-only into the controller, and keep `qdmi.env`
private to its administrator. Credentials stay outside the image. The
integration overlays can also pass IQM and AWS credential variables into the
controller at runtime. Configure each device's session parameters in the shared
[catalogue](../../docs/qdmi/configuration.md).

Users submit jobs with their own credentials. Slurm exports the submission
environment to jobs; a batch script can also select credentials before its
`srun` steps. IQM and Braket settings use distinct variable names and can be
present together. A successful site probe does not authorize an individual user.

## Device availability

`qdmi-availability@ID:COUNT.service` probes each device every 30 seconds. It
reserves the device's full license count before probing and removes the
reservation only when the device reports `IDLE` or `BUSY`. Failed or timed-out
probes leave jobs pending with reason `Licenses`. Running jobs and other device
licenses remain unaffected.

The monitor runs outside Slurm daemons. Each check has a ten-second timeout;
systemd restarts failed monitors and cleans up their child processes. Stopping a
monitor closes its license until the service is restarted:

```console
systemctl stop qdmi-availability@iqm.emerald.mock:1.service
systemctl start qdmi-availability@iqm.emerald.mock:1.service
```

Run these commands on the controller, for example through `docker compose exec`.
The monitor owns reservations named `qdmi-unavailable-ID`; do not create
conflicting reservations. Inspect both `Free` and `Reserved` in
`scontrol show licenses`: reserved tokens can still appear in `Free`.

Availability is a snapshot. Jobs must handle failures after allocation, and
administrators must investigate controller communication errors because an
unreachable controller cannot receive reservation updates. Monitor credentials
must represent site access; a user's expired token must not block everyone.

## Integration tests and cleanup

`test/slurm/run_integration.py` uses this cluster with a test overlay. Device
implementation repositories supply their workload, credentials, and optional
native-library installation through `MQT_CORE_SLURM_WORKLOAD`,
`MQT_CORE_SLURM_SETUP_SCRIPT`, `PROVIDER_RUNTIME_COMPONENT`, and
`PROVIDER_INSTALL_MODE` (`native` or `wheel`). Their Python adapters use the
same environment in either mode. The independent device build stage lets MQT
Core wheel changes reuse compiled device libraries.

`test/slurm/multivendor.py` runs the IQM Emerald Resonance mock and Braket SV1
workloads together, and checks that an IQM outage leaves Braket usable. These
smoke tests make small, paid simulator requests; they do not use quantum
hardware.

Stop the cluster and remove its runtime files with:

```console
docker compose -f examples/slurm/compose.yml down --volumes
rm -r build/slurm
```

Images and the build cache remain available. For independent clusters, select a
Compose `--project-name`, set `MQT_CORE_SLURM_RUNTIME` to an absolute directory,
and pass that directory to `prepare.sh`.
