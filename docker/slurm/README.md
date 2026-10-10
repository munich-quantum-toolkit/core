# Local Slurm cluster

Run a small Slurm cluster for QDMI demonstrations and integration tests. It uses
one controller and as many compute containers as requested. Each compute node
offers two CPUs and 512 MiB of scheduled memory.

Use rootful Docker Compose on a disposable Linux host with cgroup v2. The
containers are privileged and share the host cgroup hierarchy so that Slurm can
enforce job CPU and memory allocations. This setup is for local use; it does not
configure a production cluster.

From the MQT Core checkout, build exactly one wheel and prepare the shared
files:

```console
uv build --wheel --out-dir dist -Ccmake.define.DEPLOY=ON
sh docker/slurm/prepare.sh
docker compose -f docker/slurm/compose.yml up --build -d --wait --scale node=2
```

Submit jobs as the unprivileged user shared by all nodes:

```console
docker compose -f docker/slurm/compose.yml exec --user 10000:10000 controller \
  srun --licenses=mqt.sc.default python3 /workspace/test/slurm/sc_job.py
docker compose -f docker/slurm/compose.yml exec controller sinfo
docker compose -f docker/slurm/compose.yml up -d --wait --scale node=4
```

Slurm registers each compute container dynamically using its unique hostname.
The default partition accepts up to 128 nodes. Scaling down stops containers;
Slurm retains their inactive node records until `scontrol delete NodeName=...`
or a fresh cluster is created. Drain nodes and wait for their jobs before
scaling down an active cluster.

The shared `/jobs` directory is `build/slurm/jobs` on the host. Edit
`build/slurm/slurm.conf` to change the license counts or Slurm configuration,
then run `scontrol reconfigure` in the controller. Stop the cluster with:

```console
docker compose -f docker/slurm/compose.yml down --volumes
rm -r build/slurm
```

The image and Docker build cache remain available. To run isolated clusters, use
a different Compose `--project-name`, set `MQT_CORE_SLURM_RUNTIME` to an
absolute directory, and pass that directory to `prepare.sh`.

The integration runner in `test/slurm/run_integration.py` uses these same images
and services. Its test-only overlay adds daemon environment sentinels; the
reusable image contains no test configuration. Run it with `--nodes 3` to check
a different cluster size. `MQT_CORE_SLURM_DIST` selects an existing directory
containing one MQT Core wheel. Device implementation tests can extend the image
with `MQT_CORE_SLURM_WORKLOAD`, `MQT_CORE_SLURM_SETUP_SCRIPT`, and the build
arguments `PROVIDER_RUNTIME_COMPONENT` and `PROVIDER_INSTALL_MODE` (`native` or
`wheel`). Set build arguments on `controller`; `node` reuses that image. Device
implementation overlays supply credentials to the submission container at
runtime. Slurm exports these settings to the workload; credentials are never
part of the image build.
