# Test and document QDMI device admission with Slurm

Status: historical implementation record.

## Goal and scope

Provide an end-to-end test and a concise tutorial for the mechanism-specific
QDMI Slurm adapter. A real Slurm controller admits jobs through local static
licenses whose names are stable QDMI device IDs. The test proves both scheduler
admission and QDMI execution without coupling the generic driver to Slurm. The
license environment selects a device. It does not attest the allocation or
authorize access.

The fixture uses Slurm 25.11 or newer, one controller, two compute nodes,
systemd, Munge, and cgroup v2. Each node has two CPUs. Slurm configures two
`mqt.ddsim.default` licenses and one `mqt.sc.default` license.

## Constraints

- Parser unit tests are not sufficient for Slurm license syntax. The
  orchestrator submits non-unit, AND, and OR forms and checks the adapter
  diagnostics and absence of a DDSIM result.

- Current Slurm documentation continues to define licenses as cluster-wide
  shared resources and recommends cgroup v2 with systemd. The fixture uses the
  packaged systemd units and checks `Delegate=yes` for `slurmd`.

- `SLURM_JOB_LICENSES` is ordinary process environment. It is suitable for
  selection in a cooperative job, but it cannot attest an allocation or
  authorize device access.

## Decisions

- Keep the existing three-container topology. Rationale: Separate compute-node
  hostnames prove cross-node scheduling and avoid Slurm multi-daemon emulation.

- Request exactly one device license in every valid job. Rationale: A configured
  count of two is an admission pool for independent jobs, not two device handles
  for one job.

- Exercise invalid expressions through real `sbatch` jobs. Rationale: Unit tests
  prove parser grammar, while this layer must prove the actual
  `SLURM_JOB_LICENSES` value and batch-job failure.

- Keep credentials outside the adapter and fixture. Rationale: IQM, Amazon
  Braket, and future providers own different credential sources. The tutorial
  documents this boundary.

- Keep the adapter as a selector and place authorization at the real resource
  boundary. Rationale: local devices can use GRES device-file cgroup
  enforcement, while remote services must authorize provider operations. A
  different Slurm lookup would not prevent direct QDMI device opening.

- Install the built wheel in each image. Rationale: the fixture must test the
  distributable package without source-tree or `PYTHONPATH` effects.

## Outcome and validation

The integration uses packaged systemd units and installs a built wheel without
`PYTHONPATH`. Syntax, Compose configuration, lint, documentation, and a macOS
wheel build passed. Linux installation, the packaged `Delegate=yes` assertion,
and the privileged Slurm fixture were not executed in the recorded local
validation.

## Code and ownership

`.github/workflows/slurm.yml` builds the current wheel and runs the host harness
with uv. The Docker build installs that wheel into each image.
`test/slurm/compose.yml` defines one controller and two compute nodes. The
systemd preparation unit installs one generated Munge key and creates the
required spool directories. `slurm.conf` selects `select/cons_tres` and the
cgroup process, task, and accounting plugins. `cgroup.conf` selects the cgroup
implementation automatically and constrains cores and memory.

`test/slurm/run_integration.py` is the single orchestrator. It validates cgroup
v2 and Slurm 25.11+, waits for both nodes, checks the persistent QDMI IDs and
license counters, runs negative adapter jobs, and then exercises the complete
admission scenario. The DDSIM workload executes a 256-shot Bell circuit. The SC
workload proves that a free CPU remains usable while DDSIM admission is full.

`docs/qdmi/slurm.md` is an administrator and user tutorial. It states that a
Slurm license is cluster-wide admission. It does not represent provider
availability or a provider queue. Provider credentials remain provider-owned.

## Acceptance

The test accepts only Slurm 25.11 or newer and cgroup v2. Both compute nodes
must report `Delegate=yes`, two CPUs, and an idle state. The controller must
import the installed wheel from outside the source and runtime mounts. Initial
license records must report DDSIM total two and SC total one.

Invalid jobs must fail with the adapter diagnostic and produce no device result.
The two held DDSIM jobs must run on different nodes. The third must be pending
with reason `Licenses` while both nodes report one allocated CPU. The SC job
must finish before either held allocation ends. Releasing one DDSIM job must
allow the third job to run without reconfiguration.

Each DDSIM result must contain exactly 256 samples and only outcomes `00` and
`11`. Documentation must remain between 800 and 1,200 words and pass the Sphinx
warnings-as-errors build.

## Interfaces

The workload uses:

    from mqt.core.qdmi import ProgramFormat, slurm

    device = slurm.open_device_from_license()
    job = device.submit_job(program, ProgramFormat.QASM2, 256)

The fixture adds no production dependency. Its system dependencies are Ubuntu
26.04, Slurm 25.11 or newer, systemd, Munge, Docker Compose, and a cgroup v2
host.
