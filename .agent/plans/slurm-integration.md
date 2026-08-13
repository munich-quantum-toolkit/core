# Test and document QDMI device admission with Slurm

This ExecPlan is a living document. Maintain it in accordance with
`.agent/PLANS.md`.

## Purpose / Big Picture

Provide an end-to-end test and a concise tutorial for the mechanism-specific
QDMI Slurm adapter. A real Slurm controller admits jobs through local static
licenses whose names are stable QDMI device IDs. The test proves both scheduler
admission and QDMI execution without coupling the generic driver to Slurm. The
license environment selects a device. It does not attest the allocation or
authorize access.

The fixture uses Slurm 25.11 or newer, one controller, two compute nodes,
systemd, Munge, and cgroup v2. Each node has two CPUs. Slurm configures two
`mqt.ddsim.default` licenses and one `mqt.sc.default` license.

## Progress

- [x] (2026-08-13 04:04Z) Reconstruct the layer on the exact rewritten #2025
      head and retain only the workflow, fixture, tutorial, and changelog work.
- [x] (2026-08-13 04:12Z) Update all workloads and documentation to use
      `mqt.core.qdmi.slurm.open_device_from_license()`.
- [x] (2026-08-13 04:12Z) Add live rejection cases for AND, OR, and non-unit
      QDMI license expressions.
- [x] (2026-08-13 04:33Z) Run Python syntax, Compose configuration, changed-file
      hooks, full lint, and the Sphinx warnings-as-errors build.
- [x] (2026-08-13 12:47Z) Rebuild the layer on the final #2025 head and preserve
      the workflow, fixture, tutorial, and changelog boundary.
- [x] (2026-08-13 12:52Z) Complete an independent adversarial review and add the
      missing real-Slurm OR-expression rejection case.
- [x] (2026-08-13 14:45Z) Document the process-mutable environment trust
      boundary and separate Slurm admission, QDMI selection, provider state, and
      access control.
- [x] (2026-08-13 14:45Z) Replace staged Python modules with one wheel installed
      in the container by a pinned uv binary, and remove all `PYTHONPATH`
      overrides.
- [x] (2026-08-13 14:45Z) Remove the local `slurmd` unit override and require
      the packaged unit to report `Delegate=yes` at runtime.
- [x] (2026-08-13 14:53Z) Pass Python and shell syntax, Compose configuration,
      changed-file and full lint, the Sphinx warnings-as-errors build, and a
      local Python 3.14 wheel build.

## Surprises & Discoveries

- Observation: Parser unit tests are not sufficient for Slurm license syntax.
  The orchestrator submits non-unit, AND, and OR forms and checks the adapter
  diagnostics and absence of a DDSIM result.
- Observation: Current Slurm documentation continues to define licenses as
  cluster-wide shared resources and recommends cgroup v2 with systemd. The
  fixture uses the packaged systemd units and checks `Delegate=yes` for
  `slurmd`.
- Observation: `SLURM_JOB_LICENSES` is ordinary process environment. It is
  suitable for selection in a cooperative job, but it cannot attest an
  allocation or authorize device access.

## Decision Log

- Decision: Keep the existing three-container topology. Rationale: Separate
  compute-node hostnames prove cross-node scheduling and avoid Slurm
  multi-daemon emulation. Date/Author: 2026-08-13, Lukas Burgholzer and Codex.
- Decision: Request exactly one device license in every valid job. Rationale: A
  configured count of two is an admission pool for independent jobs, not two
  device handles for one job. Date/Author: 2026-08-13, Lukas Burgholzer and
  Codex.
- Decision: Exercise invalid expressions through real `sbatch` jobs. Rationale:
  Unit tests prove parser grammar, while this layer must prove the actual
  `SLURM_JOB_LICENSES` value and batch-job failure. Date/Author: 2026-08-13,
  Lukas Burgholzer and Codex.
- Decision: Keep credentials outside the adapter and fixture. Rationale: IQM,
  Amazon Braket, and future providers own different credential sources. The
  tutorial documents this boundary. Date/Author: 2026-08-13, Lukas Burgholzer
  and Codex.
- Decision: Keep the adapter as a selector and place authorization at the real
  resource boundary. Rationale: local devices can use GRES device-file cgroup
  enforcement, while remote services must authorize provider operations. A
  different Slurm lookup would not prevent direct QDMI device opening.
  Date/Author: 2026-08-13, Lukas Burgholzer and Codex.
- Decision: Install the built wheel in each image. Rationale: the fixture must
  test the distributable package without source-tree or `PYTHONPATH` effects.
  Date/Author: 2026-08-13, Lukas Burgholzer and Codex.

## Outcomes & Retrospective

The implementation now states the selector trust boundary and uses the packaged
systemd units. The workflow builds one wheel, installs it in the container with
uv, and runs without `PYTHONPATH`. Local syntax, Compose configuration, lint,
documentation, and a Python 3.14 macOS wheel build pass. The tutorial has 1,177
words. The local Docker daemon was unavailable, so the Linux wheel installation,
the packaged `Delegate=yes` assertion, and the complete privileged Slurm fixture
remain replacement-CI gates. The independent review must run on the amended
exact head.

## Context and Orientation

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

## Plan of Work

Build the current branch wheel inside the Docker build context. Copy a pinned
official uv binary into the image and install the wheel into the system Python.
Start the controller and compute nodes under systemd. Require the unified cgroup
v2 hierarchy, the configured Slurm version, and `Delegate=yes` from the packaged
`slurmd` unit.

First submit a job that requests two DDSIM licenses, one job that requests one
DDSIM plus one SC license, and one job that requests either license. All three
allocations can start, but the QDMI adapter must reject their
`SLURM_JOB_LICENSES` expressions before a device job is created. Check the exact
diagnostic and verify that Slurm returns the licenses.

Next pin two DDSIM jobs to different nodes. Each job requests one CPU and one
DDSIM license, executes its Bell circuit, writes its result, and retains its
allocation. Submit a third DDSIM job. It must remain pending for `Licenses`
while each node has one free CPU. Submit an SC job and require it to finish on a
free CPU. Release the first DDSIM job and require the third job to execute.
Validate all three Bell results, then release the final held job.

Run Python compilation, YAML and Compose validation, full repository lint,
documentation, spelling, and link checking locally where available. The
privileged Ubuntu 26.04 workflow is the authoritative real-Slurm gate.

## Validation and Acceptance

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

## Idempotence and Recovery

The orchestrator uses one fixed Compose project and always runs
`compose down --volumes --remove-orphans`. It removes only ignored result files
below `test/slurm/runtime`, generates a new Munge key for each run, and prints
daemon, node, queue, license, job, and batch-output diagnostics on failure. The
wheel is stored separately below the ignored `test/slurm/dist` directory.

## Interfaces and Dependencies

The workload uses:

    from mqt.core.qdmi import ProgramFormat, slurm

    device = slurm.open_device_from_license()
    job = device.submit_job(program, ProgramFormat.QASM2, 256)

The fixture adds no production dependency. Its system dependencies are Ubuntu
26.04, Slurm 25.11 or newer, systemd, Munge, Docker Compose, and a cgroup v2
host.
