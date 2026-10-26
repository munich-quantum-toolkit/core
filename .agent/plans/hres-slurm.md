# HRES Slurm playground and hybrid workflow

Status: implemented and locally validated; draft review remains pending.

## Goal and scope

Replace the playground's local-license admission monitors with Slurm HRES and
Lua. Demonstrate preparation, target compilation and QDMI sampling, and
classical QC-AFQMC propagation as dependent Slurm jobs. Exercise the real
QDMI-on-IQM and Amazon Braket QDMI libraries with local services that simulate
the submitted circuits. Live service configuration is documented separately.

Affected entry points are `examples/slurm/`, `test/slurm/`,
`test/python/test_slurm_integration.py`, `.github/workflows/slurm.yml`, and
`docs/qdmi/slurm.md`. A separate presentations change will illustrate the
validated integration without publishing private presentation assets in Core.

## Decisions

- HRES becomes the default; remove the replaced license-reservation monitor
  after checking equivalent admission, recovery, accounting, and isolation.
- HRES owns cluster capacity, placement, and availability. Application-owned
  QDMI sessions use user credentials; compilation stays above QDMI and targets
  the same selected device used for execution. No remote capacity reservation is
  implied by a Slurm allocation or an operational device status.
- Dependent jobs release classical and quantum allocations between stages.
  Heterogeneous allocations would acquire stage resources together.
- Include a minimal scientific workflow derived from the presentation prototype
  with upstream attribution, excluding presentation capture machinery.
- Pin Slurm source with Lua HRES support. Candidate inspected revision:
  `ba178ef837c4ba799fb3de72bcdaf30bc2b1616d` (26.11.0rc1 development source).
  Current source supports dynamic-node HRES membership through `HRES=`. The user
  chose a pinned source patch after testing exposed lost membership during
  controller reconfiguration; restore registration once per process.
- Use one Python 3.14 workload environment. The default scientific run uses
  4,096 circuits, 1,024 shots, 128 walkers, and 240 steps.
- Keep frontend compilation explicit in the example; automatic shared adapter
  integration is separate work in Core PR #2767.
- The email and attached YAML/Lua are untested design references. Verify
  callbacks and failure behavior against the pinned Slurm implementation.

## Work remaining

- [x] Build pinned Slurm with Lua and preserve the playground topology.
- [x] Verify initial closed admission, successful probes, failure/timeouts,
      recovery, per-device capacity, node eligibility, and accounting.
- [x] Exercise both actual provider libraries and compiled payloads against
      local services, including asymmetric result ordering.
- [x] Run the staged QC-AFQMC example and retain scientific and scheduler
      checks.
- [x] Update documentation and visually inspect the rendered changes.
- [x] Submit signed Core and presentations draft PRs with accurate validation.

## Validation

The final native image uses Core `b7d852be5800965ba3a762f3bdbc5e7c578b7835`,
rebased onto `865060b29a9be6a5896b9b6658f532c65e21e370`. Python 3.14, 15
host-side tests, Lua callback self-check with a real deadline, executable
documentation, and repository lint pass. The fresh cluster regression passed in
361 seconds including its build/startup (87 seconds for execution checks):
capacity, node eligibility, probe outage/recovery, running jobs across
reconfiguration, administrator membership changes across QAN restart,
environment isolation, and persistent accounting.

Full 4,096 x 1,024 chains completed through DDSIM (jobs 1/2/3), IQM (4/5/6), and
Braket (15/16/17), including four-node classical propagation. Every
source/payload hash and scientific/allocation check passes. All 16 basis states
with reversed classical destinations pass on all three devices. Braket also
passes the reopened-ProgramSet shots/counts check. Its final fixture metadata
correction is revision `6276527df65ac4472a911d5ca951178660390d82`. Retained IQM
and Braket wire requests contain all 4,096 programs; each Braket embedded source
contract matches the original QDMI input.

The native Braket correction is separate draft provider PR #288. Its 167 offline
C++ tests, 66 Python tests (one live test skipped), independent simulator
parity, lint, and full-file clang-tidy checks pass apart from existing AWS SDK
warnings. Provider documentation builds with notebook execution disabled and
five lexer warnings suppressed. No live notebook execution was attempted.

The initial unpatched controller lost dynamic HRES membership on reconfigure;
the source patch and regression resolve this. An earlier full Braket job also
survived a controller restart. Fresh-cluster retries initially hit host inotify
exhaustion; stopping unused task-owned clusters released watchers. No host
settings or unrelated containers were changed.

The separate five-slide presentations teaser uses retained run evidence. No
cloud or hardware jobs were submitted; remote reservations and site-specific
credentials remain unvalidated. Slurm HRES remains a development interface.
