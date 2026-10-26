# HRES Slurm playground and hybrid workflow

Status: implemented; final provider run and draft publication are in progress.

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
- [ ] Exercise both actual provider libraries and compiled payloads against
      local services, including asymmetric result ordering.
- [ ] Run the staged QC-AFQMC example and retain scientific and scheduler
      checks.
- [x] Update documentation and visually inspect the rendered changes.
- [ ] Submit signed Core and presentations draft PRs with accurate validation.

## Validation

Pinned Slurm and the Core/provider images build with Python 3.14. The 15
host-side tests, Lua callback self-check (including a real process deadline),
complete executable documentation build, and repository lint pass. The initial
real-cluster integration passed admission capacity, outage/recovery, isolation,
MPI environment, and persistent accounting. The final patched cluster passed in
277 seconds including image build/startup; its 84-second execution checks
additionally verify node eligibility, reconfiguration with running allocations,
and preservation of administrator membership changes on QAN re-registration.

Full 4,096 x 1,024 runs completed through DDSIM (jobs 41/42/43) and IQM
(44/45/46), including four-node classical propagation and retained artifact
checks. All 16 basis states with reversed classical destinations pass both
providers. Braket exposed a native classical-output ordering bug; the fix is
separate draft provider PR #288, pinned by this example. Its native C++ tests,
Python tests, and independent Qiskit/Braket simulator comparison pass. The final
Braket full run remains to be recorded.

Fresh-cluster retries hit host inotify exhaustion before scheduling; the unused
task-owned base cluster was stopped to release watchers. This was a host
resource failure, not an AFQMC or Slurm job failure.

The separate five-slide presentations teaser uses retained run evidence. No
cloud or hardware jobs were submitted; remote reservations and site-specific
credentials remain unvalidated. Slurm HRES remains a development interface.
