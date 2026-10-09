# QDMI workloads on Slurm

Status: implemented; publication and hosted checks pending.

## Goal and scope

Provide one Slurm integration above MQT Core's QDMI client interface. Device
implementations supply their runtime, credentials, and a small smoke test. The
Docker cluster in `docker/slurm/` supports tests and local demonstrations.

## Decisions

- Slurm license names identify QDMI devices; admission and device authorization
  remain separate. The selector accepts one ID with an optional `:1` count.
- Use Slurm's environment export for job settings. The optional source-only
  SPANK module supplies license-specific site defaults without loading devices.
- Run the availability command after selecting the workload environment. Keep
  its bounded native worker; remove automatic SPANK validation and per-step
  state.
- Use the shared native command launcher and runtime installation helpers.
- Use one scalable Compose compute service, with disposable rootful Linux
  cgroup-v2 hosts as the supported deployment boundary.
- Consolidate the unreleased Core stack so these contracts are reviewed
  together.

## Progress

- [x] Simplify the plugin, checker, documentation, and test coverage.
- [x] Separate reusable cluster setup from test assertions and validate scaling.
- [x] Refresh device integrations on upstream main, simplify smoke tests, and
      make ordinary IQM CI work without live service credentials.
- [x] Validate focused native/Python checks, standalone SPANK, and cluster
      scaling.
- [x] Pass IQM and Braket smoke tests in both native and wheel modes.
- [ ] Publish the revised PRs and inspect hosted checks.

## Validation

Run the focused checker tests, native command tests, standalone SPANK lint,
cluster admission/environment tests, and IQM/Braket native/wheel smoke tests.
Run each repository's required lint. Check final-head CI separately from local
results. Real quantum hardware is outside this task.
