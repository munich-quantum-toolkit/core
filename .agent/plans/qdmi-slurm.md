# QDMI workloads on Slurm

Status: rebased onto #2726; local multi-vendor validation passed.

## Scope and decisions

- Slurm licenses identify devices and limit allocations. Use ordinary job
  environments for catalogue and credential configuration; no SPANK module is
  needed by MQT Core, IQM, or Braket.
- A privileged site monitor reserves all licenses for an unavailable device.
  Block before probing; only a successful bounded check reopens scheduling.
  Running jobs continue. Polling needs supervision and is not a reservation at
  the remote device service.
- The checker uses installed Python catalogue metadata when launched through
  Python and bounded process-tree cleanup on POSIX and Windows.
- The scalable Docker cluster is shared by tests and demonstrations. Credentials
  enter at runtime. IQM Emerald mock and Braket SV1 execute small workloads in
  native and wheel modes; no quantum hardware is in scope.
- Keep the unreleased Core work consolidated in #2599 and update the two device
  PRs against upstream main. Preserve credentialed ordinary CI lanes.

## Validation

Run checker CLI/discovery and descendant-cleanup tests, the focused runner
suite, and a real Slurm availability block/recovery scenario. Exercise both
credentialed device workloads in native and wheel modes. Run repository lint and
full-file C++ lint. Record local and hosted results separately; publishing is
not a request to monitor CI.

Local validation: 48 focused Python cases and seven native tests pass. The
three-node cluster proves license capacity, outage blocking, and recovery.
Credentialed Emerald mock and SV1 tests pass in both installation modes.
Executable documentation and lint pass. The CMake changes from PR #2726 are
supplied by upstream; no overlapping patch remains. The Windows Python launcher
fix awaits hosted validation.

## Multi-vendor review

The standard QDMI session initializes every enabled catalogue entry. Use
separate installed environments and per-device catalogue overlays for
independent vendor jobs and monitors. The shared Docker image accepts an
optional wheel directory per environment; one controller schedules both vendors
concurrently. A hung or blocked vendor must not block the other.

The AFQMC application is prepared in a separate local follow-up worktree.
Quantum shadow collection holds one device license; dependent classical
propagation releases that license and uses CPU jobs. Keep the external
scientific helpers pinned and preserve their license provenance.

The combined wheel cluster passed real Emerald mock and SV1 workloads in
overlapping allocations, selective credential export, failed-IQM admission
blocking, a successful Braket check during the outage, and recovery. The
three-node core cluster and 48 focused Python cases also passed. The native
Windows checks passed on the previous hosted head; its Python launcher lost
nonzero exit codes. The launcher now waits and forwards the exit status on
Windows, with real help/error console tests replacing the mocked exec test.

The local AFQMC follow-up passed 32 DDSIM shadow circuits with 64 shots each
through Slurm, followed by an afterok CPU array with no quantum licenses. Four
walkers completed six steps and produced a summary. This is execution coverage,
not a chemistry convergence result.
