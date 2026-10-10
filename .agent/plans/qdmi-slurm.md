# QDMI workloads on Slurm

Status: implemented and published; hosted checks pending.

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
Executable documentation and lint pass. Windows execution awaits hosted CI. The
upstream #2726 portable-CI condition is mirrored until it merges; the
Cache.cmake workaround is removed.
