# Native multi-program QDMI execution

Status: in progress; provider draft publication and the successor native
error-handling migration remain.

## Goal and scope

Use one ordered program list for QDMI submissions, with common format and shots
per program. Qiskit and PennyLane group compatible inputs and retain their
original result order. Unsupported native groups fall back to independent
submissions. This builds on the replaceable driver, stable IDs, and installed
runtime staging.

## Decisions

- `qdmi::Device` owns configuration and admission; only a setter returning
  `NOTSUPPORTED` can trigger fallback. Submission errors preserve uncertainty.
- `python/mqt/core/plugins/qdmi_batch.py` retains the existing recovery model.
  Attempts carry a shared job handle and program index. Read failures never
  cause replacement execution; only confirmed failed programs consume automatic
  retry budgets.
- DDSIM supervises reusable subprocess workers through one LLVM physical-core
  thread pool. Each program receives fresh compiler, runtime, and DD state, with
  nested MLIR threading disabled. Compact DD results move to parent-owned
  packages so workers can be reused independently of result lifetimes.
- Per-job completion tracking protects queued cancellation and active process
  termination without waiting on LLVM deferred futures. A crashed worker fails
  its current program without replaying it or discarding siblings.
- Worker isolation is not a security sandbox or a rollback guarantee for
  arbitrary native process-global effects. Broad native error-handling changes
  remain in the successor PR.

## Work remaining

- [x] Public bindings, native forwarding, concurrent DDSIM execution, and worker
      staging.
- [x] Shared-handle submission, sequential fallback, selective recovery, and
      uncertain operations in both adapters.
- [x] Controlled concurrency, cancellation, process reuse, seeded output, and
      retained results.
- [ ] Complete clean installed provider SDK checks; installed C++/Python
      consumers already pass with both drivers.
- [ ] Publish the validated stack and inspect hosted platform checks.

## Validation

Use the repository's build, lint, and test entry points in
[AGENTS.md](../../AGENTS.md). Native tests under `test/qdmi`, shared recovery
tests under `test/python/plugins/test_qdmi_batch.py`, and adapter suites must
exercise observable behavior rather than scheduling timing. Installed checks
must resolve worker and device assets after relocation. Local validation passes:
3,633 native tests with one expected SC skip, 1,543 Python tests, generated
stubs, documentation and generated-link checks, and native lint. The
affinity-limited worker tests pass with the simultaneous-worker case skipped
when only one core is available. Identical installed C++ and Python applications
execute ordered programs through Core's driver and the QDMI example driver
without rebuilding clients; the installed runtime helper stages the DDSIM worker
and SC assets. The example-driver test also covers optional unsupported
authentication without suppressing permission errors.

The specialist review fixed unconditional worker completion, shutdown ordering,
Windows UTF-8 executable/argument handling, and active-job retry protection.
Framing, execution ownership, and retained attempt state are required by these
contracts; no larger abstraction is needed.

A Linux AArch64 MinSizeRel/LLVM 23 measurement at native commit `3c98fdf9d` ran
four 12-qubit, 12-layer programs with 4,096 shots each and seed 17. Sequential
cold/warm-median times were 0.705/0.711 seconds; native concurrent times were
0.235/0.228 seconds. Sampled peak process-tree RSS was 100/328 MiB respectively,
counting shared pages per process. This is a representative workload, not a
general throughput claim. Counts matched exactly. Hosted CI is reported
separately.
