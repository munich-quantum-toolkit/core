# Contract audit: Python QDMI calls and concurrency

Status: accepted fixes implemented. Date: 2026-09-08. Core baseline:
`7e2a2679f`; isolated worktree initially clean. Braket baseline: `a5ace5a`; its
separate fix is PR #217. Related open Core PRs were refreshed: #2472 covers
Slurm overhead; #2373 changes native multi-program jobs. Neither implements
these GIL boundaries.

## Result

Release the GIL across native QDMI opening, queries, job operations, and
compiler-target snapshots. Protect the SC provider's shared job map before
allowing Python submissions to overlap. Keep Python conversion under the GIL.

The maintainer declined registry indexing and IQM timeout changes, and retained
eager child initialization. These are not remediation work for this PR.

## Contract and ownership

- `bindings/qdmi/qdmi.cpp` delegates to `qdmi::Session`, `Device`, `Site`,
  `Operation`, and `Job` in `include/mqt-core/qdmi/Client.hpp` and
  `src/qdmi/Client.cpp`. Their native calls do not use Python objects.
- `src/qdmi/driver/Driver.cpp` copies definitions under `stateMutex_` before
  provider work. Its module cache serializes provider initialization within one
  loaded module. Its job map has its own lock. Fresh sessions and child handles
  retain their existing ownership and eager initialization.
- Provider calls can block during initialization, metadata queries, submission,
  cancellation, retrieval, and results. Restricting GIL release to `Job.wait`,
  shots, and counts does not cover those other entry points.
- `bindings/mlir/register_mlir.cpp` has separate QDMI paths in
  `CompilerTarget.from_device` and `from_device_id`. Their snapshot work is
  native; `takeResult` converts errors at the Python boundary.

## Findings and implemented changes

### 1. Release the GIL across native provider work

The nanobind call guards cover direct native methods. Scoped guards cover native
parts of lambdas, including custom-property queries and binary submissions.
Bytes pointers and lengths are obtained before releasing the GIL; the immutable
Python payload remains alive for the synchronous native call. Results are
converted to Python only after the guard is destroyed. Custom type inspection,
`nb::bytes` construction, and compiler error conversion keep the GIL.

The regression uses a named pipe as the SC configuration file. A Python writer
thread cannot write until native opening has opened the reader. Holding the GIL
then prevents the writer from progressing. A subprocess timeout bounds failure
without a timing threshold for successful execution. The test covers generic,
Slurm, and compiler-target opening with the real SC provider. It is POSIX-only.

All three cases timed out against the previous bindings and passed with the fix.
A separate native fixture with a 400 ms session-init delay measured a maximum
Python heartbeat gap of 405-406 ms before the fix and 5.91-5.96 ms after it.
Opening still took about 400 ms and created one fresh session: the change
restores Python progress without reducing provider latency.

### 2. Protect the SC provider's shared job map

`src/qdmi/devices/sc/Device.cpp::createDeviceJob` and `freeDeviceJob` mutate one
session's unordered map without synchronization. Python submission allocates a
job before the SC provider rejects execution, so even unsupported submissions
can reach this race. The driver job-map lock does not cover provider allocation
or release. Add one provider-owned mutex around insertion and erasure.

A native regression creates and frees 4,000 jobs on four workers sharing one
initialized session. Other SC metadata is immutable after initialization. DDSIM
already protects its session/job maps and lazy result materialization. These
checks do not certify every external provider for concurrent calls on one
handle; provider thread-safety requirements still apply.

## Further opportunities and retained boundaries

- Python object destruction can still block while a provider frees a job or
  session. DDSIM can wait for its asynchronous job during destruction. Explicit
  waiting already releases the GIL. Changing finalization needs a separate
  lifetime design and regression; it is not equivalent to adding a call guard.
- Do not add a global provider-call lock: it would prevent cancellation and
  independent queries from progressing during a wait. Shared-handle concurrency
  remains provider-owned and is documented in `docs/qdmi/driver.md`.
- Keep the ordered registry without an index. The earlier 32-definition probe
  measured about 0.53 microseconds per warm native open; the maintainer expects
  at most a couple dozen definitions.
- Keep eager child initialization and its construction-time errors. The delay
  fixture still initialized one, two, and nine sessions for zero, one, and eight
  children, respectively.
- Keep IQM's timeout behavior as requested. No live provider requests or
  hardware executions were made.
- Braket PR #217 dispatches local and unsupported properties before GetDevice.
  Its refreshed baseline already caches architecture and refreshes status/queue;
  it does not fetch architecture for every warm property query.

## Validation

- `uvx nox -s tests-3.14 -- test/python/qdmi test/python/test_mlir.py`: 304
  passed, including the three named-pipe cases and existing custom-value,
  binary-job, exception, lifetime, and compiler-target tests.
- `mqt-core-qdmi-sc-device-test`: 44 passed; one existing unsupported job-ID
  test skipped. The concurrent job lifecycle regression passed.
- `uvx nox -s stubs`: regenerated successfully, with no committed stub changes.
- `uvx nox -s cpp-lint`: all five changed C++ source files checked in full, zero
  findings. `uvx nox -s lint`: passed.
- Measurements describe local native fixtures, not network latency. Windows,
  live providers, and hosted CI are not validated by these local checks.
