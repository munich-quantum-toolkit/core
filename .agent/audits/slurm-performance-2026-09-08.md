# Contract audit: Slurm build, selection, and tests

Status: applied locally. Baseline: `552feffdbf5e2396660bb3e91a21936a9d7adf63`,
initially clean. Date: 2026-09-08.

## Result

Three supported improvements remove avoidable costs and bound fixture failures:

1. Remove the adapter's full device-catalog copy before opening one device.
2. Bound fixture commands, isolate concurrent runs, and verify terminal job
   status.
3. Enable the existing compiler cache and remove the wheel archive from image
   layers.

The adapter does not call the Slurm controller. Provider initialization and
network calls remain provider-owned; this audit cannot guarantee a universal
latency bound. Existing issues #2310-#2313 and #2366 cover availability and
optional admission/checker work and remain separate from static-license
selection.

## Findings

### Remove duplicate catalog work

`src/qdmi/Slurm.cpp::openDeviceFromLicense` copied every registered ID and
searched that copy before `Session::openDevice` performed the existing driver
lookup. Syntax validation now precedes driver initialization, and the adapter
opens the selected ID directly. An unknown ID still raises the adapter's runtime
error. Each call opens a fresh session and accepts IDLE or BUSY; no cached
provider status or device handle changes those semantics.

Contract sources: `include/mqt-core/qdmi/Slurm.hpp`,
`src/qdmi/Client.cpp::Session::openDevice`,
`src/qdmi/driver/Driver.cpp::Driver::openFresh`, and `docs/qdmi/slurm.md`.
Native and Python adapter tests retain malformed, compound, remote, unknown, and
unavailable-device rejection. The added native test checks fresh handles and the
unknown-ID diagnostic through the shared lookup.

A native ARM64 probe registered the selected fixture device first, warmed the
adapter, then counted allocations and elapsed time across 100 opens. It repeated
with 1,000 and 10,000 unrelated registered IDs. The selected device used the
existing session fixture; registration and startup were outside measurement.
Only the adapter translation unit changed between the two linked probes.

| Unrelated IDs | Before allocations | After allocations | Before ms | After ms |
| ------------: | -----------------: | ----------------: | --------: | -------: |
|             0 |              1,400 |             1,200 |     0.494 |    0.215 |
|         1,000 |            101,400 |             1,200 |    13.002 |    0.212 |
|        10,000 |          1,001,400 |             1,200 |    63.826 |    0.225 |

This isolates the removed catalog copy. The shared driver lookup still scans its
registry, so selecting the last ID remains linear. An index is deferred until
representative registry sizes and lookup positions justify shared-driver
changes.

### Bound failures and keep real scheduler coverage

`test/slurm/run_integration.py` previously allowed subprocesses to wait without
a deadline and shared a fixed project and runtime directory across invocations.
Commands now have deadlines and terminate their process group on interruption,
including Compose children holding output pipes. Startup, diagnostics, and
teardown have separate limits. Batch jobs request five minutes. Unique Docker
projects, runtime paths, and Munge keys prevent concurrent runs from overwriting
or tearing down each other's resources.

Checked command failures retain output. Diagnostics cannot suppress the original
failure or bypass teardown. Successful runs remove their resources; failures
retain artifacts. This bounds ordinary subprocess stalls, not an unresponsive
kernel or Docker daemon's internal work after its client has been terminated.

Positive and negative jobs now require the expected `scontrol show job` terminal
state and exit code. Queue disappearance and an earlier result alone cannot turn
a later job failure into success. Result writers atomically publish JSON. The
real two-node test retains license contention, an independent SC job, Bell
results, and invalid license coverage. No accounting daemon is required.

Fourteen inexpensive runner tests cover subprocess descendants, checked error
output, terminal states, preflight cleanup, diagnostic failure, and invocation
isolation. These run before the wheel build in CI. Two concurrent real fixtures
passed in 31.41 and 31.25 seconds and cleaned up independently. The final
fixture, including the five-minute batch limit, passed in 32.69 seconds.

The original fixture took 56.60 seconds locally; changed Docker cache warmth
prevents attributing that whole difference to this patch. Polling stays confined
to the disposable test controller;
[Slurm warns that repeated client RPCs can burden a production controller](https://slurm.schedmd.com/squeue.html#SECTION_PERFORMANCE).

### Cache compilation and reduce image layers

A recent successful hosted Slurm job spent 323 seconds building its wheel and 77
seconds in integration (run 34216386560). `cmake/Cache.cmake` already supports
sccache, so `.github/workflows/slurm.yml` now installs the pinned action and
sccache 0.16.0 with its GHA backend enabled. It also adds missing build/test
path triggers and 15-minute wheel and integration step limits.

A controlled local comparison used GCC 13, sccache 0.16.0, the same configured
build directory and source, and a clean CMake build target before each wheel
build. The first used an empty dedicated disk cache; the second reused it.

| Build | Compile requests | Cache hits | Cache misses | Wall seconds |
| ----- | ---------------: | ---------: | -----------: | -----------: |
| Cold  |               67 |          0 |           67 |        63.27 |
| Warm  |               67 |         62 |            5 |        41.90 |

Both had zero cache errors. The local warm build reduced wall time by 33.8%.
This validates compiler-cache usefulness, not hosted cache availability or a
hosted speedup. The action reports counters; inspect requests and hits together
with wall time after publication. Configuration follows the
[sccache action inputs](https://github.com/Mozilla-Actions/sccache-action/blob/v0.0.11/action.yml)
and
[GHA backend documentation](https://github.com/mozilla/sccache/blob/main/docs/GHA.md).

`test/slurm/Dockerfile` installs the wheel through a BuildKit bind mount and
uses a uv cache mount. The archive no longer persists in a COPY layer. Image
size fell from 516 MB to 450 MB; controller and node images share layers, so
this is not three independent disk savings. No custom wheel variant or
compiler-cache framework was added.

## Validation

- Native rebuilt `mqt-core-qdmi-test --gtest_filter=SlurmAdapterTest.*`: 6
  passed.
- Updated wheel, Python 3.14.7,
  `pytest -o addopts= -q test/python/qdmi/test_slurm.py`: 4 passed.
- Runner tests: 14 passed with the standalone command documented in
  `docs/qdmi/slurm.md`.
- `uv run --no-project --python 3.14 test/slurm/run_integration.py`: passed on
  privileged Linux Docker with cgroup v2, including two simultaneous runs.
- `uvx nox -s lint`: passed, including Python type checks.
- `uvx nox -s cpp-lint -- 552feffdbf5e2396660bb3e91a21936a9d7adf63`: passed with
  zero findings across both changed C++ files, checking all their lines.

Hosted CI for this patch has not run. Provider/network tail latency, hosted
compiler-cache hit rate, and large-registry lookup position remain measurement
limits, not established bottlenecks or guarantees.
