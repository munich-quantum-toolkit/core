# QDMI performance and efficiency investigation

Status: proposed. No item is measured or implemented yet. Work items 1–4 are
ready to start. Items 5 and 6 need a maintainer decision after measurement.

## Goal and scope

Find and remove avoidable work in the QDMI libraries on paths that users hit
repeatedly: batch sampling on the DDSIM device, sample readout in the PennyLane
and Catalyst frontends, process exit after DDSIM use, and the first device open.
Every change must keep the QDMI Client interface, result formats, error codes,
and the documented DDSIM contract
([`docs/qdmi/ddsim_device.md`](../../docs/qdmi/ddsim_device.md)) unchanged
unless the item names a decision for the maintainer.

The review covered these components:

- the QDMI driver: `src/qdmi/driver/Driver.cpp`, `DeviceRegistry.cpp`;
- the C++ QDMI library: `include/mqt-core/qdmi/QDMI.hpp`, `src/qdmi/QDMI.cpp`;
- the DDSIM device: `src/qdmi/devices/dd/Device.cpp`, `Worker.cpp`,
  `WorkerProtocol.cpp`, `WorkerMain.cpp`;
- the SC device: `src/qdmi/devices/sc/Device.cpp`;
- bindings: `bindings/qdmi/qdmi.cpp`, `bindings/mlir/Target/PennyLane/`;
- consumers: `mlir/lib/Compiler/QDMIAdapter.cpp`,
  `mlir/lib/Runtime/Catalyst/QDMIDevice.cpp`, `python/mqt/core/plugins/`.

Out of scope: the QDMI standard, external QDMI device libraries (IQM, Amazon
Braket), simulation and QIR JIT algorithms, and MLIR passes.

### Earlier decisions to respect

These audits already fixed or declined several candidates. Do not reopen them
without new evidence:

- [`qdmi-driver-performance-2026-09-08`](../audits/qdmi-driver-performance-2026-09-08.md):
  module-level initialization locking and configuration copies are fixed.
  Registry indexing, a job pool, a metadata cache, and an extra loader cache are
  declined.
- [`qdmi-main-performance-2026-09-08`](../audits/qdmi-main-performance-2026-09-08.md):
  dense-result guards and allocation-free size queries are fixed. Note that
  `stateVecOnce_` no longer exists. The DDSIM result code now lives in
  `qdmi::dd::ProgramResult`.
- [`qdmi-pre-release-performance-2026-09-10`](../audits/qdmi-pre-release-performance-2026-09-10.md):
  GIL release and scalar site queries are fixed. A full DDSIM target snapshot
  still visits all 65,535 sites. Users can limit this with the session width
  (`CUSTOM1`, see the DDSIM documentation).
- [`qdmi-python-gil`](../audits/qdmi-python-gil.md): the maintainer kept eager
  initialization of child devices and of the client catalog.

Those audits refer to `include/mqt-core/qdmi/Client.hpp` and
`src/qdmi/Client.cpp`. These files are now `QDMI.hpp` and `QDMI.cpp`.

## Work items

Order is by expected effect and confidence. Each item lists evidence from the
current code, a measurement, the threshold for acting, and the smallest fix.
Record each measurement with the workload, baseline revision, machine, and
median of at least five runs
([performance evidence](../../AGENTS.md#performance-evidence)). Keep harnesses
outside the repository.

### 1. DDSIM moves and keeps the sampled state even when no caller asks for it

**Evidence.** When a sampling program uses terminal sampling, the worker keeps
the uncollapsed state. `execute` in `WorkerMain.cpp` serializes it with
`dd::serialize`. `encode(const WorkerResponse&)` copies it into the frame.
`decode(StringRef, WorkerResponse&)` copies it again with `state->str()`. The
submit task in `MQT_DDSIM_QDMI_Device_Job_impl_d::submit` then creates a
separate `dd::Package` for each program, deserializes the state, and keeps both
until the job is freed. The DDSIM documentation promises state results for these
jobs, so the state must stay available. However, no frontend queries it: the
Qiskit, PennyLane, and Catalyst paths read only shots or counts.

**Expected impact.** CPU time on both sides of the socket, plus parent memory
that grows with Σ (DD size) across all programs in a batch. Structured circuits
produce small DDs. Highly entangled 16–24-qubit circuits can produce DDs with
roughly 2^n nodes, multiplied by the number of programs (for example, a
400-circuit PennyLane batch).

**Measure.** Use OpenQASM 3 and Base QIR batches of 100 programs at 12, 16, 20,
and 22 qubits. Use both GHZ-like circuits and random circuits with full
entanglement, at 1,000 shots each. Record:

1. the serialized state size for each program;
2. worker time in `dd::serialize`;
3. parent time from `readFrame` to the end of `deserialize`;
4. peak and retained parent RSS until the job is freed;
5. end-to-end time to read counts.

Compare against a local prototype that drops `response.state` in the worker. Use
this prototype only as a measurement baseline, never as a fix.

**Act if** deserialization or retained parent memory accounts for at least 5 %
of end-to-end time, or adds at least 100 MiB for a 100-program batch, in any
realistic row.

**Smallest fix (no contract change).** Keep the received serialized bytes in
`ProgramResult` and deserialize them at most once, on the first state or
probability query, behind a `std::once_flag`. This follows the pattern of
`countsOnce_` and `sparseAmplitudeCountOnce_`. It removes the per-program
`dd::Package` and the deserialization cost for results that nobody queries. Move
the frame payload into the result instead of calling `.str()` on a view into it,
so the copy is removed too.

Decide and document one change in failure timing. Today, a qubit-count mismatch
marks the program `FAILED`. With lazy deserialization, the program finishes as
`DONE` with valid samples, and the first state query returns `QDMI_ERROR_FATAL`.
The worker is a trusted child process, so this case indicates an internal
defect.

**Escalation (maintainer decision).** Lazy deserialization still pays for
serialization and transfer. If those costs remain material, the options are:

- (a) a new DDSIM job parameter that skips state retention, which Qiskit,
  PennyLane, and Catalyst would set;
- (b) keep the state in the worker until the job is freed, which reduces how
  often the pool can reuse workers.

Option (a) is a new public device contract, so document it in
`docs/qdmi/ddsim_device.md` and in the PR's migration notes.

**Tests.** In `test/qdmi/devices/dd/results_statevector_test.cpp` and
`results_sampling_test.cpp`, check that a sampling job still returns
statevectors, probabilities, and sparse results that match today's values. Check
that cold concurrent state queries on one result deserialize once and agree.
Check that sampling-only consumers never deserialize the state. Count
deserializations through the public API (for example, query timing or a test
build of the device), not through a private hook in production code.

### 2. Shutting down the DDSIM worker pool takes time linear in the number of idle workers

**Evidence.** `WorkerPool::shutdown` in `Worker.cpp` calls `Worker::shutdown`
for each idle worker in turn. Each call writes the close frame and then polls
`hasExited` in 10 ms sleeps for up to 1 s, before it moves on to the next
worker. The pool can grow to `Executor::limit` workers (physical cores, or
`MQT_CORE_DD_WORKER_BUDGET`). The static `Executor` destructor runs this path at
process exit. `MQT_DDSIM_QDMI_device_finalize` also runs it. The driver leaks
its device-library cache on purpose, so in practice the exit path is the one
that runs.

**Expected impact.** Process exit is delayed by about 10 ms or more per idle
worker. On a 64-core node, this is 0.6 s or more after any batch that used the
whole budget. CLI calls, test suites, and SLURM job steps all pay this delay.

**Measure.** Run a Python script that submits a batch with as many programs as
`limit`, reads the counts, and exits. Time interpreter exit with 1, 8, and
`limit` idle workers.

**Act if** exit time grows with the number of workers. The code predicts this,
so the measurement mainly sizes the gain.

**Smallest fix.** Use two phases in `WorkerPool::shutdown`. First, write the
close frame to every idle worker and reset its stream. Then wait for all of them
against one shared 1 s deadline. Terminate any worker that has not exited by
then. Keep the existing behavior for active workers (`terminate`) and for
`accepting_`. Do not change `Worker::shutdown` for its single-worker caller in
`WorkerPool::release`.

**Tests.** Add a regression to `session_lifecycle_test.cpp` or
`concurrency_test.cpp`. It warms N workers, calls the device finalizer through
the public C API, and requires that no worker process remains. Do not assert a
wall-clock bound. Use a generous bound only if one is needed, and give its
reason.

### 3. Catalyst and PennyLane readout repeats string work for every shot

**Evidence.** `readout` in `mlir/lib/Runtime/Catalyst/QDMIDevice.cpp` reads
`qdmi::Job::getShots`. `detail::parseShots` allocates one `std::string` per shot
(a heap allocation for widths above the SSO limit). `readout` then runs
`std::erase`, `ResultLayout::decodeBits` (another new string), a digit check,
and `std::ranges::reverse` for every shot. Finally, `decodeSamples` expands the
results into `int8_t`. For DDSIM jobs without captured output, the device has
already expanded SHOTS from a compact histogram (`ProgramResult::getShots`), so
the number of distinct outcomes is often far below the number of shots. The
Qiskit frontend already encodes each distinct bitstring only once (`QDMIJob`,
`encoded = {bits: ... for bits in set(shots)}`).

**Measure.** Time `mqt.core.mlir._pennylane.samples` on DDSIM with {1,000,
100,000, 1,000,000} shots × {4, 32} wires, with and without a result layout.
Report the share of time spent in `getShots`/`parseShots` and in the per-shot
loop of `readout`.

**Act if** the per-shot loop takes at least 20 % of `samples()` at 100,000
shots.

**Smallest fix.** In `readout`, validate, decode, and reverse each distinct raw
bitstring once, using an `llvm::StringMap`, and keep shot order. Keep every
current error message and every validation of width, digits, and total count.
Change only `readout`. A new C++ QDMI library API that returns one contiguous
SHOTS buffer would add a public commitment. Propose it separately, and only if
`parseShots` still dominates after this fix.

**Tests.** The existing Catalyst and PennyLane sample tests in
`mlir/unittests/Runtime/Catalyst/test_qdmi_device.cpp` and
`test/python/plugins/qdmi_pennylane/` must pass unchanged. Add one case with
repeated outcomes and a non-identity layout, and one malformed-sample case that
appears only after the first occurrence of an outcome.

### 4. The histogram in the C++ QDMI library allocates one map node and key per outcome

**Evidence.** `getSparseResult` in `src/qdmi/QDMI.cpp` builds
`std::map<std::string, T>`. This costs a node allocation, a key string, and an
O(log n) insert per outcome. The bindings then convert the result into a Python
`dict`. High-entropy sampling (20 qubits or more, 10^6 shots) yields up to 10^6
distinct keys.

**Measure.** Time `Job::getCounts` and Python `Job.get_counts` on DDSIM with
uniform-superposition circuits at 16, 20, and 24 qubits and 10^6 shots. Split
the time into provider copy, parsing and map insert, and Python conversion.

**Act if** map construction takes at least 25 % of `get_counts`. If so, insert
with `emplace_hint(end())` when the provider keys are already sorted. DDSIM keys
come from `std::map` in `ProgramResult`, so they are already in sorted order.
QDMI does not guarantee key order, so keep a fallback for unsorted keys. Do not
change the return type of `Job::getCounts`. Doing so would break
`mlir/tools/run/Run.cpp`, Catalyst, and the bindings, and it needs maintainer
approval.

### 5. The first `open_device` opens every configured device in sequence (decision required)

**Evidence.** Python `qdmi.open_device`, `qdmi.device_ids`,
`QDMIBackend.from_device_id`, and `compilerTargetFromDeviceId` all reach
`qdmi::Session::create`. This calls `QDMI_session_alloc` and
`Driver::materializeClientCatalog`, which opens every configured device, one
after another, the first time any session is allocated. With N remote devices,
the first open of any device costs Σ (session-init latency). The targeted path
`builtin_driver::openDevice` (`Driver::sessionAllocForDevice`) avoids this cost.
The maintainer previously kept eager catalog opening because lazy opening
changes when errors appear.

**Measure.** Use an ad hoc device library with 200 ms of session-init delay.
`test/qdmi/driver/session_device.cpp` is a starting point, but it has no delay
today. Register it K = 1, 4, and 8 times with a manifest. Time the first
`open_device("mqt.ddsim.default")`.

**Options for the maintainer.**

- (a) Open the catalog devices concurrently in `materializeClientCatalog`. Eager
  semantics, error timing, warning text, and registration order all stay the
  same, and latency drops from Σ to max. Devices in the same library already
  serialize on the module mutex.
- (b) Route ID-specific entry points through the targeted session. This changes
  session sharing and how strict parameter handling is.

Option (a) is the smaller contract change. Do not implement either option
without a decision.

### 6. Idle DDSIM workers never shut down (decision required)

**Evidence.** `WorkerPool::release` keeps every reusable worker until the
process exits, so a burst of `limit` concurrent programs leaves `limit` worker
processes alive. Each idle worker keeps the resident memory of its last program,
depending on the allocator. Long-lived hosts are affected: Jupyter kernels, the
SLURM availability service in `examples/slurm/`, and the PennyLane device.

**Measure.** Record per-worker RSS after it is idle, following programs at 4,
16, and 22 qubits, and the total idle RSS after a `limit`-wide batch.

**Options for the maintainer.** Cap the number of idle workers, add an idle
timeout, or keep the current behavior. All of these are policy choices. Keep the
trade-off with startup cost in view: measure the latency of the first program on
a cold worker (process spawn, socket accept, and LLVM/MLIR start-up in
`Worker::start`) before deciding.

## Candidates not promoted

These candidates were checked and not promoted. The reasons are given so that
nobody repeats the same work:

- **Driver path resolution on every open.** `packagedDriverPath` probes up to
  seven candidates. `normalizePath` calls `weakly_canonical`. The cost is
  microseconds per open, far below provider session initialization.
- **`std::shared_ptr` copies in `Site` and `Operation` wrappers.** There are
  65,535 atomic increments per full DDSIM `getSites`. This is small next to the
  five provider queries per site that the snapshot already makes, and those
  queries are a retained decision.
- **`std::vector<QDMI_Site>` in `Operation::queryProperty`.** It allocates only
  when sites are passed. The snapshot passes sites once per calibrated tuple.
- **Growth of the `readFrame` buffer in 64 KiB chunks.** Doubling amortizes this
  growth. It also deliberately avoids trusting an announced length. Fold the
  related copies into item 1 instead.
- **SC device queries.** These are binary searches over sorted tuples, with no
  global locks.
- **`discover_qdmi_manifests` at import time.** It scans the entry-point group
  once per process. Measure it with
  `python -X importtime -c "import mqt.core.qdmi"` only if import latency is
  reported.
- **Sorted SHOTS output for compact DDSIM jobs.** This is documented behavior
  ("the sequence of shots has no defined meaning"), not a defect.

## Validation

For every implemented item:

- Configure and build with `cmake --preset release`, then
  `cmake --build --preset release`.
- Run the focused native tests, `ctest --preset release -R qdmi`, and the
  affected component binaries with GoogleTest filters.
- Run the focused Python tests,
  `uv run --no-sync pytest test/python/qdmi test/python/plugins`.
- Run `uvx nox -s cpp-lint` and `uvx nox -s lint`. If a binding signature
  changes, also run `uvx nox -s stubs`.
- Report before and after measurements for the item's workload, and state the
  limits of the evidence (a single machine, no hardware, no free-threaded
  Python).

Items 1–3 do not change any public API. Item 1 escalation (a), item 4 beyond
hint insertion, and items 5 and 6 need maintainer approval first.
