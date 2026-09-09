# Contract audit: QDMI performance on refreshed main

Status: all five findings implemented with user authorization on 2026-09-09.
Audit date: 2026-09-08. Baseline: `0c3fac2cb2861e9e25262857a84650cba651f466`,
clean upstream `main`. The user's checkout and its branch were preserved.

## Result

Five actionable opportunities remain. The first also fixes a concurrency bug:

1. Share DDSIM's dense-vector initialization guard between probabilities and
   statevectors.
2. Answer dense result size queries without materializing the vector.
3. Decode PennyLane samples with NumPy instead of Python lists of integers.
4. Stop querying QDMI operation names to look up PennyLane's existing cache.
5. Read Qiskit's device-wide duration conversion once per target snapshot.

The implementation is based on `ce608b082`, refreshed upstream `main`.
Prototypes and raw local evidence are in `/tmp/qdmi-main-audit`.

## Scope and recent changes

Inspected the C++ client/result decoders, driver/session and job ownership,
DDSIM and SC providers, Python bindings, Qiskit and PennyLane adapters,
`mlir/lib/Compiler/QDMIAdapter.cpp`, and the newly merged target environment.

Merged #2460/#2472/#2475 already address driver lock/copy costs, Slurm opening,
and the Python GIL. The DD/QCO execution changes and #2219 target environment
were included in the refreshed baseline. The target environment already reuses
its prepared target through the analysis manager; another cache is not
justified.

Open #2373 changes native multi-program jobs in the same DDSIM source. Its
inspected head `c8b4ac3d87e0db0265017b2767f470253c8ff003` still contains the
unguarded probability initialization. Coordinate a future provider fix with that
PR. The exact-payload/replaceable-driver stack remains separate work.

## Findings

### 1. Share the dense initialization guard

**Priority:** high. **Confidence:** high.

`src/qdmi/devices/dd/Device.cpp::getStateVector` initializes `stateVec_` with
`std::call_once(stateVecOnce_, ...)`. `getProbabilities` instead checks
`stateVec_.empty()` and assigns the same vector without synchronization.

Two cold probability readers can therefore write the vector concurrently. A
probability reader can also race a statevector reader. The released Python GIL
makes these overlapping calls possible from Python as well as C++. Even
sequential probability-then-statevector queries materialize the vector twice,
because the first path does not set the once flag.

The public C API probe used an all-zero 22-qubit state. Probability-size and
then statevector-size queries each allocated a 64 MiB vector. Four concurrent
cold probability-size queries allocated four such vectors. This confirms
duplicate initialization; the data race follows from the unprotected shared
writes. No ThreadSanitizer claim is made.

**Smallest fix:** use the existing `stateVecOnce_` in both methods. The isolated
prototype removed the second allocation; the second size query took 0.416 us
instead of 24.7 ms. All 65 DDSIM tests passed.

**Coverage gap:** `Concurrency.ConcurrentStatevectorReads` queries the size
before starting its workers, thereby warming the vector. Add a public-API
regression for cold probability and mixed probability/statevector readers. No
private test hooks or new synchronization abstraction are needed.

This corrects the earlier `qdmi-python-gil.md` statement that DDSIM already
protects all lazy result materialization: its probability path was missed.

### 2. Make dense size queries allocation-free

**Priority:** medium. **Confidence:** high.

Both dense accessors materialize `stateVec_` before reporting its byte count or
rejecting a too-small output buffer. The client uses the normal QDMI two-call
size/data protocol, so a caller merely sizing a buffer can trigger an
exponential allocation. The vector dimension is already available from the DD
root.

The same 22-qubit probe returned 32 MiB for probabilities and 64 MiB for complex
amplitudes. Baseline size-only queries took 17.8 ms and 24.7 ms and each
allocated 64 MiB. A prototype computed the required size from the root, checked
arithmetic overflow, and materialized under the shared once flag only for valid
data requests. The two size queries took 0.56 us and 0.192 us with zero dense
allocations. All 65 DDSIM tests passed.

Preserve terminal-result behavior, reported sizes, and small-buffer errors. The
implementation tests addressability and overflow through the public provider
API. Sparse sizes still depend on the number of nonzero entries and must not use
the dense rule. This removes work from size-only/rejected requests; full result
retrieval still requires dense data.

### 3. Vectorize PennyLane sample decoding

**Priority:** medium. **Confidence:** high for tested inputs.

`python/mqt/core/plugins/pennylane/device.py::_samples` validates every bit in
Python, creates a list of Python integers per shot, and copies those rows into a
NumPy `int8` array. NumPy is already a required dependency in this module.

An isolated variant retains per-shot width and binary validation, packs the
validated ASCII data with `np.frombuffer`, reverses the QDMI bit order, selects
the requested columns, and subtracts the ASCII zero. It avoids the nested Python
lists and per-bit `int` calls.

Median of five local conversions:

|   Shots | Wires |   Baseline | Prototype |
| ------: | ----: | ---------: | --------: |
|   1,000 |     2 |   0.430 ms |  0.059 ms |
| 100,000 |     2 |  50.588 ms |  5.570 ms |
|   1,000 |    32 |   2.385 ms |  0.105 ms |
| 100,000 |    32 | 272.540 ms | 12.702 ms |

Equality checks covered dtype, row order, reordered/subset/empty/repeated column
selections, spaces, wrong shot counts, wrong widths, invalid binary characters,
and non-ASCII input. Keep validation before packing: checking only total byte
length could accept individually malformed shots. The packed representation
still needs a temporary byte buffer; this is not a zero-copy result API.

### 4. Reuse PennyLane's local operation name

**Priority:** medium. **Confidence:** high.

`python/mqt/core/plugins/pennylane/converter.py::_validate_qdmi_contract` calls
`qdmi_operation.name()` for every gate to form the key of
`_operation_contracts`. The converter already resolved that operation against an
immutable advertised mapping. The Python operation name is available without
calling the provider.

A one-line prototype uses `(operation.name, spec.wires)` as the key. Repeated
10,000-gate RX conversion made zero extra QDMI name calls instead of 10,000;
median conversion time fell from 73.022 ms to 23.887 ms with the existing test
metadata double. Payloads were identical. These timings include mock overhead;
the eliminated query count is the stronger evidence for real providers.

Retain the per-converter cache lifetime and validation of each bound parameter
and wire. Aliases may create a few separate cache entries for one QDMI spelling;
that is harmless and avoids a new reverse map. Error-path queries can remain.

### 5. Snapshot Qiskit duration units once

**Priority:** medium. **Confidence:** high.

`python/mqt/core/plugins/qiskit/backend.py::_duration_seconds` queries the
session-wide duration unit and scale for every calibrated gate placement. The
surrounding `_build_target` already creates a calibration snapshot.

With two calibrated operations on 1,000 sites, target construction queried the
unit 2,000 times and the scale 2,000 times. A lazily initialized conversion
factor reduced each to one query, preserving the complete target duration
mapping. Median local mock time fell from 2.830 ms to 2.174 ms. At 32 sites,
each query count fell from 64 to one.

Keep initialization lazy: absent durations currently require no unit and must
not cause an error. Preserve invalid-unit and nonpositive/nonfinite-scale
checks. Scope the factor to the target/backend snapshot, never a global device
ID cache. The implementation resets the conversion at the start of target
construction. It retains the scale and unit separately to preserve
floating-point evaluation order.

## Retained boundaries and lower-priority candidates

- Registry indexing remains declined for the expected couple dozen devices.
- Eager child initialization and construction-time failures remain unchanged.
- IQM timeout configuration remains excluded as requested.
- Job destruction can still block Python; it remains the previously recorded
  lifetime-design question, not a newly established easy optimization.
- Sparse result parsing still uses an `istringstream`, but replacing it has not
  established a benefit comparable to the confirmed findings. Do not widen the
  patch into a new decoder or change public result ordering without evidence.
- Preserve frontend validation, complete preflight before batch submission,
  genuine ordered Qiskit memory, session lifetime, and native provider errors.

## Validation and reproducibility

Environment: ARM64 DGX Spark, GCC 13 release/IPO build, LLVM/MLIR 23.1.0; Python
3.14.7, NumPy 2.5.3, PennyLane 0.45.1, Qiskit 2.5.2.

- Configured `cmake --preset release` and built `mqt-core-qdmi-test` and
  `mqt-core-qdmi-ddsim-device-test`.
- Baseline: 241 client tests and 65 DDSIM tests passed.
- Shared-guard and allocation-free-size prototypes: 65 DDSIM tests passed for
  each. Restored baseline source, rebuilt, and reran all 65 successfully.
- Focused Nox session `tests-3.14`: 92 passed across the Qiskit mock-backend
  tests and PennyLane converter/device tests listed in
  `/tmp/qdmi-main-audit/frontend_variants_test.py`.
- The three frontend variants together passed those same 92 tests in-process,
  plus the sample-decoding and payload/target equality probes.
- `/tmp/qdmi-main-audit/dense_probe.cpp` uses the public provider C API and
  counts allocations matching the 64 MiB dense vector; run `dense-probe` for
  sequential reads or `dense-probe concurrent` for four cold readers.
- `frontend_probe.py`, `samples_probe.py`, and `frontend_variants_test.py` in
  that directory reproduce frontend call counts, timings, equality, and tests.
  `dense-lazy.patch` retains the provider prototype outside the repository.
- No live cloud/device calls, hardware jobs, hosted CI, or Windows validation.
  Timing probes isolate the relevant local work and do not predict end-to-end
  network latency. Publication is authorized for the five confirmed findings.

## Implementation and validation

- Both dense accessors calculate their byte count with checked shifts and
  multiplication. Size queries and rejected small buffers do not construct the
  vector. Valid data requests share `stateVecOnce_`; requests beyond the
  vector's addressable capacity return `QDMI_ERROR_OUTOFMEM`.
- Cold public-API regressions start probability-only and mixed readers together
  and check Bell amplitudes/probabilities. Size tests cover both output formats
  at representable and overflowing dimensions without allocating dense vectors.
- PennyLane retains per-shot width, binary, and shot-count checks, `int8`
  output, spaces, and requested wire order. Its existing cache test also
  verifies that valid gates require no further operation-name queries.
- Qiskit tests cover lazy metadata reads across multiple calibrated operations
  and placements, zero durations, and independent target snapshots. Existing
  invalid-unit and invalid-scale checks remain.
- Local native validation: all 67 DDSIM and 241 client tests passed.
- Focused frontend validation: all 102 Python tests passed on Python 3.14.
- Full `uvx nox -s lint` and `uvx nox -s cpp-lint -- ce608b082` passed; C++ lint
  checks every line of each changed C++ file.
- Final 22-qubit probe: four concurrent size queries allocated no dense vectors;
  subsequent probability/statevector size queries took 0.03-0.16 us.
- Open #2373 still has the same overlapping DDSIM source at the audited head;
  this PR does not include its multi-program changes.
