# QC/QCO wire and converter follow-up

Date: 2026-10-19. Follow-up to the
[combined audit](qc-qco-combined-2716-20261018.md) for
[PR #2716](https://github.com/munich-quantum-toolkit/core/pull/2716).

## Changes and contracts

Neither `WireIterator` nor `WireRange` had been deleted. The earlier audit
replaced three read-only range uses with direct scans, leaving the range unused
in production. Those bundled measurements did not isolate a benefit from the
caller changes. This follow-up restores `WireRange` in rotation-chain
collection, single-qubit fusion scanning, and measurement routing. The direct
tail-first fusion deletion loop remains: it obtains the predecessor before
erasing the current operation.

The iterator now uses its sole `OpOperand` position to select the corresponding
unitary output, respecting the quantum operand range offset. Walking all W wires
through a W-wide barrier or unitary call therefore requires O(W), rather than
O(W²), correspondence work. Redundant tail classification is removed; the
existing default branch preserves the same terminal input value. Backward lookup
already uses result positions for wide operations and remains unchanged. Scalar
QCO qubits, linear def-use chains, and positional unitary wire identity
(including SWAP) are documented requirements. Generic calls and unknown carriers
remain boundaries.

The converter removes three bookkeeping maps: structured capture order now lives
in one record before and after conversion, and modifier yields use their
preserved block arguments. The record moves only after region seeding, avoiding
references invalidated by map growth. Scalar capture filtering uses a vector
because the source capture set is already unique; register identities still need
deduplication. Two total type helpers replace `TypeConverter` and its pattern
plumbing without broadening the supported type domain. Region-local live
mappings, preflight validation, source retention, and deferred erasure stay.

Independent review found that removing modifier bookkeeping also removed an
implicit rejection: a misplaced `qc.yield` can verify, then assert while
resolving classical block arguments. An explicit converted-modifier parent check
preserves graceful rejection. The frozen baseline exits 1, the intermediate
prototype aborts, and the final CLI exits 1 again. One small regression test
covers this; an existing wire-call test now checks two wires and a leading
classical operand. Final specialist reviews found no remaining issue.

This follow-up removes 73 net production lines, including 58 in `QCToQCO.cpp`.
The complete PR removes 219 production and 409 test lines against its refreshed
upstream base, retaining the shared 132 forward and 133 reverse conversion
cases.

## Measurements

The baseline is `217874933586cc93170f89e14cd96a6ecd2d2683`: the prior PR content
rebased onto upstream `3f1ccfd69b12f101587be35c9f5eb05c42adf211`. All variants
use Clang 23.1.4, LLVM/MLIR 23.1.0, release `-O3`, assertions, and ThinLTO on
DGX Spark. Five fresh-process rounds rotate variant order on CPU 0. The host and
core are shared, not reserved; timings are diagnostic. Our builds and test jobs
were stopped during measurement.

Wire timings exclude parsing and setup. Forward traversal includes allocations,
the unitary operation, and sinks; backward traversal starts at the final
produced qubits. All 90 samples agree on traversal counts. The following medians
are normalized to one traversal of all wires, or the entire single-qubit chain:

| Case                                | Baseline forward (ms) | Updated forward (ms) |
| ----------------------------------- | --------------------: | -------------------: |
| Barrier, 32 wires                   |              0.002107 |             0.001552 |
| Barrier, 512 wires                  |              0.211623 |             0.024890 |
| Barrier, 4,096 wires                |             12.365600 |             0.198579 |
| Barrier, 8,192 wires                |             48.888000 |             0.394025 |
| Unitary call, 32 wires              |              0.001459 |             0.001676 |
| Unitary call, 512 wires             |              0.060364 |             0.027184 |
| Unitary call, 4,096 wires           |              4.550105 |             0.219462 |
| Unitary call, 8,192 wires           |             19.587000 |             0.437436 |
| Single-qubit H chain, 100,000 gates |              2.722025 |             2.560005 |

The wide barrier/call improvements reach 124×/45×. This is not a universal
speedup: the 32-wire call is 15% slower (about 0.22 microseconds per traversal).
Backward chain time is effectively unchanged, 2.723 versus 2.769 ms.

The conversion/pipeline harness compares baseline, simplified converter and
iterator with direct callers, and the same code with restored range callers. It
uses the previous audit's frozen inputs and configuration. Output IR hashes and
gate counts must agree. RSS is the process peak including imported/copied IR,
sampled before serialization; it is not converter-only live memory.

All 105 conversion/pipeline samples agree on output IR hashes and gate counts.
Medians below compare the baseline with the final range variant.

| Case                | Baseline (ms) | Final (ms) | Baseline RSS (MiB) | Final RSS (MiB) |
| ------------------- | ------------: | ---------: | -----------------: | --------------: |
| flat-1m             |       663.080 |    648.090 |             492.07 |          492.00 |
| register-loads-100k |       728.253 |    726.299 |             166.30 |          166.11 |
| sibling-loops-2k-32 |        90.842 |     88.852 |              59.27 |           57.59 |
| wide-helper-8192    |         2.578 |      2.553 |              17.90 |           17.54 |
| qft_n160            |       824.704 |    824.437 |             190.68 |          190.31 |
| qft_n18             |       181.577 |    181.105 |              23.35 |           23.23 |
| ising_n26           |        20.200 |     20.232 |              20.41 |           20.41 |

The sibling-loop case saves 1.69 MiB of peak RSS (2.8%). Other memory changes
are small. Conversion timing is essentially unchanged. Restoring the three range
callers changes the full QFT-18 median from 180.379 to 181.105 ms and Ising-26
from 20.210 to 20.232 ms: no material regression in these cases. The
full-pipeline measurements do not establish a general compiler speedup.

## Validation and artifacts

- Full release build and CTest: 4,015 tests pass, including 2,910 MLIR tests,
  with one expected `ScQDMIJobSpecificationTest.QueryJobId` skip.
- All 90 wire and 105 conversion/pipeline samples pass their output checks.
- The misplaced-yield CLI reproducer rejects cleanly, and its regression passes.
- MLIR reference generation and repository lint pass.
- Whole-file C++ lint passes for all 20 selected files, including the updated
  iterator test. Final rebuilt probes reproduce the sibling-loop and both
  full-pipeline output hashes and gate counts.

No Python, wheel, sanitizer, or native Windows runs were added for this
follow-up. The preceding portable death-test fix remains in the branch. Hosted
CI must validate the published head separately.

External artifacts are in
`/home/nvidia/.codex/experiments/core-qc-qco-combined-20261018/followup-20261019/`:
wire source and inputs, runner, manifests, raw samples, reproducer, build and
validation logs. Conversion probes and their compile/link commands are in the
sibling `measurements/` directory under the `followup-` labels. The
direct-caller variant uses the baseline versions of the three consumer files;
all other source matches the measured range variant. Final lint fixes relocate
static helpers and format initializers without changing the algorithm. No
benchmark harness or generated input is shipped.
