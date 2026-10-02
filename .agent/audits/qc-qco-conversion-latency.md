# QC/QCO conversion latency

Status: implemented and locally validated; hosted CI and human review pending.
Baseline: upstream main `7e2a2679fd6c48397f2d7ca5e3d2018d841a69a2`. Date:
2026-09-08. No pre-existing production edits.

## Decisions

The conversion passes preserve reference/value semantics, QCO linearity,
positional quantum state correspondence, deterministic implicit sinks, static
reference coalescing, register alias validation, and actual register slot
stores. The implementation changes four costs without widening the supported
input:

1. `QCToQCO::ConvertQCDeallocOp` marks consumed entries empty in the ordered
   qubit map instead of calling the linear-time `MapVector::erase`. Lookups and
   returns treat empty entries as absent; implicit sinks skip them. Entries
   remain until region cleanup, so space is proportional to region allocations.
2. Both `ConvertFuncCallOp` patterns reuse a pass-owned `SymbolTableCollection`.
   Functions retain their identity and name while signatures change in place.
3. QC-to-QCO skips general rewrite/CSE normalization when no qubit registers
   exist and all static references are unique within their function and already
   in its entry block. Register inputs and static references needing hoisting or
   coalescing retain normalization. Validation still runs on both paths.
4. QC-to-QCO skips its second conversion when no structured quantum state was
   converted. Otherwise the separate terminator phase remains necessary to
   observe the final values produced by converted region bodies.

Sources: `mlir/lib/Conversion/QCToQCO/QCToQCO.cpp`,
`mlir/lib/Conversion/QCOToQC/QCOToQC.cpp`, and the three conversion suites under
`mlir/unittests/Conversion/`. The audit also traced compiler program conversion
entry points and pipelines. Open #2196 concerned builder function support and

## 2260 QC inspection; neither overlapped these conversion changes at audit time

### Evidence

The audit's isolated constant-time deallocation prototype changed QC-to-QCO
latency for 1,000/4,000/16,000 scalar allocations, H gates, and explicit ordered
deallocations from 4.682/43.126/559.798 ms to 2.700/11.278/48.678 ms. Cached
lookup with 1,000 helpers and 1,000 calls changed forward conversion from 10.821
to 5.302 ms and reverse conversion from 9.910 to 4.377 ms. Skipping
normalization reduced forward latency by 21–22% on flat scalar circuits.
Skipping the empty second phase saved about 1–2% on larger flat input.

Production build, QC-to-QCO-to-QC round trip:

| Input                                     |   Baseline | Implementation |
| ----------------------------------------- | ---------: | -------------: |
| 16 static qubits, 1,000 RY gates          |   2.479 ms |       2.166 ms |
| 16 static qubits, 10,000 RY gates         |  23.882 ms |      20.799 ms |
| Register of 16 qubits, 10,000 RY gates    |  50.861 ms |      50.141 ms |
| 1,000 helpers and 1,000 calls             |  21.573 ms |       9.174 ms |
| 16,000 scalar allocations/H/deallocations | 601.365 ms |      77.275 ms |

These are warm conversion timings, not whole-compilation improvements. The wide
case was repeated after a contended measurement window; the table uses the
repeat. Register-backed timing is approximately unchanged. Audit probes for
compact native GHZ/standard-QFT loops also showed no gain; semiclassical QFT
improved from 0.1476 to 0.1236 ms. Ratios near one are not gains.

Method: DGX Spark ARM64 core 19, Clang/LLVM/MLIR 23.1.0, release ThinLTO, mold
2.42.0. Five alternating baseline/variant process pairs, two warmups and five
timed repetitions per process; median of 25 samples per side. Parsing, context
setup, and cloning are excluded. Pass-manager creation and default verification
are included; round trips check QCO linearity between passes. Additional output
verification runs outside timing. Shared-host noise remains.

Ignored local reproduction artifacts are in `build/qc-qco-latency-audit/`:
`probe.cpp`, `generate.py`, `build_probe.py`, `link_tests.py`, `measure*.py`,
input IR, variant source copies, raw JSON samples, and direct-exit test logs.
The audit baseline and combined prototype each passed all 334 conversion and
round-trip tests. Production validation is recorded separately below.

### Deferred candidates

- No-rollback QCO-to-QC conversion passed 153 tests but varied by workload:
  register-backed 1,000/10,000-gate inputs were about 1–2% slower, while 100,000
  gates improved from 501.213 to 409.597 ms. It changes failure mechanics and
  needs failed-rewrite testing. Both production passes retain default rollback.
- Do not remove wire-origin proof, register alias handling, QTensor caches, or
  boundary verification. These protect semantics. A proven same-slot register
  fast path, lazy origin tracing, and specialized wrapper verification need
  separate measurements and correctness evidence.
- Preflight maps could be reused when normalization does not mutate IR. That
  follow-up is not part of this change or its measured gains.

### Production validation

The release ThinLTO build passed all 509 tests: 179 QC-to-QCO, 153 QCO-to-QC, 6
round-trip, and 171 compiler tests. Four new regressions cover mixed explicit
and implicit deallocation with an owned return, consumed borrowed arguments,
duplicate entry-block static references, and repeated calls before definitions.

`uvx nox -s lint` passed.
`uvx nox -s cpp-lint -- 7e2a2679fd6c48397f2d7ca5e3d2018d841a69a2` checked all
three changed C++ files in full and reported zero findings. The `mlir-doc`
target passed. Final-code latency samples are in
`measurements-implementation.json` and the repeated wide-case samples in
`measurements-implementation-wide.json`.
