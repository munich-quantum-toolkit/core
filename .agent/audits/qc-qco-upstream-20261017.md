# QC/QCO builder and conversion audit

Baseline: upstream `main` at `056bef24722dd0a1f0813f3f7b4197c10f8eb950`. Date:
2026-10-18. Status: fixes applied and locally validated. The initial audit patch
was carried forward from `6b6efeec9` without conflicts.

This record covers the audit before integration with PR #2716. See the
[combined prototype](qc-qco-combined-2716-20261018.md) for the current result.

Scope: QC/QCO program builders, both conversions, wire/tensor traversal,
function correspondence, and their tests and immediate consumers. Work started
from a clean isolated checkout. LLVM/MLIR 23.1 and Clang 23 were used.

## Findings and changes

1. **Nested QC allocations could crash conversion.** A valid `func.func` with an
   outer `qc.alloc` and a second allocation inside `scf.if` reached the
   `lookupMappedQubit` assertion through `mqt-cc --emit=qco` (exit 134).
   `validateSupportedInput` now requires quantum allocations in a function's
   entry block and diagnoses unsupported input before rewriting. A focused test
   covers scalar allocation, register allocation, and stack allocation, and
   checks that rejected IR is unchanged. The pass reference states the limit.

2. **While iterators assumed the wrong correspondence.** `WireIterator` used raw
   operand/result positions although classical loop state may change shape;
   `TensorIterator` walked the before-region and failed when a borrowed tensor
   passed through a helper call. The existing while test now exercises different
   classical positions, a scalar wire, two tensor sizes, and a helper call.
   Before the fix it selected a classical result as a qubit and then aborted.
   Both iterators now use the quantum subsequence correspondence established by
   `qco::verifyLinearity`; recursive tensor-body traversal is deleted.

3. **Builder bookkeeping allowed invalid quantum captures and escapes.** Tracked
   values could be consumed from the wrong region, and implicit cleanup could
   use extracted values after their region ended. Shared validation now checks
   the insertion region. Explicit argument/result transfers bypass that scope
   check while retaining ownership checks. Single-target modifiers reuse their
   range overloads. One focused rejection test covers scalar and tensor
   captures, extracted-value escape, and modifier capture.

4. **Tensor insertion did not enforce its existing slot contract.**
   `QTensorOps.td` requires reinsertion into the original register and slot. The
   program builder now rejects foreign/fresh scalars and unequal constant slots
   using its existing tracking. Public builder documentation states the
   contract. One rejection test covers these three cases. Dynamic index equality
   and disjoint outstanding extractions remain program preconditions; no second
   whole-IR slot analysis was added.

5. **QC finalization disposed returned allocations.** `finalize(returnValues)`
   now excludes directly returned scalar and register allocations from automatic
   disposal, matching the existing function-building behavior. The allocation
   order test now also checks retained return values. Returning loaded register
   elements does not acquire new alias-analysis support.

6. **Call-origin collection repeated argument scans.** `collectWireOrigins` now
   resolves a callee and collects quantum argument indices once per call, then
   maps matching trailing results. This preserves fresh leading results; tying
   every quantum operand to every quantum result would be incorrect. The
   existing tensor-lifetime recursion limit is documented and applies to depths
   greater than or equal to 64.

7. **Gate and measurement scans traversed rejected operations.** Rotation
   merging, single-qubit fusion, and measurement routing now inspect direct
   single-use consumers. They stop at the same boundaries without asking an
   iterator to translate loop or call results first. Fusion deletion follows the
   already-scanned chain backward. Existing boundary, synthesis, and routing
   tests cover these paths.

8. **Unfinished builders leaked their modules.** QC and QCO builders held raw
   module pointers without a destructor. OpenQASM emission failures and
   exhausted emission budgets can return before finalization; Qiskit import can
   also throw during construction. Both builders now use `OwningOpRef<ModuleOp>`
   and release ownership in `finalize()`. The owning handle also replaces the
   separate context sentinel. Builders are move-only; their borrowed MLIR
   context must outlive the builder and its IR. No in-tree callers require
   copying builders.

9. **Flat verification retained every wire origin unnecessarily.** The existing
   linear-use walk now detects quantum function arguments or non-module regions.
   Only these cases need the second positional walk and its origin map. Static
   placement, uniqueness, and every linear-use check still run. Public
   `collectWireOrigins` remains complete for conversion consumers; unitary
   correspondence uses its already-verified quantum views without copying them.

10. **Builder calls and tensor cleanup repeated bookkeeping.** Calls now pass
    mixed operands directly to `prepareInitArgs`, removing the quantum gather
    vector and scatter loop. Scalar-only boundaries no longer allocate a scalar
    membership set needed only for register alias checks. Implicit reinsertion
    builds its local tensor chain and updates tensor tracking once. Per-qubit
    validation, deterministic order, and slot ownership checks remain intact.

11. **The default pipeline repeated successful conversion validation.** QC,
    OpenQASM, and jeff conversion paths already end with QCO linearity
    validation. The immediate duplicate check is removed. Existing QCO inputs
    still get validated because the public API exposes mutable IR.

## Test reduction

The final source/test diff removes 617 net lines, including 632 test lines.

- Share 119 identical QC/QCO builder pairs between the two conversion suites.
  All 132 QC-to-QCO and 133 QCO-to-QC cases retain their names, builders, flags,
  and equivalence assertions. Independent mechanical comparison confirmed the
  case multisets. Only suite prefixes change to common/special cases.
- Remove the always-false reference-comparison skip flag.
- Remove two tensor scenarios that moved qubits between slots contrary to the
  documented contract. The builder rejection test retains the relevant negative
  check. Remove their obsolete fixture and swap-target option.
- Replace an invalid, self-comparing index-mismatch fixture with a distinct
  constant-index row in the existing adjacent-index test. Consolidate its three
  former tests into one four-row table. Equal constants, distinct constants,
  identical dynamic values, and potentially aliasing dynamic values each retain
  explicit folding expectations and now check linearity before and after.
- Extend existing tests for returned allocations and while traversal. Only the
  three focused builder/conversion rejection tests are new.

## Design limits and overlap

- Quantum values cross regions explicitly; classical captures remain supported.
- Positional quantum correspondence is a verified QCO contract. Historical
  scalar-permutation assumptions do not override the current verifier.
- General while crossing scans the signature in linear time. Mapping still needs
  that correspondence; no cache with IR-mutation invalidation was added. No
  end-to-end speedup is attributed to the iterator changes.
- The original audit left the QC-to-QCO source-order rewrite to PR #2716. The
  linked follow-up combines both. Direct QCO export redesign remains outside
  this builder/conversion audit.

## Performance follow-up

The baseline is the refreshed upstream commit plus the initial audit patch;
measurements compare findings 8–11 against that baseline. Both binaries use the
same release preset: Clang 23.1.4, portable LLVM/MLIR 23.1.0, `-O3`, ThinLTO,
and assertions. The DGX Spark probe was pinned to Cortex-X925 CPU 0, with seven
alternating baseline/optimized process pairs. The host and core were not
reserved. These are workload-specific local measurements.

The external C++ harness builds identical IR, runs ordinary verification outside
timing, and times three or five linearity checks per process. It covers 10,000,
100,000, and 1,000,000 gates over one or 128 wires, borrowed helper arguments,
structured regions, wide calls, and extracted-register cleanup. All completed
modules pass ordinary and linearity verification. Values below are medians.

| Workload and measurement                                  |   Baseline |    Updated |
| --------------------------------------------------------- | ---------: | ---------: |
| 1,000,000 flat gates, one wire: linearity verification    |  140.91 ms |   11.87 ms |
| Same process: peak resident memory                        | 209.07 MiB | 152.20 MiB |
| 1,000,000 flat gates, 128 wires: linearity verification   |  145.31 ms |   12.56 ms |
| 100,000 gates in borrowed helper, 128 wires: verification |   13.10 ms |   10.87 ms |
| 100,000 gates in a loop, 128 wires: verification          |   13.11 ms |   10.83 ms |
| 5,000 mixed classical/64-qubit calls: construction        |   43.29 ms |   35.01 ms |
| 100,000 extracted register slots: finalization            |   98.06 ms |   96.51 ms |

Tensor-finalization ranges overlap; no meaningful speedup is claimed for the
batched tracking change. Scalar gate construction was about 1–4% slower in these
runs, with overlapping ranges for the million-gate case; it is not a claimed
optimization. Iterator-returning tracking validators and additional caches were
left out: they need separate evidence to justify their extra API machinery.

Twenty abandoned 20,000-gate builders retained 51,214,288 bytes for QC and
57,614,416 bytes for QCO before the fix. Afterward the retained allocator bytes
were 4,496 and 5,120 respectively. Increasing to 100 abandoned builders left
those values unchanged. The context is warmed and remains alive during this
measurement; this checks cleanup rather than allocator RSS release.

Complete `mqt-cc --emit=qco` runs on 100,000-gate inputs showed much smaller
changes: QC input took 379.17 → 368.83 ms, and OpenQASM took 409.04 → 399.59 ms.
Peak memory was essentially unchanged. Baseline and updated output files had
identical SHA-256 hashes for each input format. These runs include parsing,
conversion, verification, formatting, and writing to `/dev/null`; they do not
establish a general compiler speedup.

Reproduction artifacts remain outside the repository in
`/home/nvidia/.codex/experiments/core-qc-qco-audit-20261017/performance-20261018/`:
`prior-audit.patch`, `probe.cpp`, `build-probe.py`, `run-probes.py`,
`measurements.csv`, `run-cli.py`, `cli-measurements.csv`, and
`environment.json`.

## Validation

- Full `release-clang-ipo` configure/build passed. One Clang frontend crash
  reported invalid bytes in an intact generated header; retry passed without a
  source workaround.
- `ctest --preset release-clang-ipo -j6 --output-on-failure`: 4,030 passed, one
  expected `ScQDMIJobSpecificationTest.QueryJobId` skip, no failures; 41.96
  seconds, including 2,925 MLIR tests and all 265 conversion cases.
- `uvx nox -s lint`: passed. Subsequent ownership documentation and this audit
  also passed their relevant hooks.
- `uvx nox -s cpp-lint`: passed all 19 eligible changed C++ files, with zero
  formatting or clang-tidy findings.
- `cmake --build --preset release-clang-ipo --target mlir-doc`: passed.
- External probes verify module cleanup, move-only builder types, successful IR
  verification, and byte-identical compiler output. No new repository tests were
  added for the performance follow-up. Existing tests cover non-positional
  function/region results, static/use-count failures, mixed calls, borrowed
  registers, implicit reinsertion, and mutable QCO pipeline inputs.
- Specialist design review and independent cross-review found no remaining
  correctness issue in ownership transfer, positional checks, or the removal of
  repeated validation.

No hosted CI or Python suite was run. Changes remain local; nothing was
committed or published.
