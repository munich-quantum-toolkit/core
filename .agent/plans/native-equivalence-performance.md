# Native verification: measured simplification

Status: implementation complete. The measured changes preserve the supported
contract and are published on `feat/native-equivalence` as signed commits.

## Scope and contracts

The reference implementation is signed commit
`3339265d5865d21a6526aef653d07e743475b518`, based on Core
`1d384a5d23aa1f02b26fd5684dab586ef26167fe`. This follow-up adds no features,
supported input shapes, platforms, dependencies or profiling infrastructure.

| Category                  | What must be preserved or may change                                                                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Required semantics        | Concrete binding scope; common free/initialized input domain; source/output identity; phase and approximation relations; workspace cleanliness; explicit unsupported/inconclusive outcomes; owned result evidence.                               |
| Explicit supported subset | Executed entry allocations, nonrecursive calls, concrete SCF, terminal measurements, no measurement-dependent control/reset, no approximate partial or general channel comparison. These are eligibility limits, not definitions of equivalence. |
| Incidental choices        | Materialized traces, checker order, default worker/sample limits, cache representation and repeated traversals. Changing these requires a concrete benefit and preservation of the contracts above.                                              |

## Retained changes and correctness

1. **Use the existing maximum-entry traversal for maximum-only matrix checks.**
   Terminal effects and clean-workspace checks previously computed and discarded
   a long-double Frobenius norm. Generalize the vector traversal over the node's
   edges. The recurrence is unchanged: multiply each incoming weight's magnitude
   by the maximum child entry; terminals contribute the weight magnitude. Keep
   all four matrix children, finite-value guards, cancellation checkpoints and
   incoming weights on cache hits. Each walk owns fresh PMR scratch and releases
   it before garbage collection. Coherent comparisons retain `MatrixNorms` and
   its stable Frobenius residual. A specialist challenged finite checks, skipped
   levels, cache identity/lifetime and workspace leakage before implementation.

2. **Check the existing outcome budget before constructing DDs.** The selected
   output count is already fixed after interface validation. If exhaustive
   enumeration exceeds the requested cap, construction cannot decide the
   relation within that cap. Keep the overflow-safe shift check, original
   selected-bit count, verdict and reason. Preserve earlier interface/label
   validation and forced-checker eligibility. No supported subset is enlarged.

3. **Keep trace diagnostics with their caller.** The single Ponytail review
   reproduced concurrent reset/unbound-argument checks capturing each other's
   errors through a shared MLIR context. `resolveExecution` now applies the
   existing Python wrapper's caller-thread filter. Other threads' handlers can
   receive their own diagnostics; there is no new shared state or lock.

## Evidence and dominant costs

The external evidence bundle retains the public C++ harness, exact generation
commands and IR, compiler/link commands, source/binary hashes, raw JSON and perf
profiles. The harness borrows `QCOProgram` owners through `mlir::verify`;
loading and full subprocess lifetime are recorded separately. Instrumentation
remains outside the repository. The corrected loader retains its JSON binding
object; earlier measurements with the temporary-lifetime defect are superseded.

Both final comparison binaries use that corrected loader and the final shared
runtime, including the diagnostic fix. Only the verifier algorithm differs:
reference `Verification.cpp` comes from `3339265d58`; the other is the refined
implementation. GCC Release uses `-O3` and LTO with LLVM/MLIR 23.1.0. Runs use
seven alternating-order pairs of fresh processes on two Cortex-X925 cores.
Affinity does not reserve the shared host. All 252 calls retain their verdicts;
forced-checker distances match exactly.

| Existing program / compilation artifact              | Executed gates before → after | Construction API median, reference → refined |
| ---------------------------------------------------- | ----------------------------: | -------------------------------------------: |
| Frozen three-qubit terminal SWAP elision             |                        10 → 8 |                             0.673 → 0.664 ms |
| Sparse nine-bit readout, three measured bits         |                       10 → 17 |                             1.010 → 0.976 ms |
| Core QFT, eight qubits, line routing                 |                      41 → 402 |                         127.310 → 130.520 ms |
| Core QAOA, six qubits, depth two, bound angles       |                      30 → 118 |                           91.589 → 75.443 ms |
| Existing nested-SCF compiler example, 32 outer trips |                       497 → 4 |                             0.934 → 0.925 ms |
| Deterministic dense two-qubit numerical synthesis    |                        1 → 12 |                             0.753 → 0.690 ms |

IR inspection confirms actual loop trips, static sites, layout permutations and
readout destinations. The QAOA parameters are gammas `[0.31, 0.67]` and betas
`[0.23, 0.41]`; compilation uses seed 42, one mapping trial and zero refinement
iterations. Dense synthesis uses NumPy RNG seed 20261025 and a QR unitary. The
exact matrices and emitted artifacts are retained in the external bundle.

The QAOA Construction paired wall ratios are 0.815–0.830; CPU medians are 91.600
→ 75.454 ms and full subprocess medians 97.845 → 81.834 ms. Auto takes 92.417 →
75.550 ms. The residual remains `6.106630846851082e-16`. QFT Construction CPU is
essentially unchanged, 127.325 → 127.518 ms; its wall ranges overlap. QFT Auto
remains about 22.2 ms. Tiny-case differences do not justify general speedup
claims or changing orchestration.

Instrumented phase timings explain the change, rather than establish its speed:
QAOA effects take 70.0 → 51.7 ms, while validation and both traces together take
about 0.21 ms. QFT's compiled-side DD construction takes about 119 ms of a 122
ms diagnostic call. Nested SCF tracing takes about 0.48 ms of a 0.99 ms call.
These are inclusive diagnostic timings; overlapping worker times must not be
summed. Instrumentation perturbs parallel and allocation-heavy execution.

Corrected uninstrumented QAOA profiles contain about 9k/8k samples without lost
samples. Baseline software quad multiplication accounts for 11.53% and addition
for 2.35%; those leaves disappear from the refined profile. DD node creation,
number lookup and multiplication remain dominant. Allocation itself, transpose
and root registration are not dominant. Some AArch64/LTO stacks do not unwind
fully; leaf attribution is available.

Memory quantities are distinct. Accepted QAOA still makes 79,668 DD backing
allocations, but cumulative requested bytes fall from 28,628,256 to 25,533,216.
Peak live accounted DD storage stays 20,394,544 bytes, with zero such storage
live after return. Median process peak RSS stays about 37.4 MB. Thus the change
reduces allocation volume, not observed accepted-case peak memory. RSS also
includes MLIR, libraries and allocator retention; DD backing counts are not
active-node counts or process-memory bounds.

With the original outcome cap set to one, QAOA takes 27.501 → 0.228 ms and
sparse readout 0.840 → 0.102 ms. Refined DD allocation is zero, replacing peaks
of 11,642,072 and 2,763,648 bytes respectively. Both remain inconclusive with
the same outcome-limit reason. QAOA subprocess time falls 33.426 → 5.099 ms and
process peak RSS about 28.86 → 18.62 MB.

## Rejected changes

| Proposal and expected effect                                           | Correctness risk / disproof                                                                                                                                                                       | Decision                                                   |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Hoist terminal adjoints to avoid repeated transposes                   | Existing DD cache already handles repeated roots; retaining both adjoints increases lifetime/memory and loses constant-output early exits. Profiles do not identify transpose as dominant.        | Do not implement.                                          |
| Cache gates or change scratch allocation to reduce repeated allocation | Requires more retained roots/budget machinery. Many small allocations did not imply allocator CPU dominance; profiles locate discarded arithmetic instead.                                        | Reuse the existing traversal; add no cache/infrastructure. |
| Remove common constant bits from outcome enumeration                   | Safe only for equal constants on both sides, preserving labels, original cap and differing/variable/copied bits. Sparse effects cost only about 0.18 ms; the dominant QAOA case has no constants. | Defer.                                                     |
| Validate a shared containing module only once                          | Borrowed immutability makes this safe within one call. Representative before/after owners are distinct, and validation is a small fraction of dominant cases.                                     | Leave unchanged; do not add cross-call caching.            |

## Validation and remaining limits

- The new native outcome-budget regression fails on the reference: 918,968 DD
  bytes are allocated before rejection; a one-byte DD cap masks the outcome
  reason. It passes with preflight.
- The new public Python concurrent-diagnostic regression fails on the old wheel
  with an empty reset reason. It passes on the rebuilt wheel. The standalone
  reproducer failed four times; overlap uses bounded traces and a stagger, not a
  deterministic scheduler.
- All eight affected native suites pass: 2,134 tests. Existing independent
  dense, phase, permutation, sparse-output, workspace and numerical-boundary
  tests remain intact. The broadened Python suites pass 478 tests. Stubs
  regenerate without API changes. All 11 final native differential outcomes and
  distances match the pinned QCEC comparison record where its contracts overlap.
- Full-file C++ lint passes with zero findings across all 29 eligible files from
  the 42-path implementation diff; source hashes and the real index remain
  unchanged during the final run. The first pass's trailing-comma finding is
  fixed. Repository formatting and lint use `uvx nox -s lint`.
- Executable documentation and its internal-link check pass. Chrome DevTools
  inspection confirms rendered tables, formulas and examples, no console
  messages, and all 40 page requests successful.

No full-repository test pass, hosted CI result, hard RSS bound or universal
performance improvement is claimed. Further changes lack evidence sufficient to
justify their complexity. The
[extractable-PR ledger](native-equivalence-extractable-prs.md) includes the
caller-diagnostic fix; no separate PRs are submitted by this task.
