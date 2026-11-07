# Canonicalization cost in the compiler pipeline

Status: investigated, implemented and validated. Baseline revision:
`1a6b190719313031df674b50f183f80f7dd3069a`.

## Goal and scope

Reduce canonicalization time on compiler entry points while preserving program
semantics, and repair the modifier merge/unroll cycle and QTensor false-success
rewrite. Equivalent IR forms may change. Preserve failed-match immutability,
verified output, deterministic ordering, quantum gate sequences during
structural cleanup, and existing numerical semantics. Conversion patterns and
boundary verification costs are excluded.

Read the prior
[performance audit](../audits/canonicalization-performance-2253.md),
[dialect audit](../audits/dialect-canonicalization.md), its
[probes](../audits/dialect-canonicalization-probes.md), and
[conversion audit](../audits/qc-qco-conversion-latency.md).

## Outcome

- Fix Ctrl optimization/lowering mixing in multi-control decomposition and false
  rewrite success in graph-region tensor commuting.
- Use default canonicalizers at both previously unlimited compiler sites. QCO
  cleanup enables only CSE between iterations; other options use defaults.
- Batch consecutive identity tensor extract/insert pairs in one rewrite.
- Fold constant inverse angles and power phases when created. Isolated
  inverse-RX lowering uses less CPU; whole-compiler wall-time improvement is not
  established on the contended host.
- Retain full promotion/binding canonicalizers and both target cleanup runs
  because reduced pipelines have concrete correctness counterexamples. Retain
  allocation cloning because removal had no consistent measured gain.

Cleanup's unlimited iterations originated in `b3e613fa89` (#1987), when tensor
commuting was pairwise. Native synthesis inherited unlimited cleanup in
`34bd0e98d6`. Batched commuting now converges within the measured bound while
preserving mapping and scalarization's required tensor normal form.

## Measurements

Harnesses, inputs, raw samples, output snapshots and counterexamples remain
outside the repository. Baseline and candidate use the same build settings and
input programs.

### Baseline observations

The native build and editable package use Clang 23.1.6, LLVM/MLIR 23.1.0
(assertions enabled), the `release-clang-ipo` preset, ThinLTO and mold 3.0.0 on
DGX Spark ARM64. The workload sizes are GHZ 5/256, QFT 5/32, Grover 5 qubits
with 2 iterations / 8 with 12, QAOA 5 qubits at depth 1 / 64 at depth 8, and
Shor moduli 3/15. QAOA binds all gamma/beta arrays to 0.31/0.17. Generated
inputs contain classical registers and loops. Large logical circuits may remain
compact loops on untargeted paths. The explicit target is a line with
unrestricted native operations, seed 7 and 8 mapping trials. Specifying trials
matters: its default depends on available CPUs and otherwise changes target
output across CPU-affinity configurations.

[MLIR 23 canonicalizer initialization](https://github.com/llvm/llvm-project/blob/release/23.x/mlir/lib/Transforms/Canonicalizer.cpp)
collects dialect and registered-operation patterns and freezes them.
[Dynamic pipeline initialization](https://github.com/llvm/llvm-project/blob/release/23.x/mlir/lib/Pass/Pass.cpp)
checks the parent initialization generation: repeated calls within a generation
reuse initialization. Compiler contexts have 19 loaded dialects and 701
operations after dependent dialects load, rather than only the 15 registrations.

External action-handler tracing recorded these maximum iteration counts across
all ten inputs and three compiler paths, without patching MLIR or production:

| Previously unlimited run          | Maximum iterations |
| --------------------------------- | -----------------: |
| Default QCO cleanup 1 / cleanup 2 |              1 / 1 |
| Target preparation cleanup        |                  1 |
| Target cleanup after unrolling    |                  2 |
| Native synthesis cleanup          |                  2 |

All canonicalizers needed at most three iterations, including nested promotion.
Actual invocation counts are 5 for QCO, 13–15 for Adaptive QIR, and 8–68 for
target compilation (Shor repeatedly promotes while unrolling). Timing reports
aggregate repeated instances, so their row count is not the invocation count.
The limit bounds outer region sweeps, not successful rewrite cycles within a
sweep. Both formerly unlimited sites now use MLIR's default ten iterations. QCO
cleanup retains CSE between iterations; other options use factory defaults. A
compiler-corpus regression requests `test-convergence=true` for QCO cleanup.

### Correctness results

The unchanged two-control H/H decomposition timed out after ten seconds. The
one-control case produced one two-unitary Ctrl instead of two single-gate
operations. Using lowering patterns fixes both; the new regression passes. The
only other Ctrl optimization-pattern caller is SimplifyQuantum, which supplies
gate optimizations and retains its cancellation behavior. The graph-region
Insert pattern falsely returned success and failed a greedy run limited to two
iterations. Tracking actual forwarding changes fixes it; the regression checks
failed-match IR identity, convergence and verification. All seven tensor-pair
tests pass.

### Rejected narrow pattern sets

- Arith-only binding retains a dead `if (theta != 0) { ry(1/theta); }` branch
  after theta is bound to zero and then fails gate verification. Full
  canonicalization removes that branch successfully. Keep verification and the
  full canonicalizer; do not add a new partial constant evaluator.
- Arith/CBit-only promotion with region simplification disabled leaves an
  allocation, store and load whose index comes from a constant SCF branch.
  Existing promotion reduces the function to returning its input value. Even a
  later full canonicalizer leaves the memory operations, so adjacent cleanup
  does not recover identical output. This rejects item 4's proposed pattern set.
- Target synthesis needs its post-optimization canonicalizer: cancelling H/H
  inside an otherwise identity conditional leaves an empty QCO branch. Full
  canonicalization removes it before Base-QIR legality checks; omitting that run
  causes rejection. The earlier target-compilation run has a separate
  pre-unrolling dependency described below.

### Measurement limits

The host is running other builds and benchmarks. Wall samples vary by multiples,
not just a few percent. Keep raw samples and paired process-CPU diagnostics; do
not attribute that variation to this change. A 100-pair probe on a five-qubit,
straight-line star-GHZ program separates PM preparation from execution through
public pass instrumentation. The repeat measurement gives these medians in
microseconds:

| Measurement                     | Fresh PM | Reused PM |
| ------------------------------- | -------: | --------: |
| `run()` entry to first pass     |   21.416 |     0.352 |
| Pass execution and verification |   23.720 |    25.336 |
| Complete `run()`                |   45.425 |    25.736 |

The exact canonicalizer collection/freezing loop measured separately takes
20.408 microseconds. PM preparation also includes setup, so it is not labelled
pure initialization time. These measurements exclude parsing/context creation;
the host remains shared and they do not establish whole-compiler latency gains.

### Baseline timing table

Each cell is median wall milliseconds / summed canonicalizer milliseconds
(percent of reported stage time). Wall medians use nine warm samples; pass
figures come from a separate `CompilationOptions(enable_timing=True)` run,
summing Canonicalizer rows across reports. The reports round to 0.1 ms, and host
contention affects these shares. These are descriptive baselines, not
before/after speedup estimates. Parsing/generation precedes timing; compilation
uses `copy=True` and includes its copy.

| Workload     |       QCO optimized |          Adaptive QIR |        Explicit target |
| ------------ | ------------------: | --------------------: | ---------------------: |
| ghz-small    |   8.60 / 0.60 (26%) |    15.37 / 7.00 (45%) |      8.44 / 0.40 (20%) |
| ghz-large    |    1.19 / 0.10 (9%) |     26.66 / 0.30 (4%) |   151.05 / 20.50 (33%) |
| qft-small    |   6.72 / 0.70 (11%) |    26.32 / 4.30 (30%) |     17.72 / 5.20 (29%) |
| qft-large    |   6.69 / 0.60 (10%) |    31.16 / 1.10 (14%) |   260.75 / 47.10 (16%) |
| grover-small |  11.12 / 1.00 (20%) |    59.64 / 1.70 (16%) |     22.23 / 1.30 (11%) |
| grover-large |   5.75 / 2.50 (60%) |    20.08 / 4.20 (42%) |     39.31 / 5.60 (12%) |
| qaoa-small   |   3.03 / 0.60 (21%) |    13.76 / 0.90 (23%) |      2.38 / 0.70 (25%) |
| qaoa-large   |   1.27 / 0.50 (36%) |     7.99 / 0.60 (33%) |     275.69 / 3.20 (3%) |
| shor-small   | 49.31 / 23.70 (21%) |  211.88 / 20.50 (28%) |  777.78 / 207.50 (40%) |
| shor-large   | 91.63 / 24.80 (15%) | 125.55 / 118.40 (50%) | 1911.80 / 667.00 (30%) |

QAOA binding with `copy=True` takes median 0.113/0.133 ms for depth 1/8 (100
warm samples). The isolated pattern-collection cost is material relative to
binding, but the proposed arith-only replacement violates the branch contract
above. No caching infrastructure is introduced.

### Performance candidate decisions

- Keep eager constant folding in QC/QCO inverse modifiers and QCO power phase
  multiplication. Three alternating baseline/candidate process pairs, each with
  two warmups and five samples on CPU19, preserve exact output. Median
  process-CPU time across process medians for 1,000/4,000 inverse RX gates is
  2.785/13.630 ms before and 2.440/11.078 ms after. Corresponding wall medians
  are 11.779/57.335 ms and 11.573/59.022 ms: contention prevents a latency
  claim. This is an isolated lowering workload, not a whole-compiler speedup.
- Retain the canonicalizer immediately after SimplifyQuantum in target
  compilation. Skipping its execution preserved all ten benchmark outputs and
  reduced small-QAOA CPU time from 2.684 to 2.536 ms (all seven pairs favored
  omission, median paired saving 0.215 ms; wall medians 9.500/5.822 ms under
  contention). However, an independent counterexample overrides that result: a
  two-billion-iteration loop carrying a qubit with H/H in its body becomes an
  identity loop after quantum optimization. Without canonicalization, unrolling
  rejects it for exceeding its cloning budget; with canonicalization, the
  identity loop disappears and Base-QIR preparation succeeds. Preserve this
  pre-unrolling contract instead of changing the unroller.
- Retain the public default pipeline and the second compiler cleanup. Skipping
  its trailing optimization canonicalizer preserved all twenty QCO/QIR workload
  outputs, but paired timing gains were inconsistent and separating standalone
  and compiler contracts would require additional plumbing.
- Revert allocation-clone removal. Fresh-reset chains of length 1,000 had
  baseline/candidate CPU medians 6.329/6.455 ms; length 4,000 had 74.438/68.527
  ms, with one of three process pairs reversing the improvement. Wall medians
  were 42.329/39.409 ms and 446.056/425.952 ms. Exact output matched, but
  improvement was not consistent enough to retain the change. The identity-pair
  probe grows from 48.3 to 905.8 ms CPU for N=1,000 to N=4,000 (wall 229.8 to
  2,415.5 ms), consistent with quadratic rescanning; clone removal cannot fix
  this case because no reset is removed. General chain-scanning changes remain
  deferred.
- RemoveUnusedQubits was absent at the measurement baseline. It was restored
  upstream before publication; its folding behavior is unchanged by this patch.
  Optional identity-pattern relocation is omitted because its owner files need
  no other change.

## Validation on the measurement baseline

- Native `cmake --build --preset release-clang-ipo` and editable Python build
  pass with the measured implementation.
- `ctest --preset release-clang-ipo -j6`: all 4,269 tests pass, including
  sixteen compiler-corpus convergence cases and the two defect regressions.
- Focused QC/QCO modifier and numeric filters: 48 tests pass. All seven tensor
  pair tests pass, including failed-match immutability and finite convergence.
- `uv run --no-sync pytest test/python --maxprocesses=4`: 2,170 pass, two
  optional Catalyst tests skip.
- All thirty final QCO/QIR/target benchmark outputs are byte-identical to the
  baseline. Repeated outputs within every process are also identical. Wider
  tensor probes at N=1,000 and 4,000 converge with a finite ten-iteration limit.
- Independent review verified the fixes and rejected the target-pass deletion
  using the identity-loop counterexample. Both review findings are resolved.
- `uvx nox -s cpp-lint -- 1a6b190719313031df674b50f183f80f7dd3069a`: full-file
  checks of the changed C++ files pass with zero findings.
- `uvx nox -s lint`: all repository lint checks pass. No hosted CI or Windows
  run was performed during measurement.

## Publication validation

The first publication validation uses upstream main
`90dd625da3c873e8878aed2abc1d5a9ad815ee63`. Upstream changes to target
preparation, multi-control decomposition and unused-qubit cleanup are retained.
The timings and thirty-output equivalence comparison above apply to the stated
measurement baseline, not to this newer upstream revision.

The rebased native and editable Python builds pass. All 4,365 native tests and
2,258 Python tests pass; two optional Catalyst tests skip. All thirty benchmark
compilation paths and repeated parameter binding complete successfully.
Repository lint passes. Hosted CI and Windows validation are not part of these
local results.

Full-file C++ lint passes with zero findings. The final publication base is
`49466928bf77d4df7fe8090bfb9792c2e3abddf1`, which adds an unrelated QDMI adapter
change. The second rebase preserves the patch exactly (`git range-diff`).

Both builds pass on the final base. The final native suite passes all 4,365
tests; the final Python report records 2,268 passed and two optional Catalyst
skips.

## Algorithmic follow-up

### Chosen design

`FoldInsertAfterExtract` cancels a maximal consecutive identity chain in one
rewrite. It finds the head from any matched pair, then erases the pairs forward.
Each pair retains the existing tensor/scalar provenance and equal-index proof,
including equal dynamic SSA indices. Nonidentity uses, uncertain indices and
block boundaries stop extension. Graph-region order checks keep traversal
monotonic. Both operations of each pair are erased, preserving linearity before
the next rewrite.

This prevents the greedy driver from revisiting the allocation between every
pair cancellation. Finding and cancelling an N-pair chain takes O(N) work in
SSACFG regions, with no mutable cache, new analysis, pass or pipeline option.
The claim is specific to consecutive identity chains, not arbitrary tensor
graphs. Moving fresh-slot analysis into another pass would change standalone
canonicalization and introduce scheduling dependencies without helping this case
further.

Native synthesis uses `createCanonicalizerPass()` without custom options. QCO
cleanup uses `CanonicalizerPassOptions{.cseBetweenIterations = true}`; all other
options use the factory defaults, including Normal region simplification and ten
iterations. CSE remains necessary for duplicate index computations in the jeff
round-trip path. CSE before canonicalization misses equal nonlinear index
expressions exposed by folding; CSE afterwards leaves comparisons and branches
that require further folding. The existing fixed-point driver handles both
orders without another pass or an implicit dependency on folding inside
allocation cleanup. The two target post-optimization canonicalizers and full
promotion/binding canonicalizers retain their demonstrated semantic
dependencies.

### Measurements

Baseline: published commit `e7ec229c3e` with only the no-argument canonicalizer
changes. Candidate: batched identity cancellation. The isolated probe runs the
public `QCOProgram.run_pass_pipeline` with
`canonicalize{test-convergence=true}`; compiler pipeline ordering is not
involved in that comparison. Native extension builds use the toolchain and
preset stated above, on DGX Spark CPU 19. Three alternating baseline/candidate
process pairs each use two warmups and five timed runs; parsing and copying
precede timing.

| Identity pairs | Baseline CPU (ms) | Batched CPU (ms) | Baseline wall (ms) | Batched wall (ms) |
| -------------- | ----------------: | ---------------: | -----------------: | ----------------: |
| 1,000          |            43.069 |            1.158 |             43.069 |             1.158 |
| 4,000          |           732.188 |            4.975 |            732.338 |             4.975 |

Values are medians of process medians. The isolated workload improves about
37x/147x; increasing input size fourfold costs 17.0x before and 4.3x after.
Exact output agrees in every paired run. This removes the measured quadratic
identity-prefix behavior, not all tensor-chain costs. Representative GHZ, QFT,
Grover, QAOA and Shor paths are checked separately; no whole-compiler latency
improvement is claimed from this isolated result. Raw data remain outside the
repository in the existing evidence directory (`batch-pair-*`).

### Regression coverage

A direct-pattern test starts at the head, middle and tail and requires one
rewrite to cancel the full chain; it fails on the previous implementation. A
second test retains a changed-qubit barrier from either direction. Existing
equal-dynamic-index, uncertain-index, graph-region and immediate-linearity
checks remain.

A cleanup regression requires sharing of nonlinear index expressions exposed by
folding, then verifies QCO-to-QC conversion. The existing common-expression
regression checks folding after expression sharing. A graph-region identity
batch test checks that traversal stops at source-order boundaries while removing
the original root and preserving linearity.

Final validation: native and editable Python builds pass; all 4,369 native tests
and 2,268 Python tests pass, with two optional Catalyst skips. All thirty
representative compiler paths and repeated binding complete successfully; their
outputs also happen to match the previous samples. Full-file C++ lint and
repository lint pass. Hosted CI and Windows validation were not run for this
follow-up.
