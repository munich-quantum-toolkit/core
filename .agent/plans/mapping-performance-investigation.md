# Place-and-route dependency and performance analysis

Status: complete. Trial defaults remain unchanged.

## Outcome and scope

Mapping uses one immutable set of measurements required by logical quantum work.
It computes this set before placement and before the greedy-layout fast path,
and uses it for both cold candidate scoring and hot routing. Terminal
measurements remain pending while independent routing proceeds.

Semantic correctness, cold/hot score agreement, and routing quality are the
contracts. The mapped IR and SWAP count need not match an earlier
implementation. The complete-target workspace, A* heuristic, trial sequence,
defaults, and native synthesis ownership of costs remain unchanged.

## Dependency analysis

`walkOperationDependencies` in `mlir/lib/Dialect/QCO/Utils/Sorting.cpp` shares
SSA edges, nested captures, and ordered recursive memory effects with the
topological sorter. Potentially aliased storage shares an effect chain. Unknown
and value-less effects are not ordered; mapping validates its supported side
effects before using this analysis. Sorting retains its own ready queue and
pending counts, including repeated edges and deterministic discovery order.

For each block containing measurements, mapping builds predecessor lists and
walks backwards from operations consuming qubits. Measurements, sinks, and
validated tensor insertion tails are not starting points. Quantum region
terminators are starting points, so measurements cannot be deferred across those
boundaries. Measurement-to-measurement SSA edges propagate requirements through
consecutive runs without special cases. The traversal uses an explicit worklist,
not an assumption that current block order is topological.

The original measurements retain their identity during placement. Allocation
replacement and tensor-tail lowering leave measurements intact; composite
extension moves region bodies. SWAP insertion and workspace threading precede
pending operations. These rewrites preserve the logical classification, so
neither hot reanalysis nor a separate cold cache is needed. A future rewrite
that clones measurements or changes logical dependencies must transfer or
recompute the classification.

This replaces a classifier whose memory-order cutoff depended on its starting
measurement while its suffix cache was keyed only by measurement identity. An
external probe on valid, linear, alias-free IR demonstrated different answers
for the same measurements under different query orders. Rebuilding that cache
more broadly would preserve the ambiguity; dependency closure removes it.

The analysis visits each constructed dependency edge a bounded number of times,
plus the existing recursive effect collection. Temporary predecessor lists are
released per block; only the required-measurement set persists across trials.
Blocks without measurements need no graph.

## Other retained improvements

- `walkProgramGraph` advances released wires in their original order and removes
  frontier entries in one stable batch, avoiding repeated vector shifts.
- A search node hashes the complete layout using XORs of LLVM integer hashes of
  `(program, hardware)` pairs. A SWAP updates four terms. The hash replaces the
  stored depth field, preserving node size and the search-memory budget.
  Different cached hashes reject unequal keys cheaply; matching hashes still
  require full-layout equality. Keys borrow immutable arena nodes, and the
  frontier is not iterated for ordering.
- The routing reference describes making the leading interaction executable,
  with remaining interactions as lookahead and accumulated SWAP cost in `g(n)`.

## Validation

Validation base: `5be0e888565b4971f741cb31ef98dca3ff86016d`.

- The optimized native build and `ctest --preset release-clang-ipo` pass all
  4,350 tests.
- Debug binaries pass all 133 mapping tests and all 262 QCO utility tests.
- `uvx nox -s cpp-lint` checks complete changed C++ files and passes.
- Repository lint and MLIR reference generation pass.
- Final public-CLI smoke checks compile QFT and the 4,000-measurement chain,
  producing the same output as the measured implementation.

Windows and hosted CI were not run. The paired timing study below uses a
separate, explicitly recorded common base.

The focused regression keeps both measurements terminal when an earlier register
read controls an intervening operation. It fails against the prior
implementation at `d3487bfd73425217e1001fb940c3b496232fb829`. Region-extension
coverage checks measurement identity, executable output, and sampled results
through `for`, `while`, and `if`. Existing tests cover aliases, nested captures,
early/late feedback, tensor tails, cold/hot native scoring, and pass reuse.

External dependency-closure probes cover output-only runs, feedback from the
first or last measurement, and intervening control from an earlier register
read. The results are independent of query order. The implementation also
received an independent correctness and simplification review.

## Performance evidence

The logical-analysis comparison uses the prior PR head `d3487bfd73` as baseline
and the same compiler settings, programs, targets, seed, and explicit trial
count. It measures complete compiler wall time and peak process RSS, including
serialization. It does not attribute differences to MappingPass alone.

Five alternating pairs on ten Cortex-X925 workers, shared DGX Spark, seed 7,
eight trials, two refinement iterations, lookahead 20, and a 256 MiB search
budget produced identical output for all ten cases (100 successful runs). These
measurements use the common pre-rebase base `49466928bf`; the newer upstream
compile-CLI refactor is not included. Small differences are noise. The
long-chain inputs retain all 1,000 or 4,000 measurements in emitted IR. Peak
process RSS increased by about 24 MiB in two cases; it includes the entire
compiler and does not measure the analysis graph alone.

| Workload / cost                   | Baseline wall seconds, median [min–max] | Logical analysis, median [min–max] | Peak RSS MiB, baseline / candidate maximum |
| --------------------------------- | --------------------------------------: | ---------------------------------: | -----------------------------------------: |
| 12-qft-native                     |                  3.1313 [3.0852–3.1837] |             2.8480 [2.8219–3.0643] |                              472.2 / 472.2 |
| 12-qft-topology                   |                  1.7542 [1.6771–1.7703] |             1.0127 [0.9754–1.0536] |                              177.5 / 201.7 |
| 12-grover-weak-measurement-native |                  1.5930 [1.5873–1.7369] |             1.5994 [1.5933–1.7343] |                              256.5 / 279.3 |
| 12-repeat-until-success-native    |                  0.1294 [0.1283–0.1325] |             0.1313 [0.1288–0.1341] |                                72.4 / 72.4 |
| 32-qft-native                     |                  0.0683 [0.0673–0.0701] |             0.0679 [0.0665–0.0685] |                                78.4 / 78.3 |
| 32-qaoa-native                    |                  0.9426 [0.8993–0.9814] |             0.9423 [0.8993–0.9657] |                            1386.8 / 1383.0 |
| 32-repeat-until-success-native    |                  0.0725 [0.0720–0.0745] |             0.0725 [0.0707–0.0731] |                                88.8 / 92.9 |
| 32-grover-native                  |                  0.2387 [0.2273–0.2435] |             0.2281 [0.2261–0.2319] |                              330.5 / 330.6 |
| chain-1000                        |                  0.0525 [0.0516–0.0538] |             0.0521 [0.0510–0.0544] |                                59.7 / 58.4 |
| chain-4000                        |                  0.1714 [0.1698–0.1740] |             0.1718 [0.1680–0.1730] |                                78.0 / 75.2 |

The earlier five-pair study against `7cfa9b5e5c` motivated frontier batching and
incremental hashing: baseline profiles exposed repeated frontier erasure,
full-layout hashing/comparisons, and measurement forward slices. Batch removal
alone reduced 140-qubit QFT mapping medians from 2.2534 to 1.9836 seconds with
native costs and from 3.4074 to 3.0107 seconds with topology-only costs. Those
measurements used a shared DGX Spark, Clang 23.1.6, LLVM/MLIR 23.1.0 with SDK
assertions, ThinLTO, and mold. They are historical evidence for those changes,
not measurements of the final logical-analysis implementation.

Harnesses, inputs, raw profiles, outputs, and per-run measurements remain
outside the repository according to the performance-evidence policy.

## Trial-count decision

Defaults remain unchanged. The prior study used five runs at 8, 16, 32, and 64
trials on QAOA for 144- and 1024-site grids, with topology-only and native
costs, seed 7, and ten concurrent workers. No score improved beyond eight trials
in these cases. Native cost/SWAP scores were 316/30 and 52/12; topology-only
SWAP scores were 12 and 4. More trials consumed more CPU and generally more
memory. This single-seed, single-family study does not establish a universal
quality plateau.

Options remain keeping the host-dependent default, choosing a fixed trial count
with threads used only for speed, or sharing a total search-memory budget across
concurrent trials. The latter can affect routing quality. None is part of this
implementation. Cursor-buffer reuse and shrinking the routing workspace are also
excluded.
