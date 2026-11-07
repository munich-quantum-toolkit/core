# Investigate place-and-route performance

Status: complete. Trial defaults are unchanged; changing them remains a separate
maintainer decision.

## Outcome and scope

`MappingPass` retains the complete-target layout, existing A* heuristic, seeded
trial sequence, and native synthesis ownership of costs. The implementation
batches frontier removal, incrementally hashes search layouts, and shares safe
measurement classifications across cold traversals. Routing quality and default
options are unchanged. `Passes.td` now describes a search for the leading
interaction and accumulated routing cost, with the remaining window as
lookahead.

## Decisions

- `walkProgramGraph` advances released wires in the original order, marks their
  entries empty, then stably removes them together. This replaces quadratic
  vector shifting with one linear removal. The partial-release regression test
  covers a map-backed frontier and survivor/successor order; reordering the
  successors makes it fail.
- Each search node hashes the complete layout as an XOR of LLVM integer hashes
  of `(program, hardware)` pairs. A SWAP updates four terms. The cached hash
  replaces the stored depth field, preserving node size and the search budget;
  SWAP reconstruction follows parents and reverses the result. Frontier keys
  borrow immutable arena nodes. Different cached hashes reject unequal keys
  cheaply; matching hashes still require full-layout equality. Hash collisions
  cannot merge distinct layouts. The frontier is never iterated for ordering.
- Rejecting unequal cached hashes before reading layouts is necessary: an early
  prototype regressed QAOA by 26–29%. Profiling attributed about 63% of sampled
  performance-core cycles to equality probes; the SDK's enabled layout
  assertions scan the full permutation. Retain the assertions and avoid these
  unnecessary comparisons. Use LLVM's existing integer hash, not a custom mixer.
- Precompute only isolated measurements in blocks without potential classical
  aliases. Consecutive measurements share suffix state whose classification can
  depend on the starting measurement, so they retain the original per-advance
  cache. Cold trials read the shared map without mutations. Hot routing always
  uses fresh per-call classification because placement changes the IR.
- Baseline profiles justify both changes: wide-target QAOA spent about 24% of
  sampled cycles in full-layout hashing and 12% in `memcmp`; current-size QFT
  topology spent about 6% in frontier erasure, 6% in forward slicing, and 7% in
  the measurement-routing lambda. Percentages are exclusive samples weighted
  across ARM PMU groups, not summed percentages from separate events.
- Leave layout copies, cursor buffers, workspace width, trial structure, and
  heuristic unchanged. The measured improvements do not require broader changes.

## Measurement method and results

Baseline: `7cfa9b5e5c973c600f86532d12f62ea063d855d2`. Optimized native build:
Clang 23.1.6, LLVM/MLIR 23.1.0 with SDK assertions enabled, ThinLTO, mold, DGX
Spark ARM64. Measurements use ten Cortex-X925 cores (LLVM pool size verified as
ten), on a shared host without exclusive CPU reservation. No own build or test
jobs ran during timing. This is local evidence, not an isolated cross-platform
performance claim.

Use `mqt-core bench generate` and `mqt-core compile` with seed 7, eight trials,
two iterations, lookahead 20, and 256 MiB per-search memory. Targets are 12x12
and 32x32 grids, each with topology-only and explicit `u`/`cx` native costs. The
payload is OpenQASM 3.1 with control-flow and assignment capabilities. The CLI
has no timing option: temporary measurement builds enable the existing compiler
API timing/statistics options; these probes are excluded from the
implementation. `perf record` uses user cycles and DWARF call stacks.

Current-size inputs: QFT/QAOA 140 qubits, repeat-until-success 139 data qubits
plus ancilla, weak-measurement Grover 139 plus ancilla, Grover 62 (generator
maximum), teleportation three. Wide-target inputs respectively use 20, 20, 19+1,
10+1, 10, and three. QFT is flat, QAOA retains a counted loop,
repeat-until-success and weak Grover retain loops, and teleportation retains
conditional control flow; emitted IR was inspected.

Five paired runs alternate baseline/candidate order. All 240 successful runs
have byte-identical emitted QCO MLIR and identical SWAP counts within each case.
All 24 median pass times improve; only two of 120 pairs are slower (current-size
native weak Grover). Small/sub-millisecond differences are not strong evidence
on their own. Timings below describe the combined changes, not an additive
attribution to individual optimizations. RSS is process peak, including the rest
of compilation, not search allocation alone.

| Grid / workload / cost                  | Mapping seconds, baseline median [min–max] | Candidate median [min–max] | Peak RSS MiB, baseline / candidate maximum |
| --------------------------------------- | -----------------------------------------: | -------------------------: | -----------------------------------------: |
| 144 / grover-native                     |                     0.4463 [0.4442–0.7785] |     0.3401 [0.3316–0.4346] |                              132.6 / 132.2 |
| 144 / grover-topology                   |                     0.3412 [0.3391–0.5173] |     0.2285 [0.2194–0.3121] |                                79.1 / 73.8 |
| 144 / grover-weak-measurement-native    |                     0.4841 [0.4805–0.5496] |     0.4656 [0.4591–0.5888] |                              255.1 / 256.8 |
| 144 / grover-weak-measurement-topology  |                     0.5773 [0.5661–0.5876] |     0.5150 [0.4931–0.5545] |                              527.6 / 515.9 |
| 144 / qaoa-native                       |                     0.1570 [0.1440–0.1981] |     0.1273 [0.1122–0.1281] |                              169.6 / 166.6 |
| 144 / qaoa-topology                     |                     0.3166 [0.3155–0.3361] |     0.2251 [0.2084–0.2606] |                              237.9 / 232.5 |
| 144 / qft-native                        |                     2.2507 [2.2407–2.4855] |     0.8277 [0.8066–0.9050] |                              489.0 / 471.7 |
| 144 / qft-topology                      |                     3.4258 [3.3965–3.4546] |     1.1307 [1.1112–1.1390] |                              174.9 / 177.6 |
| 144 / repeat-until-success-native       |                     0.0513 [0.0506–0.0520] |     0.0384 [0.0378–0.0394] |                                74.5 / 72.7 |
| 144 / repeat-until-success-topology     |                     0.0432 [0.0429–0.0438] |     0.0310 [0.0302–0.0321] |                                66.0 / 66.0 |
| 144 / teleportation-native              |                     0.0013 [0.0013–0.0014] |     0.0012 [0.0012–0.0012] |                                55.9 / 55.6 |
| 144 / teleportation-topology            |                     0.0011 [0.0010–0.0020] |     0.0009 [0.0009–0.0009] |                                54.5 / 54.5 |
| 1024 / grover-native                    |                     0.6863 [0.6845–0.6947] |     0.1740 [0.1717–0.1850] |                              330.9 / 330.8 |
| 1024 / grover-topology                  |                     0.6810 [0.6771–0.6896] |     0.1732 [0.1570–0.1934] |                              330.3 / 329.4 |
| 1024 / grover-weak-measurement-native   |                     0.2501 [0.2379–0.2568] |     0.2006 [0.1782–0.2167] |                              349.4 / 341.9 |
| 1024 / grover-weak-measurement-topology |                     0.2379 [0.2269–0.2449] |     0.1645 [0.1632–0.1826] |                              362.8 / 359.0 |
| 1024 / qaoa-native                      |                     1.2045 [1.0565–1.3664] |     0.8011 [0.7480–0.9617] |                            1402.4 / 1398.0 |
| 1024 / qaoa-topology                    |                     1.2443 [1.1654–1.3021] |     0.8672 [0.7554–1.0610] |                            1397.1 / 1396.7 |
| 1024 / qft-native                       |                     0.3343 [0.3339–0.3346] |     0.0300 [0.0298–0.0490] |                                79.0 / 78.6 |
| 1024 / qft-topology                     |                     0.4003 [0.3963–0.4206] |     0.0950 [0.0894–0.0962] |                              212.3 / 211.5 |
| 1024 / repeat-until-success-native      |                     0.1066 [0.1059–0.1321] |     0.0464 [0.0454–0.1141] |                                89.3 / 90.3 |
| 1024 / repeat-until-success-topology    |                     0.1057 [0.1048–0.1352] |     0.0456 [0.0448–0.0676] |                                85.4 / 85.5 |
| 1024 / teleportation-native             |                     0.0284 [0.0279–0.0292] |     0.0208 [0.0194–0.0223] |                              112.8 / 111.5 |
| 1024 / teleportation-topology           |                     0.0307 [0.0291–0.0318] |     0.0250 [0.0241–0.0259] |                               125.9 / 96.7 |

Frontier batching alone, five alternating pairs on 140-qubit QFT, also preserves
all output bytes and SWAP counts:

| Cost     | Baseline seconds median [min–max] | Batch-only seconds median [min–max] | Peak RSS MiB maximum, baseline / batch |
| -------- | --------------------------------: | ----------------------------------: | -------------------------------------: |
| native   |            2.2534 [2.2445–2.2616] |              1.9836 [1.9740–2.0120] |                          489.0 / 489.2 |
| topology |            3.4074 [3.3988–3.4154] |              3.0107 [2.9944–3.0131] |                          174.9 / 175.0 |

## Trial-count decision remains with the maintainer

The default remains `max(8, logical CPUs)`. Five runs at each of 8, 16, 32, and
64 trials use the same four QAOA cases and seed 7. Score is the native cost/SWAP
pair; topology has no native cost and compares SWAPs. CPU time and RSS cover the
complete compiler process. This study measures ten concurrent workers, not 64
concurrent searches, and cannot establish a universal quality plateau from one
seed and workload family. No score improved beyond eight trials in this study;
increased trials consumed more CPU and generally more memory. Large timing
spreads reflect shared-host contention, including other compilation work during
the trial sweep.

| Sites / cost    | Trials | Score    | Mapping seconds median [min–max] | CPU seconds median | Peak RSS MiB maximum |
| --------------- | -----: | -------- | -------------------------------: | -----------------: | -------------------: |
| 144 / native    |      8 | 316 / 30 |           0.1029 [0.1024–0.4425] |               0.41 |                166.6 |
| 144 / native    |     16 | 316 / 30 |           0.7530 [0.6956–0.9758] |               1.36 |                455.7 |
| 144 / native    |     32 | 316 / 30 |           0.8230 [0.6951–1.8861] |               2.80 |                770.7 |
| 144 / native    |     64 | 316 / 30 |           0.9068 [0.7985–2.0027] |               4.54 |               1098.2 |
| 144 / topology  |      8 | — / 12   |           0.2304 [0.1996–1.0925] |               0.51 |                230.5 |
| 144 / topology  |     16 | — / 12   |           0.2670 [0.2047–0.2854] |               0.84 |                273.5 |
| 144 / topology  |     32 | — / 12   |           0.6057 [0.5186–1.3028] |               2.03 |                573.0 |
| 144 / topology  |     64 | — / 12   |           0.8561 [0.5387–2.9675] |               3.41 |                619.4 |
| 1024 / native   |      8 | 52 / 12  |           0.8819 [0.8771–3.0715] |               4.00 |               1401.4 |
| 1024 / native   |     16 | 52 / 12  |           1.4764 [1.3394–1.8554] |               8.48 |               2679.7 |
| 1024 / native   |     32 | 52 / 12  |           2.5657 [2.3677–5.5069] |              18.95 |               2684.5 |
| 1024 / native   |     64 | 52 / 12  |           4.5103 [4.3118–7.3931] |              31.93 |               2688.6 |
| 1024 / topology |      8 | — / 4    |           0.9358 [0.8192–1.6334] |               3.41 |               1398.6 |
| 1024 / topology |     16 | — / 4    |           1.4237 [1.3583–2.5150] |               8.84 |               2674.2 |
| 1024 / topology |     32 | — / 4    |           2.6558 [2.4205–4.2292] |              17.93 |               2678.6 |
| 1024 / topology |     64 | — / 4    |          5.0669 [4.1928–11.5864] |              32.76 |               2682.6 |

Options remain: keep the host-dependent default; choose a fixed trial default
and use threads only for speed; or define a total search-memory budget shared by
concurrent trials. The latter changes per-trial budgets and can affect routing
quality. None is implemented without a separate maintainer decision.

## Validation

Validation base: `49466928bf77d4df7fe8090bfb9792c2e3abddf1`. The benchmark
baseline above is separate from this validation base.

Final uninstrumented source passes:

- `cmake --build --preset release-clang-ipo --target all mlir-doc` and
  `ctest --preset release-clang-ipo`: all 4,348 tests pass. The generated
  mapping reference contains the corrected leading-interaction/cost description.
- Debug mapping suite: all 131 tests pass, including cold/hot score assertions,
  fixed-seed stability, pass reuse, aliases, and consecutive measurements. All
  six driver tests pass, including partial-release order.
- `uvx nox -s cpp-lint`: passes for the complete changed C++ files and affected
  headers. `uvx nox -s lint` also passes.

No diagnostic timing or score-printing changes remain in production code.

External evidence includes generated inputs, exact command arrays, compiler and
CPU metadata, all raw profiles and outputs, per-run timing/RSS/CPU measurements,
and hash probes. Independent hash checks cover all 40,320 eight-site
permutations and 10,000 random SWAP updates each at 24, 144, and 1024 sites;
incremental and recomputed hashes agree. Full equality remains necessary for
collisions. A size/alignment check confirms the node budget is unchanged.

Limits: shared ARM64 host, LLVM SDK assertions enabled, single seed, synthetic
grid targets, bounded generator widths, and QAOA-only trial-count study. Hosted
CI and Windows were not run. Harnesses and generated artifacts remain outside
the repository according to the performance-evidence policy.
