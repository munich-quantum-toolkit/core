# Topology-aware qubit placement

Status: implemented; maintenance checks complete.

## Scope and ownership

`PlacementPass` selects and materializes layouts; `MappingPass` routes them.
They share candidate scoring and the routing engine in `Mapping.cpp`.
`Mapping/Placement.h` owns bounded layout search, VF2 matching, and pattern
decomposition. The `patterns=none|small|extended` option is available through
pass pipelines and C++; high-level compilation uses `none`.

## Decisions and limits

- Try executable identity/path layouts before bounded search, then weighted
  greedy, identity, and random refinement on a miss. Search considers
  connectivity; native-cost scoring belongs to the router.
- Both solvers use 100,000 candidates. A pattern tuple can bind several
  vertices, so equal budgets are not equal work. Preparation and VF2
  decomposition are outside the budget. Failure to find a layout does not
  establish infeasibility.
- Direct search combines adjacency and injectivity constraints. Fixed ranges and
  quotas make results deterministic across thread counts; each task retains at
  most one result. Pattern search shares unused quotas in deterministic waves.
- Small patterns are pairs, four-cycles, and stars. Extended patterns add
  four-/six-vertex chains, six-/eight-cycles, and full 2×3/2×4 grids. Matching
  allows extra target edges and prefers arity, edge count, then pattern type.
- Target copies share lazy canonical caches. Extended groups are complete or
  unavailable after 2,000,000 probes or 65,536 occurrences; unavailable groups
  fall back to smaller selected patterns. Pair, four-cycle, and star groups are
  uncapped. Oriented indexes are built for each search.
- Prepare independent types in parallel outside `call_once`: nested executor
  work inside an initializer can re-enter it and deadlock.
- `mqt.placement_order` preserves logical tie-breaking across serialization.
  Routing validates and consumes it, then composes `mqt.layout`. Physical
  provenance must cover the target in site order. Target preparation rejects
  stale metadata; arbitrary pipelines invalidate it unless preservation is
  promised.
- High-level compilation routes newly placed inputs and preserves static native
  programs, including supported higher-arity gates. Standalone routing accepts
  static inputs. Adjacent interactions skip native-cost preparation.

## Evidence for the default

Historical measurements used 97 fixed workloads: 40 exported Benchpress graphs
and 57 synthetic cases, including chains, grids, and impossible queries. They
covered 1/2/4/8 workers, cold/warm caches, four budgets, seven samples after
warmup, and repeated default-budget runs on an Apple M1 Pro with Apple Clang 21
and LLVM/MLIR 23.1.0 release builds. Worker counts were not pinned cores. Every
returned layout was validated.

At budget 100,000, all methods found the 61 feasible fixtures. No method won
uniformly. At eight workers, geometric mean successful-query latency in the
selected synthetic group was 64.9 µs for direct, 50.2 µs for small patterns, and
81.9 µs for extended patterns. Extended patterns helped negative batches; direct
search helped successful batches. Those prototypes retained oriented indexes, so
their warm-cache results do not establish integrated compiler gains. An
experimental scheduler also lost two feasible hits at budget 10,000. Keep direct
search and its deterministic scheduler as the defaults.

## Validation

Prior validation recorded 754 passing release tests, two macOS `/dev/full`
skips, passing repository/C++ lint, and three CLI presets with unchanged output.
Maintenance baseline: 767 native tests passed, with two `/dev/full` skips. The
candidate mapping/compiler checks passed 467 tests with the same two skips.
Removed 256 duplicate runs from the eight-site embedding test: its search is
serial under either thread flag. All 128 Z-basis and 128 X-basis inputs and
their semantic oracles remain; separate multi-task tests cover thread
determinism.

Rejected hoisting invariant pattern counts out of the candidate loop. Three
alternating process pairs on the same 97 workloads, one worker and seven samples
after warmup, produced identical layouts but no useful speedup.
Candidate/baseline geometric ratios were 0.997/1.000 for small cold/warm and
0.999/0.997 for extended cold/warm. One zero-duration fixture was excluded from
these ratios and retained in sums. The unchanged direct-search control was
0.992. These are placement-helper measurements, not whole-compiler timings.
Production code is unchanged. Evidence:
`.benchmarks/open-pr-maintenance-20261028/pr2689/` in the Benchpress workspace.
Restored production passed 145 mapping and 322 compiler tests, with the same two
platform skips. Repository lint and the independent final review passed.

Retain the independent brute-force oracle over all 1,024 labelled five-vertex
queries on three targets, worker/budget checks, successful and impossible
searches, cache limits, empty/all-to-all/idle inputs, serialization,
sparse/spare sites, counted loops, invalid metadata, static native inputs, and
layout-aware semantics. Owning binaries are the mapping, compiler, MQT IR, and
QCO utility unit tests under `build/release/mlir/unittests`. See the root guide
for build and lint commands.
