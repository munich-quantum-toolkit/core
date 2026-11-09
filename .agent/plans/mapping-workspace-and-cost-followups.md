# Place-and-route follow-ups: workspace tokens and cost caches

Status: item 0 measured at `58a6541655` on 2026-11-09. Prioritize item 1; defer
items 2 and 3. Reprofile before designing item 4. No follow-up is implemented.
See the outcome below for evidence and limits.

## Goal and scope

Reduce search effort, memory, and SWAP overhead of the `place-and-route` pass
(`MappingPass` in `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`) on
targets with many more sites than program qubits, and reduce repeated numerical
work in native-cost scoring. The pass description in
`mlir/include/mqt/Dialect/QCO/Transforms/Passes.td` is the contract; tests live
in `mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`.

This plan follows
[mapping-performance-investigation.md](mapping-performance-investigation.md),
which is complete. Its contracts still hold: semantic correctness, cold/hot
score agreement, and routing quality. Mapped IR and SWAP counts need not match
an earlier implementation. Its exclusions also still hold: trial defaults stay
unchanged, and cursor-buffer reuse and shrinking the routing workspace are out
of scope.

## Background: workspace tokens

`Layout<QubitIndex>` is a full permutation over all target sites. Indices below
`Environment::numProgramQubits` are program qubits. The remaining indices are
workspace tokens: unused sites that routing may swap with program qubits. In hot
routing they are the `qco.static` qubits that `applyPlacement` creates and sinks
at the end of the entry block. Nothing reads or acts on a workspace token except
a SWAP, so its state cannot affect program results. Workspace tokens are
therefore interchangeable. The code treats them as distinct in two places:

- **Search state identity.** `SearchFrontier` deduplicates nodes by the full
  `Node::hardwareToProgram` permutation (`LayoutKeyInfo`). Two SWAP paths that
  move the front pair to the same sites through different free sites leave the
  workspace tokens in different places. They are distinct keys, so the search
  keeps both. On a grid with free space, the number of such paths grows
  exponentially with distance. Equal program positions have the same `h`, but
  path lengths and root-prefix adjustments can give different `g`. Retain the
  lowest-cost representative when merging these states.
- **Region reconciliation.** `restore`, `converge`, and `driveby` use
  `TokenSwapGraph`, whose `shouldAddEdge` moves every token, workspace included,
  to its exact destination. `Layout::operator==` also compares workspace labels,
  which affects the early exits in `restore` and `converge` and the
  refinement-convergence check in `generateLayout`. Example on a 2x2 square A-B
  / D-C: a loop body moves program qubit q around the square (swaps A-B, B-C,
  C-D, D-A) and back to A. Program positions are restored, but three workspace
  tokens are rotated, so `restore` adds two SWAPs on idle sites. Each costs a
  full native SWAP after synthesis.

Recorded evidence supports the first point but does not prove it. The study in
the predecessor plan shows 1,383 MiB peak RSS for `32-qaoa-native` on a
1024-site grid with eight trials on ten workers, against 78 MiB for
`32-qft-native` on the same target. The arena budget is
`searchMemoryLimit / (sizeof(Node) + 4 * nsites)`, about 63k nodes for 1024
sites at the default 256 MiB. Each node holds a 2 KiB `hardwareToProgram` copy,
so a full arena is about 145 MiB. Peak process RSS also includes other compiler
allocations; these figures do not establish how many arenas filled. The
`std::deque` in `Arena` keeps its peak size for the whole trial. Searches that
exhaust the budget and queued states fall back to shortest-path SWAPs, which can
cost quality.

## Decisions

- Workspace tokens are interchangeable. Items 1 and 2 rely on this. The
  complete-target workspace stays; only workspace label identity is dropped.
- Item 1 may change mapped IR. Accept it on quality and resources, not on output
  identity.
- Item 2 changes a documented contract ("loop bodies restore their entry
  layout") to "restore program-qubit positions". Update `Passes.td` in the same
  change.
- Item 3 must keep results identical: cache hits return bit-identical values
  because cache keys are exact matrices, entanglers, and seeds.
- Keep harnesses, inputs, and profiles outside the repository, as required by
  the [performance evidence guidance](../../AGENTS.md#performance-evidence).

## Work remaining

- [x] **0. Baseline and counters.** At the current `main`, rebuild the workload
  cohort from the predecessor plan: 12x12 (144-site) and 32x32 (1024-site)
  grids; QFT, QAOA, Grover, Grover with weak measurement, repeat-until-success,
  and long measurement chains; native-cost and topology-only targets. Use seed
  7, eight trials, and the default iterations, lookahead, and memory limit.
  Record wall time, peak RSS, and a `perf` profile. Add temporary counters (not
  committed unless item 5 keeps them):
  - generated candidates, accepted nodes, valid expansions, per-trial arena
    high-water marks, allocation-limit hits, and actual shortest-path fallbacks
    separately (a queued goal can succeed after the allocation limit is
    reached);
  - exact distinct full-permutation and program-position keys per search,
    keeping minimum costs and separating generated from accepted candidates;
  - Weyl coordinate computations (`weylCoordinates` and `coordinates` misses)
    per compile, split into mapping scoring and final synthesis;
  - SWAPs computed by `restore`, `converge`, and `driveby`, classified against
    the evolving layout before each SWAP. Separate cold traversal work from
    emitted hot boundary SWAPs; `driveby` only computes a candidate layout.

  Collect counters separately from clean timing/RSS runs: tracking distinct keys
  adds memory and hashing work. Pin the revision, inputs, build, CPU affinity,
  and worker count. Use the counters to order items 1 to 4; defer items without
  a material observed cost, distinguishing inconclusive evidence from evidence
  of negligible cost.

- [ ] **1. Key search states on program-qubit positions.** Change
      `LayoutKeyInfo` and the incremental hash in `Node::initializeChild` to use
      only `Node::programToHardware` (size `numProgramQubits`). The position
      updates already skip workspace tokens (the `pa`/`pb` bounds checks); apply
      the same bounds to the hash terms in `initializeChild` and
      `initializeRoot`. This merges states that differ only in workspace labels
      and makes equality checks O(program qubits). Expected effect: far fewer
      nodes, fewer budget fallbacks, lower RSS on wide targets; no effect when
      program qubits fill the target. Once keys no longer need it, consider a
      separate experiment dropping the per-node `hardwareToProgram` copy and
      finding the token at a neighbor site from the root inverse plus the node's
      own moves. Do that only if the profile still shows node copies. Keep the
      `Arena` budget formula unchanged so the node limit does not depend on node
      storage. Acceptance: no lost compilations; aggregate native two-qubit
      count and SWAP count no worse, separately for flat and structured inputs;
      list and explain each per-case regression; report node counts, fallbacks,
      wall time, and RSS against the baseline. Successor sites, goal tests, and
      the heuristic depend on program positions; the native prefix adjustment
      applies only at the root and later edges have uniform standalone cost.
      Check these assumptions when changing dominance. Retain the inverse
      permutation initially. Fewer nodes reduce retained arena storage only if
      the trial's peak allocation falls.

- [ ] **1b. Optional: break `f` ties in the frontier.** `CompareNodePointer`
      orders only by `f`. After item 1, measure whether equal-`f` plateaus still
      dominate expansions. If they do, prefer the node with the larger `cost`
      (smaller `h`) on ties, which is the standard A* tie-break. Accept on the
      same terms as item 1, measured separately.

- [ ] **2. Reconcile regions modulo workspace tokens.** Make `TokenSwapGraph` add
  edges only for program tokens, so `restore`, `converge`, and `driveby` move
  only program qubits. After reconciliation, relabel workspace tokens so the
  resulting layout equals the target layout. Then compare layouts on program
  positions in the `restore` and `converge` early exits and in the
  refinement-convergence check. Steps:
  - Establish termination for the modified algorithm, including `converge`'s
    moving destination. The colored-token result in arXiv:1602.05150v3 assigns
    destinations by minimum-cost matching before distinct-token routing; it does
    not directly justify deleting workspace edges. Do not carry over the
    four-approximation claim without a proof. Complement the argument with a
    small exhaustive check of connected graphs and partial placements for
    cycles, premature termination, and incorrect final program positions.
  - Normalize each caller's child layout after applying reconciliation SWAPs,
    including early exits. `restore` returns only SWAPs, so relabeling its local
    copy is insufficient. Change only workspace labels, keeping physical wire
    slots and program positions fixed. Relax the final assertion accordingly.
  - Check that hot routing yields per-site values, so relabeled workspace values
    stay valid (`realignQubitValues`, `realignQubitUses`, terminator updates in
    `routeComposite`).
  - Check that `mqt.layout` `routing` stays a complete permutation
    (`mlir/lib/Dialect/MQT/IR/QubitLayout.cpp`) and that its consumers read only
    program positions. Start with `ElideTerminalSwaps.cpp`,
    `ShrinkRegisters.cpp`, and `TargetCompilation.cpp`.
  - Update the `MappingPass` description in `Passes.td` and regenerate the MLIR
    reference with `cmake --build --preset release --target mlir-doc`.

  Acceptance: structured workloads emit no boundary SWAP that only exchanges
  workspace tokens; aggregate native two-qubit count no worse; sampled results
  of structured programs unchanged (the existing region-extension tests sample
  results through `for`, `while`, and `if`). Include nested regions and
  `IndexSwitchOp`, while-region handoff, result rewiring, and complete routing
  metadata. Add a focused regression test for the square-loop example above that
  asserts no boundary SWAP touches only workspace sites. Do not assert a total
  SWAP count.

- [ ] **3. Share native-cost numerical caches within a trial.** Every
      `RoutingState` constructs its own `NativeCostTracker`, and with it a fresh
      `NativeCostAnalysis` with 64-entry coordinate and decomposition caches.
      `NativeCostTracker::reset` already retains caches across refinement
      traversals within one state. Additional reuse opportunities are the final
      scoring state and every new child-region state in `routeComposite`. Let
      all trackers in one trial borrow one analysis, owned next to the trial's
      `Arena` in `generateLayout` (no cross-thread sharing). Keep pending runs,
      depth, cancellation state, and traversal direction separate for each
      tracker. Start at 64 entries: shared caches can also suffer eviction from
      interleaved regions. Measure hit rates and runtime at 64 entries and at a
      larger capacity before choosing one; justify any increase from the
      counters and record its memory cost in `Passes.td`, which currently
      documents 64 counts per live region. Keep `NativeCostTable` as the shared
      immutable layer. Acceptance: identical mapped IR and scores at fixed seed
      and trial count; fewer Weyl computations; mapping time reduced on
      native-cost workloads. Fewer misses alone do not establish a speedup.

- [ ] **4. Per-step program-graph walks (only if item 0 shows them).** Each
      routing step runs `walkProgramGraph` twice from scratch, in `advance` and
      `getWindow`. Each run visits every site cursor and reserves maps sized for
      half the target. If these walks remain a top cost after items 1 to 3,
      write a short design for keeping the frontier across steps, where a SWAP
      changes only two cursors, and bring it to the maintainer before
      implementing. It touches the shared driver in
      `mlir/include/mqt/Dialect/QCO/Utils/Drivers.h` and the release order that
      `NativeCostTracker::append` observes.

- [ ] **5. Decision for the maintainer: a fallback statistic.** The
      `search-memory-limit` option has no visible effect when it is reached. A
      permanent `num-search-fallbacks` pass statistic, counted in the final hot
      route only, would let users tune it. Propose it with item 0 data; add it
      only on approval. A hot-route-only count does not expose losing trial
      searches; state that limit in its description.

## Considered and not recommended

- Removing duplicate trial starts (greedy equal to identity): rare, and the
  duplicate only costs one trial.
- Aborting losing scoring traversals early across trials: saves at most one of
  `2 * niterations + 1` traversals per trial and needs shared mutable state.
- Replaying the winning cold route in hot routing: saves about one traversal out
  of `ntrials * (2 * niterations + 1) + 1`.
- Narrowing the `size_t` distance matrix in `CompilerTarget`: 8 MiB at 1024
  sites; reconsider only for much larger targets.
- Full-width composite extension in `place`: `32-repeat-until-success-native`
  compiles in about 73 ms in the recorded study, so there is no evidence of a
  cost.

## Validation

```bash
cmake --build --preset release-clang-ipo
build/release-clang-ipo/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping
ctest --preset release-clang-ipo
```

Run the mapping tests in a `debug` build as well, where the assert in
`MappingPass::runOnOperation` checks that cold scoring and hot routing agree.
For items 1, 1b, and 2, compare native two-qubit count, SWAP count, depth,
fallbacks, wall time, and RSS against the item 0 baseline for every cohort case.
Retain seed 7 for historical comparison and use a small additional seed set
before accepting heuristic changes. For item 3, compare mapped IR hashes. Run
`uvx nox -s cpp-lint` and `uvx nox -s lint` before handoff. Report each kept
change with its workload, baseline revision, environment, medians of at least
five runs with spread, and the measurement limits.

## Outcome

Item 0 ran at main revision `58a6541655c74eeaeeac64df94d9cd7d13c74af9`. The
branch containing this plan is separate from the measurement checkout. The
current evidence justifies implementing item 1 first; it does not establish an
implementation speedup or a routing-quality improvement.

### Measurements

The cohort reuses the predecessor's exact generated inputs: five workload
families on each of the 144-site and 1024-site grids, with native and topology
costs, plus the 1,000- and 4,000-measurement chains in both modes (24 cases).
The `12-` and `32-` prefixes denote grid side, not program width.

Five clean runs per case used seed 7, eight trials, two refinement iterations,
lookahead 20, and the default 256 MiB search budget. The build used
`release-clang-ipo`, Clang 23.1.6, LLVM/MLIR 23.1.0, ThinLTO, and mold. CPU
affinity was `5-9,15-19` on the shared DGX Spark. Other work was active; these
are shared-host baseline measurements, not isolated performance comparisons.

All 120 compiles succeeded and produced stable output hashes within each case.
All 133 mapping tests passed. A separate instrumented compile of each case
produced the same output hash as its clean baseline. Profiles and counter runs
were separate from clean timing/RSS runs. RSS is whole-process peak memory.

| Case                              | Program qubits | Wall seconds, median [min–max] | Maximum RSS MiB |
| --------------------------------- | -------------: | -----------------------------: | --------------: |
| 12-qft-native                     |            140 |            3.249 [3.017–3.944] |           428.6 |
| 12-qft-topology                   |            140 |            0.987 [0.958–1.181] |           169.5 |
| 12-grover-native                  |             62 |            1.204 [1.194–1.569] |           129.0 |
| 12-grover-weak-measurement-native |            141 |            1.640 [1.625–2.161] |           251.0 |
| 32-qaoa-native                    |             20 |            0.441 [0.405–0.563] |           765.4 |
| 32-qft-native                     |             20 |            0.065 [0.064–0.082] |            71.8 |
| 32-repeat-until-success-native    |             20 |            0.068 [0.067–0.110] |            78.3 |
| chain-4000-native                 |              3 |            0.175 [0.170–0.252] |            77.8 |

### Follow-up decisions

1. **Prioritize program-position search keys.** In `32-qaoa-native`, the eight
   trials created 1,646,386 candidates including roots, accepted 1,247,091
   nodes, and popped 873,246 valid nodes including goals. Summing exact distinct
   keys within each search gives 1,228,219 accepted full-layout keys versus
   142,182 program-position keys, an 8.64-fold difference. Five trial arenas
   reached 62,953 nodes; the other peaks were 29, 2,479, and 4,447. There were
   5,333,045 failed allocation attempts and 13 actual fallbacks. The hot route
   had no fallback. Topology-only QAOA had the same search counts. This is a
   material duplication and memory opportunity, not a predicted 8.64-fold
   speedup: merging states changes subsequent search. Near-full QFT had no
   workspace-key duplication. Keep inverse-layout storage in the first change.
2. **Defer workspace-aware reconciliation.** Every measured hot route had zero
   workspace-only boundary SWAPs. Cold trials did perform unnecessary loop
   reconciliation: `32-repeat-until-success-native` had 4,018 workspace-only
   swaps among 6,879 boundary swaps, and native wide QAOA had 2,032 among 7,765.
   The RUS compile was only 68 ms. This is real cold work, but does not
   establish a material emitted-circuit benefit. All `converge` and `driveby`
   counters were zero: branch reconciliation remains unmeasured. Reconsider
   after item 1 or with a representative branch workload showing a material
   cost.
3. **Defer trial-local cache ownership changes at capacity 64.** Across cold
   trials, native QFT made 1,366 coordinate computations; a shared FIFO shadow
   cache of 64 entries also missed 1,366 times. Native Grover gave 3,625 versus
   3,624; weak-measurement Grover gave 2,867 versus 2,627. Wide native QAOA had
   no mapping coordinate misses. RUS had larger relative reductions (86 to 50
   and 76 to 43), but small absolute counts and short compile times. The shadow
   uses exact matrix bytes and excludes immutable-table hits; it omits the
   analysis's extra last-coordinate slot, so these are estimates, not measured
   post-change miss counts. Capacity 256 reduced the estimates further (for
   example, native Grover to 1,247), but capacity and ownership are separate
   hypotheses. No cache change has demonstrated a runtime gain.
4. **Retain graph-frontier work as a later design investigation.** Direct
   `walkProgramGraph` symbols accounted for 25.32% of sampled CPU cycles in
   topology QFT and 13.04% in native QFT. Search/frontier symbols accounted for
   26.44% in wide native QAOA. These are symbol-attributed cycle samples, not
   wall-time savings or complete inclusive cost attribution. Reprofile after
   item 1 before proposing persistent frontier state. Keep item 1b separate.
5. **Do not add the proposed hot-only fallback statistic on this evidence.** The
   cohort had 30 cold fallbacks and zero hot fallbacks. A hot-only statistic
   would hide the observed problem. A trial-inclusive statistic would need an
   explicit scope decision and remains unimplemented.

### Evidence and limits

Local artifacts are under
`/home/nvidia/.codex/experiments/mapping-followups-20261109/`:
`environment.json` records configuration and input hashes; `commands.json`
records inputs and compiler arguments; `run.py` supplies affinity and resource
measurement. `baseline/runs.json` and `baseline/summary.json` contain all clean
runs. `diagnostic-summary.json` and `diagnostics/*.log` retain aggregated
per-search key counts, trial arena peaks, boundary classifications, and numeric
counts split between preparation/hot routing, cold trials, and synthesis.
`profile/*.perf`, `profile/*.flat.txt`, and `profile-summary.json` retain raw
samples and symbol attribution. `instrumentation.patch`, `probe.h`, and
`instrument.py` preserve the temporary probes outside the repository.

The independent counter review found no decisive instrumentation defect and
supported these priorities. Single-seed cohort results do not establish general
routing quality. Counter runs deliberately add expensive exact-key maps and
cache shadows; their runtime and RSS are not baseline measurements. Short-case
profiles have few samples, and kernel symbols were unavailable. No follow-up
implementation, Windows validation, full CTest suite, or hosted CI was run.
