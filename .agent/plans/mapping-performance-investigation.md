# Investigate place-and-route performance

Status: proposed. No measurements exist yet; every cost claim below is a
complexity argument from the source at `7a3e5f0`.

## Goal and scope

Reduce the compile time and memory of the `place-and-route` pass (`MappingPass`
in `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`) where measurements
show a real cost, without changing routing quality. The pass documentation in
`mlir/include/mqt/Dialect/QCO/Transforms/Passes.td` (`MappingPass`) is the
behavioral contract; tests live in
`mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`.

The architecture stays as it is. Native synthesis keeps ownership of the cost
model (`NativeCostTracker` and `NativeCostTable` in
`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp`). Cold
scoring and hot routing keep sharing one traversal. The complete-target routing
workspace remains. Changing the A\* heuristic, the trial structure, or the
window construction is out of scope unless a work item below says otherwise.

Every finding is multiplied by the number of traversals:
`ntrials * (2 * niterations + 1) + 1`. That is 41 with the defaults on a machine
with eight or fewer logical CPUs, and 321 on a 64-CPU machine.

## Decisions

- Items 1 to 3 must keep results identical. With fixed `--seed` and
  `--mapping-trials`, the mapped IR and the `num-inserted-swaps` statistic must
  match the baseline exactly. Any change in output means the change is wrong,
  not a quality trade-off.
- Measure before changing items 2 and 3. Keep a change only when the
  `place-and-route` pass time improves beyond run-to-run noise on at least one
  workload class, with no regression beyond noise elsewhere. Report medians of
  at least five runs. Item 1 is a known O(F²) to O(F) fix and may land on a
  correctness argument and unchanged output alone, but still record its effect.
- Keep harnesses, generated programs, profiles, and plots outside the
  repository, as required by the
  [performance evidence guidance](../../AGENTS.md#performance-evidence).
- Item 4 is a product decision for the maintainer. Collect the data; do not
  change defaults without approval.

## Baseline and workloads

Build `release` with frame pointers or debug info for profiling (local
configuration only; do not change `CMakePresets.json`). Generate inputs with
`build/release/mlir/tools/mqt-core bench generate` and compile them with a fixed
seed and trial count:

```bash
build/release/mlir/tools/mqt-core compile <input> \
  --target='#mqt.compilation_target<...>' --seed=7 --mapping-trials=8 \
  --mlir-timing --mlir-pass-statistics
```

Use `perf record -g` on the same command for hotspots.

Use two target classes. The square-grid builder near `MappingPassFixture` in
`test_mapping.cpp` shows how to build one.

- **Current-size:** about 150 sites (for example a 12x12 grid), with programs
  that use most of the device.
- **Wide target, small program:** about 1000 sites (for example a 32x32 grid),
  with programs of 10 to 30 qubits. This isolates costs that grow with the
  number of sites rather than with program width.

Cover flat circuits (QFT, QAOA, Grover) and circuits with mid-circuit
measurement and control flow (repeat-until-success, teleportation, Grover with
weak measurement). Run a native-cost target (explicit native operations with an
entangler basis) and a topology-only target, because native scoring changes
which costs dominate.

Record the baseline revision, compiler, CPU model, thread count, and option
values with every result.

## Work remaining

- [ ] **0. Baseline.** Produce pass time, peak RSS, and a `perf` profile for
      each workload above at the baseline revision. Rank the hot functions.
      Later items cite this profile, not the reasoning in this plan.

- [ ] **1. Batch frontier removal in `walkProgramGraph`.**
      `mlir/include/mqt/Dialect/QCO/Utils/Drivers.h` removes released operations
      one at a time with `frontier.erase(op)` on a `SmallMapVector`. Each erase
      is O(F): it shifts the vector and renumbers the map. A layer with F
      released operations therefore costs O(F²), and this runs twice per routing
      step (`advance` and `getWindow`). Advance the wires as now, then remove
      all released operations with one `frontier.remove_if(...)`. Order is kept,
      so output is unchanged. `walkProgramGraph` has no other callers.
      Acceptance: identical output on all workloads; pass time recorded on a
      wide flat circuit, where F is largest.

- [ ] **2. Per-step cost that grows with the number of sites.** Every layout
      covers the whole target (`Layout<QubitIndex>::identity(numSites())`).
      `Node::initializeChild` copies 2 * numSites indices per child.
      `SearchFrontier::push` and `pop` hash the full `getProgramToHardware()`
      array each time. `getWindow` copies every site cursor, and both it and
      `advance` scan all sites per step. Expansion only moves sites next to the
      leading gate's two qubits, and `h` reads at most 2 * (1 + nlookahead)
      positions. First confirm with the wide-target profile that node copies and
      hashing are significant. If so, try an incremental 64-bit (Zobrist) hash
      kept in `Node`, updated on each SWAP, with the full layout kept for
      equality checks in `SearchFrontier`. This keeps results identical. Reusing
      scratch buffers for the copied cursors in `getWindow` is a second, smaller
      option. Do not shrink the routing workspace or key the frontier on a
      projection of the layout; both change results and need a separate quality
      study. Acceptance: identical output; measurable gain on the wide-target
      class; no regression on the current-size class.

- [ ] **3. Share measurement classification across cold traversals.** `advance`
      creates a new `measurementRouting` cache on every call, so
      `measurementNeedsRouting` repeats its forward-slice analysis for the same
      measurements on every routing step and in every traversal. Cold traversals
      never change the IR, so the classification for the entry block and its
      regions can be computed once before `parallelForEach` in `generateLayout`
      and shared read-only by all trials. Hot routing must keep a fresh cache
      per call, because `place` rewrites control-flow operations and SWAP
      insertion changes uses. `hasPotentialClassicalAliases` already
      short-circuits the check; keep that behavior. First confirm with the
      profile on measurement-heavy workloads that `measurementNeedsRouting` or
      `getForwardSlice` is significant. Discard the item if it is not.
      Acceptance: identical output; gain on at least one measurement-heavy
      workload; the shared data is immutable during the parallel phase.

- [ ] **4. Data for the default trial count (maintainer decision).** The
      `ntrials` default is `max(8, logical CPUs)` (`MappingPass` options in
      `Passes.td`; `MappingOptions::trials` in
      `mlir/include/mqt/Compiler/CompilationOptions.h`). CPU time, output, and
      the peak search-memory cap (threads * `search-memory-limit`, 256 MiB each
      by default) all depend on the host. Random starts come from one seeded
      sequence, so more trials never give a worse score. Measure score, pass
      wall time, CPU time, and peak RSS for `--mapping-trials` of 8, 16, 32, and
      64 on a many-core machine, for both target classes. Report the point where
      the score stops improving. Present the options to the maintainer: keep the
      current default; use a fixed default and use threads only for speed; or
      make the memory limit a total budget across concurrent trials. Do not
      change code in this item.

- [ ] **5. Correct the routing description in `Passes.td`.** The `MappingPass`
      description says routing divides the circuit into layers and runs A\* per
      layer. In the code, each `search` makes only `window.front()` executable
      (`Node::isGoal`); the rest of the window is lookahead for the heuristic.
      The description also gives `g(n) = alpha * depth(n)`, but the code uses
      the accumulated cost, which differs from depth when native costs are on.
      Rewrite those paragraphs to match the code, following the
      [prose policy](../../docs/development.md#prose-and-terminology).
      Regenerate the MLIR reference with
      `cmake --build --preset release --target mlir-doc`. This item needs no
      measurement and can land on its own.

## Validation

Correctness for items 1 to 3:

```bash
cmake --build --preset release
build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping
ctest --preset release
```

The mapping suite includes `ProduceStableOutputForFixedSeed` and
`ReusePassAcrossTargetsAndSeeds`. Also run the mapping tests in a `debug` build,
where the assert in `MappingPass::runOnOperation` checks that cold scoring and
hot routing agree. Compare mapped IR hashes and SWAP counts against the baseline
for every workload in the cohort. Run `uvx nox -s cpp-lint` and
`uvx nox -s lint` before handoff.

Report for each kept change: workload, baseline revision, environment, median
pass time and spread, peak RSS, and the limits of the measurement.

## Outcome

Not started.
