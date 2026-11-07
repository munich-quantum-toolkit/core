# Decomposition and native-synthesis performance

Status: proposed. Nothing is measured yet. Investigations 1 to 4 below are open.

## Goal and scope

Find out where compile time goes in single-qubit, two-qubit, and
multi-controlled decomposition. Fix only the costs that measurements show to
matter. A source review found one proven redundancy and two likely
superlinear or repeated costs. This plan turns them into measured decisions.

In scope:

- `mlir/lib/Dialect/QCO/Transforms/Decomposition/` (`Euler.cpp`, `Weyl.cpp`,
  `BasisDecomposer.cpp`, `Pauli.cpp`, `DecomposeMultiControlled.cpp`)
- `mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/` (`TargetSynthesis.cpp`,
  `FuseSingleQubitUnitaryRuns.cpp`, `ZFramePropagation.cpp`) and
  `mlir/include/mqt/Dialect/QCO/Transforms/NativeSynthesis/NativeCost.h`
- The native-cost use in `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`

Out of scope: changes to synthesis quality (native gate counts), new passes or
options, routing heuristics, and new decomposition algorithms.

Fixed constraints:

- Output stays deterministic for a fixed `--seed`. Do not expose pointer or
  unordered traversal order.
- Native gate counts must not get worse on any measured workload. A change that
  improves speed but changes counts needs an explicit reason.
- Follow the performance evidence rules in the root `AGENTS.md`: report the
  workload, baseline revision, environment, measurements, and limits. Keep ad
  hoc harnesses, data, and plots outside the repository.

## Background

What already holds and needs no work:

- Single-qubit Euler synthesis (`synthesizeUnitary1QEuler`) is closed form and
  constant time.
- Fixed-basis two-qubit templates are built once per entangler in
  `cachedNativeBasisDecomposer`.
- Within one `NativeCostAnalysis`, `cachedDecompose` reuses the Weyl
  decomposition from cost selection for emission.
- `mergeDiagonalRotations` is a single linear sweep per block.

The expensive primitive is `TwoQubitWeylDecomposition::create`. It runs a
magic-basis transform, a 4x4 symmetric eigendecomposition with up to 100
perturbed retries, a product-gate factorization of both local factors, a
round-trip reconstruction check, and fidelity specialization. Cost queries
(`NativeCostAnalysis::coordinates`, `NativeCostTable::precompute`) call the full
`create` but keep only the three coordinates.

Two facts used below follow from `SWAP (A ⊗ B) SWAP = B ⊗ A` and from `SWAP`
commuting with the canonical gate `CAN(a, b, c)`:

- `M` and `M.reorderForQubits(1, 0)` have the same Weyl coordinates. Their
  local factors only swap (`k1l` with `k1r`, `k2l` with `k2r`).
- `M * SWAP` is the reordered form of `SWAP * M`, so both have the same
  coordinates.

## Investigations

Run the shared baseline in "Validation" first. Each investigation states the
decision it feeds. Stop an investigation when its criterion says so, and write
the measured result into this plan.

### 1. Redundant Weyl decompositions in `NativeCostTable::precompute`

Evidence (proven from source): `precompute` adds six keys per matrix through
`addOrientations`: `M`, `SWAP*M`, `M*SWAP`, and the same three for the
reordered matrix. Each key runs a full `create`. Only two distinct coordinate
triples exist, for `M` and for `SWAP*M`. The table holds at most 1024 entries,
so the redundancy also cuts coverage from about 512 runs to about 170 runs
before the walk stops.

Proposed change: in `addOrientations`, compute the coordinates of `M` and
`SWAP*M` once each. Store each result under its three keys. Keep the bitwise
keys and the capacity check unchanged.

Acceptance:

- `create` calls from `precompute` drop to at most two per distinct matrix.
- Native gate counts on the baseline workloads are unchanged.
- Existing tests in `mlir/unittests/Dialect/QCO/Transforms/Mapping/` and
  `mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/` pass.

Risk: derived keys now hold the coordinates of `M`, not values computed from the
permuted matrix. The two differ only by rounding, but a value near a count
boundary could change an entangler count. If a test changes, check whether the
new count is equally valid before you update the test.

Related, lower priority: `NativeCostAnalysis::candidates` calls `coordinates`
again for the reversed orientation. Deriving the reversed decomposition needs a
way to build a `TwoQubitWeylDecomposition` with swapped local factors, which is
a new API on that class. Do this only if investigation 2 shows that targets
with both entangler orientations on a site spend measurable time here.

### 2. Repeated Weyl work across mapping trials and traversals

Evidence (likely, unmeasured):

- `MappingPass` runs `ntrials` trials in parallel (default
  `max(8, logical CPUs)`). Each trial makes `2 * niterations + 1` traversals
  (default 5) using two `RoutingState`s: one for refinement and one for final
  scoring. Each state owns a `NativeCostTracker` with its own
  `NativeCostAnalysis`. The analysis keeps a 64-entry FIFO cache across the
  traversals of one state, and no cache is shared between trials. On a circuit
  with many more than 64 distinct runs, every traversal is therefore expected
  to miss again.
- The shared `NativeCostTable` matches keys bit for bit.
  `NativeCostTracker::appendPair` composes runs in a different order from
  `scanFusableTwoQubitRun`: backward traversals multiply on the other side, and
  single-qubit gates on the two wires join in program order, not wire by wire.
  Different floating-point rounding is therefore expected, and with it table
  misses for multi-gate runs.
- Each miss runs a full `create`, although the tracker needs only an entangler
  count.

Measure:

1. Add temporary counters for `create` calls by caller (`precompute`,
   `NativeCostAnalysis::coordinates`, `cachedDecompose`) and for hits in the
   table, `lastCoordinates_`, and the local cache. Keep them only if they become
   `Pass::Statistic` members of the owning pass and stay useful. Otherwise
   remove them before committing.
2. Run the baseline mapping workloads. Record calls per traversal, the hit
   rate of the shared table, and the share of `MappingPass` CPU time spent in
   `create` (from `perf`).

Decision:

- Continue only if `create` takes at least 20% of `MappingPass` CPU time on at
  least one baseline workload. Otherwise, record the numbers here and close
  this item.
- If you continue, prototype the cheapest option that fixes the measured
  cause, in this order:
  1. Key lookups so that order-equivalent compositions hit.
  2. A deterministic memo shared read-mostly across trials.
  3. A coordinates-only path that skips local-factor extraction.

  Option 3 must keep fidelity specialization and the failure behavior that
  `countUnitary2QWeyl` relies on. Verify on randomized unitaries that it returns
  the same counts as the full path before you adopt it.

### 3. Ancilla search in `DecomposeMultiControlled`

Evidence (the scaling follows from the code; the impact is unmeasured):

- For each candidate `CtrlOp`, `findAncillas` scans every block argument and
  every operation from the start of the block up to the candidate. It stops
  early only after finding `k - 2` clean ancillas, which is rare when all
  qubits are busy.
- For tensor registers, `ancillaSlots` walks the full extract/insert chain back
  to its allocation, and `ancillaState` walks it again per slot.
- Rewriting earlier candidates inserts operations into the same block, which
  can invalidate the block's operation order and force renumbering in
  `isBeforeInBlock`.

The expected total is O(candidates × block size) and up to
O(candidates × k × chain length).

Measure: generate `multiplexer` and `modular_multiplier` instances at
increasing sizes. Compile them with an all-to-all device and with a
nearest-neighbour device. Record the time spent in `DecomposeMultiControlled`.

Decision:

- Continue if doubling the number of multi-controlled gates more than triples
  the pass time, and the pass is a noticeable share of total compile time.
  Otherwise, record the result and close this item.
- If you continue, replace the per-candidate backward scan with one forward
  sweep per block. The sweep visits candidates in block order and keeps the
  live wire values and their clean or dirty state up to date. It must keep the
  current selection rule (clean first, then source order) so the output does
  not change. Add a scaling regression test only if it can run with a stable
  bound. Otherwise, document the measurement here.

### 4. Two-qubit runs costed again in each pass

Evidence (an open question, not a defect): `FuseTwoQubitGates` (before
mapping), the mapping cost table and trackers, and `TargetNativeSynthesis` each
build their own `NativeCostAnalysis`. They recompute Weyl data for largely the
same runs. These are not strict duplicates: after placement, coupling
direction, site-specific support, and `shrinkOnly` can change the answer.

Measure: use the per-caller counters from investigation 2 to report `create`
calls per pass.

Decision: propose sharing results between passes only if one pass repeats
work that another pass already did and that work is a significant share of
compile time. Any sharing must respect the inputs that differ after
placement. Record the decision either way.

### 5. Other profile hotspots

While profiling, check for two costs the source review did not examine:

- `SingleQubitRunFusion::apply`, which calls `applyOpPatternsGreedily` once
  per run.
- `propagateZFrames`.

Report any hotspot above 10% of `TargetNativeSynthesis` time. Do not change
them without a measured cause.

## Validation

Build and baseline, all commands from the repository root:

```bash
cmake --preset release
```

```bash
cmake --build --preset release
```

```bash
ctest --preset release
```

The CLI is `build/release/bin/mqt-core`. Use `mqt-core bench list` and
`mqt-core bench describe <name>` to find parameter ranges. Generate instances
outside the repository with
`mqt-core bench generate --instance-specification <spec.json> --format qc --output <dir>`.
A spec looks like this:

```json
{
  "schema_version": 1,
  "benchmark": "multiplexer",
  "parameters": { "qubits": 6 }
}
```

Baseline workloads:

- Mapping and synthesis: `qft`, `qaoa`, `grover`, and `modular_multiplier`,
  compiled with `--qdmi-device=mqt.sc.iqm.garnet`.
- Multi-controlled decomposition: `multiplexer` and `modular_multiplier`,
  compiled with `--qdmi-device=mqt.ddsim.default` (all-to-all) and with
  `--qdmi-device=mqt.sc.iqm.garnet`.

Run every compile with a fixed `--seed` and a fixed `--mapping-trials`. See
`mqt-core compile --help` for input and output options. The CLI registers
the pass-manager options, so `--mlir-pass-statistics` works. Per-pass
`--mlir-timing` was not found to be registered in
`mlir/tools/compile/Compile.cpp`; use `perf record -g` and wall-clock time
instead, or register timing as a separate, justified change.

For every change:

- Compare native gate counts and the emitted program before and after, at the
  same seed. Explain any difference.
- Run the focused unit test binaries for `Decomposition`, `NativeSynthesis`,
  and `Mapping` under `build/release/mlir/unittests/`. Then run
  `ctest --preset release` and `uvx nox -s cpp-lint`.
- Record the baseline revision, machine, workload sizes, and median of at
  least five runs for each timing claim.

Latest result: none. No measurements have been taken.
