# Native cost in routing selection

Status: complete. Shared analysis and routing are implemented and rebased on
upstream main `949bf5cf6`, including Arena storage from #2598 and the DD fix
from #2606. Cache tuning, independent review, and the comparison were completed
before this final rebase; their measured baseline remains `e82bb0f0e` and their
measured implementation is `0e78f1b59`. The final publication review found no
further safe simplifications. After rebasing, all 996 affected tests, whole-file
C++ lint, and repository lint pass again.

## Scope and ownership

Native synthesis owns native support, operand direction, numerical
decomposition, and fusion decisions. Routing uses those read-only decisions for
candidate selection and bounded first-SWAP guidance. It ranks native two-qubit
count, then maximum block qubit-dependency depth, then existing candidate order.
Unknown costs fall back to SWAP count after successful native scores. Final
synthesis remains authoritative. Only the winning route is materialized.

Counts include branch reconciliation and loop restoration, with each body
counted statically. Depth does not model classical scheduling or runtime control
flow. Forward/backward layout refinement, trial generation, seeds, options,
bounded Arena storage, and distance-reducing fallback retain their contracts.

## Current simplifications

- Skip native-cost preparation when greedy placement needs no routing.
- Stop shared precomputation once its bounded table is full.
- Keep one full decomposition record per cache entry; the most recent query uses
  an index instead of copying its matrix, entangler, and decomposition.
- Keep exact matrix keys, seed binding, unavailable results, and per-traversal
  caches. No experimental controls, counters, locks, or approximate keys enter
  production.

Native count remains primary. Calibration-aware fidelity and timed success
probability need their own validated inputs; no speculative scoring framework is
added here.

## Experiments and validation

- [x] Publish the independent wide-RUS DD addition fix as #2606. Native DD and
      QCO utility tests, 54/120/150-qubit sampling, and lint pass.
- [x] Consult an independent synthesis/routing specialist with Matthias's
      original comments and TZAP. Review cache lifetime and numerical contracts.
- [x] Compare native guidance on/off across fourteen inputs, ten families, five
      seeds, and 4/20/40/80 trials: all 560 compilations succeed. At twenty
      trials, guidance reduces aggregate native count by 1.13%. Selected
      estimates match emitted count and depth. More trials improve minima, but
      do not uniformly narrow seed spread. Preserve guidance and the existing
      trial default.
- [x] Compare shared/local/full cache sizes and record representation across 732
      process measurements. All cache pairs preserve exact output hashes and
      stage metrics. Keep capacities 1024/64/64; alternatives do not win
      consistently. The full-decomposition index removes duplicate state, but
      the measurements do not establish a uniform speedup.
- [x] Run final mapping, synthesis, compiler, and QCO IR tests: 996 pass.
      Whole-file C++ lint and repository lint pass. Independent cache review and
      the full complexity review leave no further scoped findings.
- [x] Regenerate width, largest-input, structured, semantic, timing, and Grover
      scaling comparisons. The combined flat cohort has 868 paired successes, no
      lost baseline compilation successes, 6.92% fewer native two-qubit gates,
      6.77% lower depth, and 1.87% fewer SWAPs. Native wins/ties/losses are
      528/285/55; QPE accounts for 32 losses.
- [x] Validate all 52 small semantic cases and ten full-capacity RUS solution
      outputs. The independent DD sampler finds six incorrect baseline RUS
      outputs. Excluding these leaves fourteen paired structured cases with a
      0.40% native-count reduction. Compilation-only counts are not quality
      evidence for an incorrect baseline output.
- [x] Repeat dedicated timing with five process repeats for seven original cases
      and three for the expanded sweep. Across 82 successful expanded
      configurations, native count falls 7.20%, while summed median wall time,
      CPU, and peak RSS rise 4.05%, 23.66%, and 2.09%. The original seven cases
      have 6.12% fewer native gates and 8.53% more wall time. These are
      descriptive shared-host measurements, not uniform improvements.
- [x] Repeat the 16/64/256-iteration Grover resource grid with four/twenty
      trials and serial/parallel execution. All repeat and serial/parallel
      output hashes agree. The largest parallel case has unchanged median wall
      time, but 15.02% more CPU, 1.49% more RSS, and 1.56% more native gates.
      This individual regression remains in the report.

The timing guard checks build/test tools and native executables in other
experiment directories. Provisional batches with incomplete interference
detection remain archived and are excluded from the final timing summaries. The
final publication rebase adds only the upstream DD fix, outside routing and
synthesis. Benchmark plots retain their measured revisions; they are not
presented as new measurements of the rebased commit.

Artifacts remain outside the repository at
`/home/nvidia/.codex/experiments/pr2562-uniform-routing-20260922/`. The prior
completed comparison is preserved under `pr2562-arena-comparison-20260922`.
Existing untracked work is checked against the starting hash manifest.
