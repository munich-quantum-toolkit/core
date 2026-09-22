# Native cost in routing selection

Status: in progress. Shared analysis and routing are implemented and rebased on
upstream main `e82bb0f0e`, including Arena storage from #2598. Final cache
tuning, validation, and the refreshed comparison remain.

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

## Experiments and remaining work

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
- [ ] Compare shared/local/full cache sizes and record representation. Require
      identical output. Exclude timing records that overlap other host work.
- [ ] Run final mapping, synthesis, compiler, QCO IR, C++ lint, and repository
      lint.
- [ ] Regenerate all width, largest-input, structured, semantic, timing, and
      Grover scaling comparisons against the same upstream base. Retain
      individual losses and use the separate fixed DD sampler for wide RUS
      validation.

Artifacts remain outside the repository at
`/home/nvidia/.codex/experiments/pr2562-uniform-routing-20260922/`. The prior
completed comparison is preserved under `pr2562-arena-comparison-20260922`.
Existing untracked work is checked against the starting hash manifest.
