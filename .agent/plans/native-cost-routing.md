# Native cost in routing selection

Status: implementation, rebase, independent reviews, validation, and repeated
comparisons are complete. Changes are committed locally; no remote push was
performed.

## Scope and ownership

Rank completed candidates by estimated native two-qubit count, then maximum
block qubit-dependency depth, retaining candidate order on ties. Shared
read-only analysis in native synthesis owns support, direction, decomposition,
and fusion choices. Candidate scoring observes original gates and virtual SWAPs
without cloning modules or running synthesis passes. Final synthesis remains
authoritative.

Use a first-SWAP fusion discount in bounded A* only when standalone SWAP cost is
available and uniform. Forward/backward refinement keeps its existing distance
cost. Preserve topology-only routing, unsupported-cost fallback, seeds, trial
generation, deterministic traversal, and the distance-reducing fallback.

Counts are static across branch arms and loop bodies. Depth is a maximum within
blocks; it does not model classical scheduling or runtime control flow. Native
estimates do not simulate general canonicalization. No public option or general
cost-model framework is added.

## Bounded synthesis caches

Mapping prepares one immutable table of up to 1024 numerical counts from
original gates and constant two-qubit runs, in both operand orders and with
adjacent SWAP products. Each live routing tracker has 64 local fallback counts.
Synthesis retains up to 64 full decompositions per analysis. Keep the
most-recent query fast path and use a fingerprint before exact byte comparison.
Results remain bound to the entangler and compilation seed; unavailable results
remain unavailable. Target support and operand direction are checked before
lookup.

No experimental settings, profiling counters, approximate keys, persistent
caches, locks, or IR-handle caches enter production. Cache payload and indexing
are separate from search storage. The pass documentation records their bounds.

## Validation

Independent reviews checked MLIR ownership, parallel trials, structured regions,
and synthesis/routing contracts. A singleton native RXX(pi) exposed a mismatch:
emission preserved one gate while estimation resynthesized its matrix to zero.
The tracker now permits fusion only after another member joins the run,
including a constant single-qubit gate. The same condition prices a first-SWAP
discount. A regression compares emitted counts, singleton cost, and
trailing-gate fusion. The complexity review removed two repeated eligibility
checks and a local flag.

Performance validation caught a scalar byte loop in the standard-library range
comparison on the measured platform. Explicit byte spans with memcmp preserve
the evaluated comparison and pass lint without a suppression. All final
benchmark cohorts are rerun after this change.

The implementation is rebased onto upstream main `151a69f9f`, including Arena
search storage and the revised structured-control-flow contracts. Arena node
construction and reset carry the native cost and first-SWAP discount; root reuse
clears that cost. Layout dominance retains the signed accumulated cost.

The wider RUS cohort exposed a region-boundary routing defect. Physical output
realignment must also rebind while-body and region-result consumers to their
logical states. Cold scoring mirrors those permutations and stops at placement
boundaries. A coherent-state regression covers if/while/switch results, an
independent neighboring region, idle wires, native and topology-only targets,
zero and finite budgets, two seeds, and serial/parallel determinism. The full
complexity review removed the ready-region vector, duplicate suppression, and
sort; the router retains the earliest ready region and refreshes the frontier
after routing it. Traversal and lookahead share one block-order boundary cursor
to cover regions hidden behind earlier gates and keep later adjacent gates from
stalling the search. Both regressions fail against the prior fence.

All 123 mapping, 68 native-synthesis, 230 compiler, and 574 QCO IR tests pass.
Whole-file C++ lint, repository lint, and whitespace checks pass. The 52 small
semantic cases pass with 4096 shots. Full-width RUS sampling remains
inconclusive: the DD sampler fails even before routing at 120 qubits, and other
wide probes time out. Wide structured cases establish compilation and static
costs only.

The expanded sweep covers 499 inputs, ten families, five target architectures,
and widths through 150 qubits, with routing seeds 7 and 99. All 868 paired flat
successes are retained; the same 24 paired attempts exceed the unroll limit.
Total native two-qubit count falls 6.92%, two-qubit dependency depth falls
6.77%, and inserted SWAPs fall 1.87%. Native count improves on 528 pairs, ties
on 285, and regresses on 55. Original width and largest-input metrics match the
prior evaluation. All 45 largest-input pairs succeed, with native count down
5.90%.

All 106 structured attempts compile in the solution; upstream main compiles 20.
The paired static counts fall 15.96%, while static depth increases 7.64%. These
are body counts and maximum block depths, not execution-weighted costs.

Five process repeats on the original seven timing cases give a 3.04% increase in
summed median wall time, 36.72% in CPU time, and 3.18% in peak RSS. The added 82
paired timing configurations, with three repeats each, give increases of 5.16%,
24.19%, and 1.78%, respectively. Geometric mean wall-time increases are 12.95%
and 8.34%; the aggregate does not promise a small per-case overhead. Final
synthesis wall time falls 9.42% and 5.55%, while mapping grows 32.84% and
33.36%. These full-revision measurements do not isolate cache effects.

The largest decomposed Grover stress case, 256 iterations and twenty parallel
trials, increases median wall time from 3.71 to 4.01 seconds, CPU time from 8.50
to 10.19 seconds, and peak RSS from 682.55 to 694.17 MiB. Native count increases
1.56% and depth 11.86%. This individual loss remains visible beside the broad
quality gains. All resource-case output hashes agree across serial/parallel
execution and three repeats. The host is shared; repeated medians and ranges are
descriptive, not statistical guarantees.

Data, exclusions, source/binary hashes, per-case losses, and PNG/SVG plots are
in `/home/nvidia/.codex/experiments/pr2562-arena-comparison-20260922/`. The
report records all statuses and the limits of the semantic checks. Benchmark
artifacts remain outside the repository, and pre-existing untracked files are
preserved.
