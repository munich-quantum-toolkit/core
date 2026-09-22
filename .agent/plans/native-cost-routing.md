# Native cost in routing selection

Status: implementation and reviews complete; final resource measurements
pending.

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

The 121 mapping, 68 native-synthesis, 228 compiler, and 572 QCO IR tests pass.
Required C++ lint and supplemental synthesis-test/header checks pass. Cache
regressions cover eviction, numerical failure, seed binding, operand direction,
native bypass, and shared-table lifetime. Repository lint and diff whitespace
checks pass.

Against upstream main ee7edb68a, the 360 width pairs retain all 350 baseline
successes and the same ten unroll-limit failures. All 45 largest-input pairs
succeed. Total native gates fall 7.60% and 5.90%, respectively; total depth
falls 6.61% and 7.39%. All 27 semantic probes pass in both revisions at 4096
shots. The cache preserves the earlier redesign's measured routing and gate
metrics.

Five-process medians on seven timing cases show an 8.95% increase in summed wall
time and a 25.62% geometric mean increase. Final synthesis wall time falls 6.71%
by summed medians, but mapping remains more expensive. This is a quality versus
compilation-cost tradeoff, not a uniform speedup or a guarantee of only a few
percent overhead. Individual losses, repeated measurements, source/binary
hashes, and PNG/SVG plots remain outside the repository.

At 256 decomposed Grover iterations and twenty parallel trials, median wall is
3.86 versus 4.23 s, CPU is 8.85 versus 10.55 s, and peak RSS is 683.02 versus
693.32 MiB. This input increases native gates by 1.56% and depth by 11.86%. The
final assessment retains these losses; neither estimation nor bounded first-SWAP
guidance guarantees improvement on every circuit.

Repeated resource outputs match byte-for-byte across serial and parallel runs.
Samples that overlapped compiler activity or local lint were excluded with their
paired revision and repeated after a quiet window. The final timing and resource
samples have no detected build overlap. Measurements use a shared ARM64 host;
they are descriptive evidence, not a statistical guarantee.
