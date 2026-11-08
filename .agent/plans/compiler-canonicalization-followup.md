# Canonicalization cost follow-up

Status: proposed. Source review on upstream main `7681f45`. Nothing here is
implemented or measured on this revision. Item 0 comes first; items 1 and 4
depend on its results.

## Goal and scope

Reduce the remaining canonicalization cost on compiler entry points without
changing semantics. The predecessor plan,
[canonicalization cost in the compiler pipeline](compiler-canonicalization-cost.md),
fixed two non-terminating patterns, bounded the compiler canonicalizers, and
batched identity tensor cancellation. Read it first. Its rejected approaches
still hold, and each rejection has a counterexample:

- narrower pattern sets for promotion and parameter binding;
- removing either target post-optimization canonicalizer;
- splitting the public default pipeline from the compiler cleanup;
- removing the allocation clone in `RemoveFreshSlotResets`.

The same constraints apply. Failed matches leave IR unchanged, output verifies
and stays deterministic, and structural cleanup does not change quantum gate
sequences. Equivalent IR forms may change.

The scope covers canonicalizer passes and greedy drivers run by
`mlir/lib/Compiler`, `mlir/lib/Support/Passes.cpp`,
`mlir/lib/Dialect/MQT/Transforms/PromoteClassicalMemory.cpp`, and
`mlir/lib/Dialect/QCO/Transforms/LegalizePayloadControlFlow.cpp`. Conversion
patterns and boundary verification remain out of scope.

## Remaining cost, from the predecessor's baseline

The predecessor's baseline table predates #2966 and #2967, so treat it as
direction only:

- Canonicalization is 10–60% of reported stage time on most workloads.
- The largest absolute costs are Shor: 207.5/667.0 ms of canonicalization on the
  explicit target (small/large) and 118.4 ms on Adaptive QIR (large).
- Target compilation ran 8–68 canonicalizer invocations per compile. Shor ran
  the most, because promotion runs again in every unrolling round.

Since then, #2967 removed the QC cleanup from the QIR path. Adaptive QIR
preparation no longer promotes classical memory.

## Work items

"Proven" means the conclusion follows from the source. "Hypothesis" means it
needs item 0 before any code change.

### 0. Re-baseline on current main and attribute cost to sites

The predecessor's timing reports add up repeated pass instances, and nested
dynamic pipelines hide where the time goes. Before you change code, measure on
current main with the predecessor's workloads, sizes, target (line topology,
seed 7, 8 mapping trials), and build settings:

1. Canonicalizer time and successful rewrite count for each invocation, labelled
   by call site. Include the instance nested in `PromoteClassicalMemory`, and
   say whether its caller is a cleanup pipeline, an unrolling round, or
   `intoQIR`.
2. For each `UnrollLoopsForPayload` run: the number of rounds, and the time of
   each part of a round (promotion, `foldStaticBranches`, `UpliftStaticWhile`,
   unrolling).
3. Per-pattern time for the two most expensive workload/path pairs (expected:
   Shor on the target and on Adaptive QIR). Use the same out-of-tree action
   handler approach as the predecessor, or `mlir::tracing::ActionProfiler`
   registered on the compiler context.

Keep harnesses and raw data outside the repository. Record in this plan: the top
three call sites by time, the share of promotion normalizations that apply no
rewrite, and rounds per unrolling run.

### 1. Promotion normalizes input that is already canonical

Impact: likely the largest remaining lever on unrolled workloads. Evidence: the
call pattern is proven; its cost share is a hypothesis.

`PromoteClassicalMemory::runOnOperation` runs a full module canonicalizer
whenever any promotable or destructurable allocation exists, before SROA and
Mem2Reg. This happens even when the input is already canonical:

- `UnrollLoopsForPayload::runOnOperation` runs promotion at the start of every
  round. Each round also runs `foldStaticBranches` and `UpliftStaticWhile` over
  the whole module. Those greedy drivers fold the constant bound arithmetic that
  unrolling leaves behind, so the nested canonicalizer is not what makes inner
  trip counts literal.
- Allocations that promotion cannot remove, such as registers with dynamic
  indices inside loops, make every later promotion run pay a full canonicalizer
  and fail again.
- On the default and Adaptive QIR paths, promotion runs in both QCO cleanups and
  in `QCProgram::intoQIR`. Each run comes shortly after another canonicalizer,
  with only a few non-canonicalizing passes in between.

Do not narrow the pattern set. The predecessor shows that promotion needs full
canonicalization for indices that come from constant SCF branches. Candidate
designs, in order of preference:

1. **Promote first, then normalize and retry only if allocations remain.** Run
   SROA and Mem2Reg. If any promotable or destructurable allocation remains, run
   the canonicalizer and then SROA and Mem2Reg again. This stays inside the
   owning pass and adds no option. The cost goes up only where an allocation
   stays unpromotable, which item 0 quantifies.
2. **If option 1 does not pay off,** skip promotion in unrolling rounds that did
   not clone or promote any loop containing a classical memory access. The
   condition must be exact, not heuristic. An index that flows into an access
   outside the unrolled loop through a loop result also counts.

Prerequisite: add the predecessor's constant-SCF-branch promotion counterexample
as a regression test in
`mlir/unittests/Dialect/MQT/Transforms/test_classical_simplification.cpp`. The
predecessor kept it outside the repository, and it protects this item. If you
change rounds, also add the two-billion-iteration H/H identity loop as a test
under `mlir/unittests/Compiler`.

Acceptance: outputs are identical on all thirty predecessor paths. Process-CPU
time drops consistently on Shor target compilation, measured with alternating
paired runs. The classical simplification, compiler pipeline, and payload
control-flow suites pass.

### 2. Bound the last unlimited greedy driver

Impact: robustness; it runs in every unrolling round. Evidence: proven.

`foldStaticBranches` in `LegalizePayloadControlFlow.cpp` still uses
`GreedyRewriteConfig::kNoLimit` with top-down traversal. The predecessor's
reason for bounding the compiler canonicalizers applies here too: a pattern that
reports success without a change turns into a hang. Item 0 records its maximum
iteration count. Replace `kNoLimit` with the default limit. Unlike the
canonicalizer pass, `applyPatternsGreedily` reports non-convergence as failure,
and this function's callers already signal pass failure. A bound therefore turns
a hang into a diagnosed failure, not into silently non-canonical IR. Check that
the payload control-flow and compiler pipeline tests still pass.

### 3. Extend convergence coverage to every compiler canonicalizer

Impact: protects output quality now that canonicalizers stop after ten
iterations. Evidence: proven coverage gap.

`CompilerPipelineTest.CanonicalizationConverges` in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp` sets `test-convergence`
only for `populateQCOCleanupPipeline`. A bounded canonicalizer that does not
converge leaves non-canonical IR without any diagnostic. That shows up as worse
output, not as a failure. Apply the same option to the `canonicalize` passes in:

- `populateTargetCompilationPipeline` and `populateTargetSynthesisPipeline`,
  using an existing target fixture from the compiler tests;
- `qco::populateTargetNativeSynthesisPipeline`;
- `populateQCExportPipeline` and `populateQIRCleanupPipeline`.

Prefer parameterizing the existing test over the pipeline builders to copying
it. The canonicalizer nested in `PromoteClassicalMemory` is not reachable this
way. Do not add a pass option only for testing; item 0's iteration counts cover
it.

### 4. Adaptive QIR canonicalization on Shor

Impact: possibly high (118.4 ms of canonicalization on the predecessor's Shor
large baseline). Evidence: hypothesis; #2967 changed this path afterwards.

Decide after item 0. If the share is still high, name the dominant call site and
patterns from the per-pattern profile, then write a focused item with a
reproducer. Do not guess at the QIR cleanup or the LLVM dialect patterns before
that.

## Considered and not recommended now

- **Caching a pass manager for parameter binding.** A fresh pass manager costs
  about 21 µs of setup per call. The predecessor measured a 0.113 ms binding
  median, so at most about a fifth of binding is at stake. Caching would add
  lifetime and concurrency state to shared contexts. Revisit this only if a
  binding-latency requirement exists.
- **Fewer unrolling rounds.** Rounds are needed: branches that depend on the
  induction variable fold between rounds, and that can expose further loops.
  Reduce the cost per round (item 1) instead.
- **Nesting canonicalizers per function.** The predecessor's reasons still hold:
  compiled programs usually have one function after inlining.
- **Aligning the QC export canonicalizer configuration with QCO cleanup.** QC
  export still uses top-down traversal and CSE between iterations. There is no
  performance evidence either way, so change it only if item 0 shows a cost.
- **General QTensor chain scanning.** This stays deferred until a per-pattern
  profile shows a chain-walking pattern among the top costs.

## Validation

Run from the repository root:

- `cmake --preset release` and `cmake --build --preset release`
- `ctest --preset release`, plus focused filters for the suites named above
- `uv sync --inexact --no-dev --no-build-isolation-package mqt-core`, then
  `uv run --no-sync pytest test/python`
- `uvx nox -s cpp-lint` and `uvx nox -s lint` before handoff

Record the baseline revision, preset, toolchain, machine, and paired before and
after process-CPU and wall medians for each performance item. Leave items that
show no consistent gain unchanged, and record why.
