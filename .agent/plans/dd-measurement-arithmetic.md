# DD measurement and arithmetic performance

Status: complete. Measurement, basis construction, and multiplication changes
are implemented and validated for PR #2455.

## Goal and scope

Improve common state construction and measurement in PR #2455. Reject invalid
measurement indices at the native boundary. Evaluate addition and multiplication
on mixed workloads and include arithmetic changes only when correctness and
performance evidence supports them.

### Decisions

- Preserve bit ordering, measurement probabilities, normalization checks,
  approximate-zero policy, root phase, reference ownership, and weak-cache
  invalidation.
- Reuse the existing matrix-vector cache for specialized projection; its keys
  and unscaled results must retain ordinary multiplication semantics.
- Read basis entries directly through a private compile-time accessor instead of
  materializing temporary enum vectors.
- Keep benchmark experiments separate from production until evaluated. A
  contrived reversed-addition cache hit is not sufficient evidence to change
  general addition. Check sparse gates, compressed states, dense operands,
  cold/warm caches, mixed operation sequences, and GC.
- Recompute the active matrix level after cache misses to skip shared implicit
  identities in matrix multiplication. Keep vector recursion levels: multiplying
  a gapped matrix by a scalar vector still needs zero extension.
- When the left matrix skips the current right-operand level, recurse directly
  over right successors. Preserve weight factoring and existing cache entries.
- Numerical accuracy is the measurement contract; exact baseline comparisons are
  diagnostic checks, not guarantees about rounding or random-engine state.
- Keep addition unchanged. Skipping shared matrix levels helped sparse sums but
  regressed dense matrix addition by 17–27%; a narrower guard did not fix it.
  Reverse cache lookup found no reuse in the profiled workloads.
- Keep probability accumulation unchanged. Regrouping was accurate but gave
  little benefit across the measured states.

### Validation and limits

Both release and Clang assertion-enabled builds passed all 3,988 configured
CTest cases, with one expected QDMI skip each. The rebuilt Python package passed
58 DD/QCO tests. General lint and full-file C++ lint passed against the PR merge
base. The new dense-reference test covers complex arithmetic, skipped levels,
collection, and scalar-vector extension.

Three final paired runs measured 24–26% lower time for a 128-qubit compressed
state sequence and 8–11% for six-qubit unitary construction. Mixed 12-qubit
circuits were near unchanged. Forced collection dominates some cases; tiny fully
dense products sometimes ran slower. These measurements justify the specialized
path, not a universal speedup. Addition candidates were rejected using separate
paired dense and sparse probes. Probability regrouping remained accurate but
gave too little benefit across the measured states.
