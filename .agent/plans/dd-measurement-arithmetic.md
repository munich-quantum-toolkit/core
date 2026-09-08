# DD measurement and arithmetic performance

Status: in progress. Measurement and basis changes are implemented and
validated; arithmetic evaluation remains.

## Goal and scope

Improve common state construction, sampling, and collapsing measurements in PR

## 2455. Reject invalid measurement indices at the native boundary. Evaluate

addition and multiplication on mixed workloads and include arithmetic changes
only when correctness and performance evidence supports them.

### Decisions

- Preserve bit ordering, RNG consumption, normalization checks, approximate-zero
  policy, root phase, reference ownership, and weak-cache invalidation.
- Reuse the existing matrix-vector cache for specialized projection; its keys
  and unscaled results must retain ordinary multiplication semantics.
- Read basis entries directly through a private compile-time accessor instead of
  materializing temporary enum vectors.
- Keep benchmark experiments separate from production until evaluated. A
  contrived reversed-addition cache hit is not sufficient evidence to change
  general addition. Check sparse gates, compressed states, dense operands,
  cold/warm caches, mixed operation sequences, and GC.

### Work remaining

- [ ] Evaluate addition and multiplication; retain only supported changes.
- [ ] Complete required checks, update the PR description, and record results.

### Validation

Use the existing DD and QCO test targets, Python DD/QCO tests, general lint, and
full-file C++ lint. Check dense numerical oracles and root lifetime across GC.
Compare baseline and candidate builds under identical compiler settings; report
measured workloads and variability rather than universal speedups.
