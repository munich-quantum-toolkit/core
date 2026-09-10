# Preserve structured tensor programs through target compilation

Status: implementation updated; final validation and publication in progress.
Integration base: `a66a9b206` (PR #2506).

## Authorized scope

Implement all findings in `../audits/compiler-placement-reassessment.md`:
target-aware placement without mandatory tensor-loop expansion, shared
constant-slot scalarization including SCF for loops, and correct partial tensor
release. Retain the classical tensor QIR fix and safe placement diagnostics.
Update the compiler PR and keep the documentation PR stacked on its final head.
Apply the accepted upstream reassessment: reuse qubit-store role metadata,
retain upstream routing and modifier contracts, and remove the superseded mapper
guard and unreachable tensor-lifetime alternatives. Run the requested ponytail
review before publication.

## Contracts

- Preserve loop bodies when the selected target and payload can represent their
  indexed quantum operands. Site-dependent synthesis and routing must continue
  to check the actual physical placements.
- Keep capacity, site-ID, quantum lifetime, control-flow, and expansion-budget
  checks. Unsupported valid input must produce a diagnostic, never an assertion
  or partial successful compilation.
- Scalarization preserves quantum wire identity, classical state, attributes,
  region argument/result order, and loop feedback. Share the lifetime proof and
  extraction/reinsertion code without introducing a control-flow framework.
- Tensor extraction transfers ownership of a slot. Releasing the remaining
  tensor must not release an extracted qubit again. Array storage and ownership
  of the qubits it references are distinct concerns.

## Implementation

All three findings are implemented: allocation-level placement for homogeneous
Adaptive QIR targets, shared constant-slot scalarization, and partial tensor
ownership. Existing exact-site routing and payload checks remain in force. The
audit records the supported inputs, ownership boundary, and matched scaling
results. No new runtime API, public option, or repository dependency is needed.

## Validation and integration

Rebased onto `a66a9b206`, retaining the upstream control-flow and QIR analysis
improvements. Remaining gates are native and Python regression tests, refreshed
benchmark results, full-file C++ lint against that fixed base, general lint,
generated MLIR documentation, and the final complexity review. The separate
README/RtD plan owns the stacked documentation checks and PR #2509. PR #2506
contains the compiler implementation and its measured limits.
