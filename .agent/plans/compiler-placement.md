# Preserve structured tensor programs through target compilation

Status: implementation, validation, and review complete. Integration base:
`fce58f02d` after PR #2519 (PR #2506).

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
audit records the supported inputs, ownership boundary, and compilation limits.
No new runtime API, public option, or repository dependency is needed.

## Validation and integration

Rebased onto `fce58f02d`, retaining PR #2519's clean API renames and terminology
corrections. The added indexed-placement test uses `fromOpenQASMString` and
`OperationCapability`. All 1,026 selected native tests, including CLI alias
rejections, pass. All 305 selected Python compiler, loop, integer-interchange,
QCO DD, and device-compilation tests pass after rebuilding the extension with
both packaged providers. Full-file C++ lint against that fixed base, general
lint, and generated MLIR documentation pass. Temporary benchmark artifacts have
been removed from the PR; native regressions retain coverage. The final
complexity review found no further C++ cuts.

The separate README/RtD plan owns the stacked documentation checks and PR #2509.
PR #2506 contains the compiler implementation and its supported-input limits.
