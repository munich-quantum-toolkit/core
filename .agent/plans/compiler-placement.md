# Preserve structured tensor programs through target compilation

Status: implementation, validation, and review complete. Integration base:
`fce58f02d` after PR #2519 (PR #2506).

## Authorized scope

Implement all findings in `../audits/compiler-placement-reassessment.md`:
target-aware placement without mandatory tensor-loop expansion, shared
constant-slot scalarization including SCF for loops, explicit tensor lifetime
limits, and consistent QDMI result bit order. Retain the classical tensor QIR
fix and safe placement diagnostics. Update the compiler PR and keep the
documentation PR stacked on its final head. Apply the accepted upstream
reassessment: reuse qubit-store role metadata, retain upstream routing and
modifier contracts, and remove the superseded mapper guard and unreachable
tensor-lifetime alternatives. Run the requested ponytail review before
publication.

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
- Placed tensors hold static qubit references in local buffers. Dynamic tensor
  slots must be restored before region exit or deallocation; unsupported
  ownership transfers fail before conversion. QC needs no new operation.
- DDSIM reverses recorded QIR bits before assembling its QDMI histogram. Shots
  and counts use the same most-significant-bit first order for every payload.

## Implementation

The implementation uses allocation-level placement for homogeneous Adaptive QIR
targets, shared constant-slot scalarization, and explicit dynamic tensor
lifetime checks. DDSIM normalizes QIR shots at its QDMI boundary. Existing
exact-site routing and payload checks remain in force. The audit records the
supported inputs, ownership boundary, and compilation limits. No new runtime
API, public option, or repository dependency is needed.

## Validation and integration

The final implementation builds with LLVM/MLIR 23.1.0 against `fce58f02d`. All
922 selected native compiler, conversion, QTensor, QIR runtime/JIT, and DDSIM
device tests pass. All 315 selected Python compiler, loop, integer-interchange,
QCO DD, and device-compilation tests pass with fresh packaged providers. The
simplified conversion test was rebuilt and its 22 regressions rerun; all 37
device-compilation tests also pass after formatting. Generated MLIR
documentation builds successfully. Full-file C++ lint against the fixed
integration base and general lint pass.

The ponytail review removed unused conversion state, mask-only memref legality
and alias exceptions, and mask-filtering test helpers. QC has no net dialect
change against the integration base. Temporary benchmark artifacts remain
excluded; regression tests retain coverage.

The separate README/RtD plan owns the stacked documentation checks and PR #2509.
PR #2506 contains the compiler implementation and its supported-input limits.
