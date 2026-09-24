# QDMI program output conformance

Status: in progress.

## Goal and scope

Adopt
[QDMI's output contract](https://github.com/Munich-Quantum-Software-Stack/QDMI/pull/552)
in DDSIM and the Qiskit/PennyLane plugins. Binary results use output bit zero on
the right. Complete typed output is separate; no numeric value may be silently
dropped to produce shots. Compare public plugin results against native Qiskit
and PennyLane executions, including asymmetric layouts and initialized holes.

## Ownership

- The OpenQASM frontend selects final outputs and retains their names/types.
- QCO execution captures complete classical return values while keeping the
  existing terminal-sampling optimization. DDSIM encodes named JSON output.
- QIR runtime records determine binary eligibility per execution. DDSIM retains
  the complete standard stream whenever nonbinary output can be produced.
- A serializer must retain source-output reconstruction when its target payload
  cannot express classical destinations directly. Generic plugins consume that
  mapping; device implementations do not interpret SDK-specific measurement
  keys.
- IQM and Braket provider PRs normalize backend responses independently of
  physical placement. Unsupported program features fail explicitly.
- IQM keeps its existing direct QIR submission path, without an LLVM dependency.
  Its terminal-measurement limitation is documented; full conformance work uses
  IQM JSON for the frontend integrations (user decision, 2026-09-24).

## Work remaining

- [x] Implement full QIR/OpenQASM results and strict binary eligibility in
      DDSIM.
- [x] Preserve source layouts in serializers and plugin result reconstruction.
- [x] Compare DDSIM/plugin results with native SDK executions.
- [ ] Implement and validate coordinated IQM and Braket provider adoption.
- [ ] Publish three draft implementation PRs and reconcile superseded proposals.

## Validation

Use native DDSIM, QCO/QIR execution, OpenQASM frontend, QDMI binding, and plugin
tests for changed contracts. Run repository lint, whole-file C++ lint, and stub
generation for binding changes. Provider HTTP/AWS stubs must cover scrambled
keys/columns and placement permutations; SDK comparisons verify public outputs.

Core validation: release build succeeded; the 3599-test native suite passes with
one existing skip after updating output-contract fixtures. The focused Python
suite passes 562 tests. Documentation and generated stubs build. Final
whole-file static analysis and publication remain in progress.
