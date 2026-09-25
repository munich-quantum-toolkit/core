# QDMI program output conformance

Status: implemented and submitted as draft; human review pending.

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
  IQM JSON for the frontend integrations.

## Completion

- [x] Implement full QIR/OpenQASM results and strict binary eligibility in
      DDSIM.
- [x] Preserve source layouts in serializers and plugin result reconstruction.
- [x] Compare DDSIM/plugin results with native SDK executions.
- [x] Implement and validate coordinated IQM and Braket provider adoption.
- [x] Publish three draft implementation PRs and reconcile superseded proposals.

## Validation

Use native DDSIM, QCO/QIR execution, OpenQASM frontend, QDMI binding, and plugin
tests for changed contracts. Run repository lint, whole-file C++ lint, and stub
generation for binding changes. Provider HTTP/AWS stubs must cover scrambled
keys/columns and placement permutations; SDK comparisons verify public outputs.

Core validation: release build succeeded; the 3599-test native suite passes with
one existing skip after updating output-contract fixtures. The focused Python
suite passes 562 tests. Documentation, generated stubs, repository lint, and
whole-file static analysis pass. The release build disables IPO locally to avoid
a prebuilt LLVM/LTO linker conflict.

Draft implementations:
[Core #2626](https://github.com/munich-quantum-toolkit/core/pull/2626),
[QDMI-on-IQM #276](https://github.com/iqm-finland/QDMI-on-IQM/pull/276), and
[Braket #242](https://github.com/munich-quantum-software/amazon-braket-qdmi-device/pull/242).
IQM passes 165 native tests, 70 Python tests with released Core, and all 21
serializer tests with this Core branch. Braket passes 144 native tests, 70
Python tests on both Python 3.14 and minimum direct dependencies on Python 3.11,
and all five SDK comparisons with this Core branch. Live test skips are
intentional. Provider lint, documentation, and whole-file static analysis were
also run; Braket retains existing dependency-header warnings. No hosted-CI
status or hardware execution is claimed.
