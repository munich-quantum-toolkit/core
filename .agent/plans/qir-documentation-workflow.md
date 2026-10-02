# Compilation, QIR execution, and documentation workflow

Status: implemented and validated.

## Goal and scope

Make the user guide follow compilation and execution, QDMI device access,
decision diagrams, and benchmarks. Keep MLIR dialect and pass references in the
API section and consolidate development policy and agent instructions. The QIR
notebook must compile Base and Adaptive programs, inspect capability metadata,
serialize text and bitcode, and execute both through DDSIM. Include direct
Qiskit conversion and QIR output records.

## Decisions

The compiler already produces profile and capability metadata. Examples inspect
that metadata rather than manually constructing it. Existing compiler APIs cover
Qiskit conversion and both serializations.

DDSIM currently disables textual output. Add opt-in capture using its existing
QDMI custom parameter/result mechanism: boolean CUSTOM2 enables capture and
CUSTOM1 returns the null-terminated stream. Keep CUSTOM1's independent job
parameter role as the sampling seed. The JIT already frames each shot and
disables terminal sampling when an output stream is attached. Capture is only
supported for QIR jobs with positive shot counts; ordinary sampling keeps its
current cost and state-retention behavior. No generic binding API is needed.

The tutorial series adds QIR, QDMI, and jeff chapters after hardware
compilation. A dedicated jeff guide owns interchange API and subset
documentation; tutorials exercise phase-sensitive unitary preservation, loops,
and process boundaries.

Two examples exposed API defects. Base conversion now marks SCF illegal so it
fails before returning an unserializable QIR program. QCO DD functionality
construction prebinds statically sized entry-block QTensor allocations alongside
scalar allocations, preserving instruction order; dynamic and nested allocation
remain unsupported. Both fixes live in the shared native owners.

## Validation

- Native release build and CTest passed: 3,531 tests, one existing SC-device
  skip.
- Python compiler, DD, QDMI, and QIR-runner suites passed: 405 tests.
- Full repository lint and changed-file C++ lint passed with zero C++ findings.
- A clean executable docs build and generated internal-link checks passed. All
  six tutorials execute independently and provide downloadable notebooks. The
  landing page has six cards and the sidebar follows the intended order.
- Documented jeff and QIR CLI commands preserve the checked program result.
- External linkcheck passed.
