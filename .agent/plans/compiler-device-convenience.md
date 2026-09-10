# Device-directed compiler and submission API

Status: complete.

## Goal and decisions

`compile_program(program, target=device_or_id)` returns an immutable serialized
artifact without a session. `device.submit` accepts source or compiled
artifacts; `submit_program` resolves a device and delegates. Submission defaults
to 1024 shots, with zero retained for simulator extraction. No compiler result
mode or artifact submission method is introduced.

The shared C++ QDMI adapter selects Adaptive QIR, OpenQASM 3, then Base QIR
(binary before text), before payload-aware compilation. Explicit format choices
never fall back. Absent QDMI capability metadata, assume maximal
compiler-supported language/profile features. A versioned private CUSTOM2 report
overrides the fallback with explicit capabilities and constraints; DDSIM reports
maximal support. Malformed recognized reports fail. Contract comparison uses
effective capabilities, independent of whether they were assumed or reported.

Artifacts retain the compiler-verified contract. Submission compares current
ordered site mapping, topology, operation support/applicability, relevant timing
units, and payload capabilities. Calibration-only changes and names are ignored.
Different contracts require recompilation. Raw QDMI submission stays available
without claiming compiler provenance. Compatibility does not guarantee provider
acceptance or eliminate metadata races.

## Validation and delivery

Local tests passed: 209 compiler tests, 126 QIR IR tests, 72 DDSIM tests, and
the combined 334-test Python MLIR/QDMI suite. They cover all convenience paths,
exact formats, source import once, session lifetime, matching and changed
contracts, maximal fallback and private reports, malformed reports, defaults,
negative/zero shots, integer computation, and measurement-controlled loops.

Adaptive loop metadata includes the back-edge block and safely classifies loops
without a conditional exit. QIR artifacts require the existing i64 () entry ABI;
void returns become success status 0. Scalar outputs that change the signature
require OpenQASM or local temporaries. Result-accessor redesign remains outside
this change.

Bindings/stubs were regenerated. Repository lint, whole-file clang-tidy for the
changed C++ sources, and all four executable cells in the updated documentation
passed locally. The source-tree C++ adapter owns selection and compatibility;
the low-level QDMI client remains independent of MLIR.

The QDMI adapter header uses forward declarations and the shared
custom-parameter type without including the throwing client implementation. GCC
compiles the `mqt-cc` consumer with exceptions and RTTI disabled.

This change follows the merged simulator state-retention change (#2494).
