# Structured compilation fixes

Status: implementation and regression checks complete. Local C++ lint remains
unverified because clang-tidy 23 is unavailable.

## Goal and scope

Support structured programs that currently fail after a jeff round trip or
during native synthesis. Fix the owning conversion or transformation and retain
diagnostics for unsupported inputs. Preserve quantum phase, wire identity,
classical memory ordering, and QCO linearity.

The affected boundaries are Boolean tensor conversion in
`mlir/lib/Conversion/JeffToQCO/`, measurement destination preparation in
`mlir/lib/Conversion/QCToQIR/QIRCommon/`, and structured unitary lowering in
`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/`.

## Decisions

- Use direct GoogleTest regressions in each owning MLIR subsystem, with semantic
  checks and negative cases for retained restrictions.
- Match the initial optimization when comparing direct and jeff exchange paths.
  Keep experiment artifacts local; production tests use reduced inputs.

## Outcome

- Boolean array constants and creation import directly to CBit registers; length
  becomes the static register width. Invalid element counts and types fail
  before conversion. Wider integer arrays retain tensor conversion.
- Shared QIR preparation moves pure, speculatable index computations before
  measurements after validating all output stores. Classical memory effects,
  measurement-dependent indices, and unsupported computations remain barriers.
- Native synthesis removes unobservable entry-point phases inside classical
  control flow while preserving helper and quantum-modifier phases. Runtime
  single-controlled phase gates use three phase gates and two CX gates, then
  reuse target synthesis with the original ordered sites. Other unsupported
  runtime two-qubit operations retain diagnostics.

## Validation

Build the following targets with the release preset and run their executables
under `build/release/mlir/unittests/`:

- `mqt-core-mlir-unittest-jeff-round-trip`: 156 tests passed, including
  serialized exchange, bit ordering, malformed arrays, and wider integer arrays.
- `mqt-core-mlir-unittest-qc-to-qir-adaptive`: 164 tests passed, including safe
  index scheduling and rejection without mutation.
- `mqt-core-mlir-unittest-qc-to-qir-base`: 138 tests passed.
- `mqt-core-mlir-unittest-target-synthesis`: 68 tests passed, including full
  controlled-phase unitaries with runtime angles and reversed native sites.
- `mqt-core-mlir-unittests-compiler`: 230 tests passed.

Repository lint and `mlir-doc` generation passed. The `cpp-lint` session
requires clang-tidy 23; the available version is 22, so that check remains
pending.

After rebuilding the editable package, the local pilot passes all 16 small DDSIM
cases through both direct and jeff exchange routes, plus four larger exchange
cases. Each successful case passes its 4096-shot analytic check. These are
correctness observations, not performance measurements or claims of unrestricted
benchmark support.
