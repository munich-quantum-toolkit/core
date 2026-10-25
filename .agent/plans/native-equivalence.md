# Native equivalence checking

Status: implemented. This record preserves the initial implementation evidence.
The [measured simplification follow-up](native-equivalence-performance.md)
records subsequent changes, validation and publication.

## Outcome and boundaries

A native `verify` API compares verified QCO functions or borrowed QCOProgram
objects. One bounded execution trace reuses the simulator's classical SCF
interpreter. Construction, alternating, and simulation consume the same
immutable resolved gates. Python exposes the native API without compiling input
objects or depending on QCEC.

The supported contract is documented in
[verification](../../docs/compilation/verification.md). The default leaves all
source circuit inputs free and preserves global phase. Initialized allocations,
explicit zero ancillas, concrete classical bindings, source/layout identity,
terminal readout, phase-insensitive comparison, normalized process/amplitude
distances, and exact selected terminal effects are supported. Outcomes
distinguish unsupported inputs, inconclusive computation, equivalence, and
non-equivalence.

Measurement-dependent control, resets, unresolved symbolic inputs, recursive
calls, nested quantum allocation, general channel equivalence, approximate
partial equivalence, HSF, and additional checker families remain excluded. These
are the investigation's boundaries, not incomplete implementation tasks.

## Durable decisions

- Preserve phase by default because a function can later be controlled. Exact
  comparison retains the preliminary helper's maximum entry residual and
  per-test tolerance. Approximate process distance ignores phase; normalized
  amplitude distance preserves it.
- Resolve only supplied classical bindings. Success makes no claim for other
  bindings or symbolic termination. Terminal sampling eligibility is not the
  admissibility test: a terminal measurement in a call is accepted, and an
  unsupported operation in an unexecuted branch does not reject execution.
- Materialize a bounded gate trace rather than introducing a second interpreter
  or resumable coroutine. Interpreter semantics remain shared with simulation.
- Decode initial placement, routing, and site IDs directly. Preserve a common
  source count and destination labels; never run the compiler transformation
  under test to normalize both inputs. Synthetic idle wires retain their source
  positions. Private workspace outputs require a clean-output proof.
- Construction uses weighted DD residual norms, including root weights and
  initialized-input dimension. Alternating uses sound Frobenius bounds for the
  maximum-entry relation. The overlap phase is a witness, not a phase-minimax
  solution; the unresolved interval stays inconclusive.
- Finite simulation is a counterexample search, not a positive all-input proof.
  The single initialized input can be exhaustive. All-input checks that omit
  workspace require construction to prove it clean before comparing outputs.
  Approximate average-distance checking does not use sampled worst-input errors
  as decisive evidence.
- Terminal observational checking compares each joint-outcome effect `A† Π_y A`,
  including constants/copied bits and arbitrary garbage. It does not claim
  reduced-state equality.
- Each checker owns its DD package and roots. An aggregate allocator budget and
  stop token cover worker packages and comparison caches. Worker grants are
  shared with sampling. A decisive worker cancels siblings, all workers join,
  and conflicting completed evidence is inconclusive.
- DD canonicalization tolerance remains unchanged by verification. External
  changes to the legacy process-global epsilon cannot overlap verification.
  Acceptance thresholds and numerical uncertainty are separate settings.
- Payload/DD limits do not bound process RSS or MLIR context storage.
  Interpreter dictionaries grow under a total step bound. Cancellation is
  cooperative, including recursive DD arithmetic; garbage collection remains an
  atomic safe-point operation.

## Ownership and deletion

| Layer                 | Owning files                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| DD resources          | `include/mqt-core/dd/ResourceControl.hpp`, `src/dd/ResourceControl.cpp`, package/table/slab storage |
| Execution and workers | `DDFunctionality.h/.cpp`, `DDWorkerBudget.h`                                                        |
| Interface             | `VerificationInterface.h/.cpp`                                                                      |
| Checkers              | `Verification.h/.cpp` in QCO Utils                                                                  |
| Borrowed programs     | `mqt/Compiler/Verification.h`, `mlir/lib/Compiler/Verification.cpp`                                 |
| Python                | `bindings/mlir/register_mlir.cpp`, regenerated `python/mqt/core/mlir.pyi`                           |
| Tests                 | DD tests, QCO Utils tests and frozen artifacts, compiler tests, Python verification tests           |

All 53 preliminary helper calls are migrated to `VerificationTest.h`, a thin
construction/phase-preserving assertion. `ExactUnitaryTest.h` and
`MLIRExactUnitaryTest` are removed. Independent matrix/reference calculations
are retained. Raw static synthesis fixtures explicitly declare their assumed
identity input layout before transformation. Raw physical compilation fixtures
supply their externally known identity to both sides only after asserting that
the compiler emitted no layout; production provenance checks remain strict.

The [extractable-PR ledger](native-equivalence-extractable-prs.md) tracks
standalone refactors/fixes and unimplemented findings without expanding this
task.

## Baseline and independent evidence

Core baseline: `1d384a5d23aa1f02b26fd5684dab586ef26167fe`. QCEC differential:
`4319be6bbf58cd7958a6447b42022721fb74ee3b`, with its supported Core dependency
`80daeb5f9a2c862dcafb6e3ddb6e45a75b410029`. The implementation has no QCEC
runtime dependency.

Frozen compiler artifacts and their provenance live in
`mlir/unittests/Dialect/QCO/Utils/testdata/verification`. A separate dense
oracle in `test_verification_dense.cpp` does not use Core DD construction or
output normalization. It checks coherent distances, mutations, zero ancillas,
terminal effects, reduced-state distinctions, and the projective maximum-entry
gray zone.

Pinned QCEC/native commands and raw outcomes are in
[the implementation probe](../audits/native-equivalence/implementation/qcec-differential/probe.py),
with
[native results](../audits/native-equivalence/implementation/qcec-differential/native.json)
and
[QCEC results](../audits/native-equivalence/implementation/qcec-differential/qcec.json).
Eleven cases agree where contracts overlap. Expected differences retain Core's
phase-sensitive default and inconclusive approximate boundary.

## Validation

The normal release preset built all eight affected native targets. Their suites
passed **2133/2133** tests: QCO Utils 300, Compiler 321, MQT transforms 31, QC
IR 369, QCO IR 586, native synthesis 99, optimizations 207, and DD 220. The
Utils suite includes 119 runtime tests, 15 interface tests, and six independent
dense tests. Commands and per-binary evidence are retained in
[the native record](../audits/native-equivalence/implementation/validation/native-tests.md).

The final broadened Python run passed **477/477** tests with the local SC
provider configured. Generated stubs, 34 focused Python checks, and all 11
pinned QCEC/native differential cases also pass. The initial broader run's six
setup failures came from the optional provider being absent. The final nox stub
refresh also required restoring the test dependency group before pytest could
load its parallel runner. Neither issue changed production code.

The full executable documentation build and internal-link checks pass. The final
API page was inspected with Chrome DevTools: rendered tables and mathematics, no
horizontal overflow or MathJax errors, no console messages, and all 40 loaded
requests successful. Full-file C++ lint passes with zero diagnostics: 28
translation units and the test helper header, with all 42 changed C++ path
hashes unchanged throughout. Repository lint passes without modifying files. See
[handoff validation](../audits/native-equivalence/implementation/validation/handoff.md).

The numerical review found a concrete powered-matrix defect: a matrix accepted
within the IR's tolerance becomes nonunitary or non-finite after large powers.
Resolved execution now rejects those gates before DD construction. The
[executable evidence](../audits/native-equivalence/implementation/validation/pow-amplification/README.md)
separates observed trace acceptance from the inferred failure of the old
distance formula. The orthogonal-unitary distance boundary independently tests
the small roundoff allowance; larger isometry violations remain inconclusive.
Zero approximate tolerance also obeys the numerical uncertainty interval, with a
regression preventing a zero-distance special case from bypassing it. A
[workspace eligibility regression](../audits/native-equivalence/implementation/validation/workspace-eligibility.md)
prevents simulation from issuing a verdict before the all-input clean-workspace
premise has been proved.

Local evidence does not establish hosted CI success or a full-repository test
pass. No representative end-to-end performance claim is made. DD accounting is
not a process RSS limit, and numerical guards are not interval certificates.
