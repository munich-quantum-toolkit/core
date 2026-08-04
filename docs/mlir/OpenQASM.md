# OpenQASM input and output

The compiler reads a supported subset of OpenQASM 3 through a staged lexer,
parser, and semantic analyzer, then emits the QC dialect directly. The
[OpenQASM live specification](https://openqasm.com/index.html) defines the
language; this page records the subset implemented by MQT Core. The compiler can
also emit OpenQASM 3.1 from QC after optimization or directly from a
{code}`mlir::QCProgram`. OpenQASM remains a boundary format rather than an
intermediate dialect: compilation uses QC, QCO, and standard MLIR operations
internally.

## OpenQASM emission

The QC exporter is independent of the legacy QASM importer and circuit IR. It
prints validated QC and SCF operations directly and buffers the complete source
before writing it. A failed translation therefore never leaves partial OpenQASM
in the destination stream.

The public C++ translation functions support strings and arbitrary LLVM output
streams:

```cpp
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"

auto source = mlir::qc::translateQCToOpenQASM3(moduleOp);
if (mlir::failed(source)) {
  // The diagnostic identifies the unsupported operation and its location.
}
```

The compiler API wraps successful output in an {code}`mlir::OpenQASMProgram`:

```cpp
auto qc = mlir::QCProgram::fromQASMFile("input.qasm");
auto direct = qc->toOpenQASM3(); // Does not consume or optimize qc.
direct->write("direct.qasm");

auto optimized = mlir::runDefaultPipeline(
    mlir::CompilerInput{std::move(*qc)}, mlir::ProgramFormat::OpenQASM3);
```

Python exposes the same two paths:

```python
from mqt.core.mlir import OutputFormat, QCProgram, compile_program

qc = QCProgram.from_qasm_file("input.qasm")
direct = qc.to_openqasm3()
print(direct.source)
direct.write("direct.qasm")

optimized = compile_program("input.qasm", output=OutputFormat.OPENQASM3)
optimized.write("optimized.qasm")
```

The command-line driver writes OpenQASM to standard output by default or to the
file passed with {code}`-o`:

```console
mqt-cc input.qasm --emit=openqasm3
mqt-cc input.qasm --emit=openqasm3 -o optimized.qasm
```

The compiler-pipeline path runs the selected target compilation, the QCO
optimization pipeline, and QCO-to-QC conversion before emission. Calling
{py:meth}`~mqt.core.mlir.QCProgram.to_openqasm3` or
{code}`mlir::QCProgram::toOpenQASM3` emits the current QC program without that
optimization round trip.

### Emission and round-trip support

| QC or MLIR concept                         | Emission support                                                                       | Round-trip notes                                                                                               |
| ------------------------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Logical and physical qubits                | Scalar and static rank-one registers; physical qubits use {code}`$N`                   | Mixing allocation modes remains subject to the QC verifier                                                     |
| Measurement, reset, barrier                | Supported                                                                              | Bit outputs retain their scalar or register shape                                                              |
| Standard and extended QC gates             | Supported                                                                              | Extended gates receive focused private definitions under the {code}`_mqt_` prefix                              |
| {code}`ctrl`, {code}`inv`, and {code}`pow` | Supported, including multi-operation regions through generated private gates           | Modifier regions must contain only unitary operations and printable scalar expressions                         |
| Arithmetic, comparisons, casts, and math   | {code}`i1`, {code}`i64`, {code}`f64`, and internal {code}`index` values                | Explicit cast expressions are valid OpenQASM 3.1 output, but the current input grammar does not yet parse them |
| {code}`scf.if` and {code}`arith.select`    | Supported with result variables declared before the branch                             | Selects are materialized as structured {code}`if`/{code}`else`                                                 |
| {code}`scf.for`                            | Constant bounds and a positive constant step                                           | MLIR's exclusive upper bound is rendered as an inclusive OpenQASM range                                        |
| {code}`scf.while`                          | Side-effect-free printable conditions, unchanged forwarding, and type-preserving state | Loop-carried scalar and bit state uses temporary next-state values                                             |
| {code}`scf.index_switch`                   | Deterministic nested {code}`if`/{code}`else` blocks                                    | Case and default results are assigned to variables declared outside the chain                                  |
| Multiple classical results                 | Supported                                                                              | Import metadata preserves valid output names and {code}`bit`, {code}`bool`, signed, and float kinds            |
| Runtime safety machinery                   | Deliberately unsupported                                                               | Surviving assertions, checked-index scaffolding, or live poison values cause an explicit diagnostic            |
| Dynamic indices and ranges                 | Unsupported                                                                            | Constant folding may remove the dynamic machinery before emission; no unsafe approximation is emitted          |

The exporter emits {code}`OPENQASM 3.1;` and {code}`include "stdgates.inc";`. It
uses standard gates where possible and defines only the extended gates used by
the program. Generated qubit, temporary, and helper identifiers use a
collision-safe {code}`_mqt_` prefix. Valid output names recorded by the importer
are retained. Scalar declarations use the unsized OpenQASM {code}`bool`,
{code}`int`, {code}`uint`, and {code}`float` types. Result metadata must agree
with the MLIR result type. Without metadata, a user-visible signless {code}`i64`
result is accepted only when an explicitly signed or unsigned operation
determines its OpenQASM type; otherwise emission diagnoses the ambiguity.

Emission accepts exactly one defined, argument-free function. It rejects calls,
arbitrary CFGs, multi-block SCF regions, dynamic memory indices or loop ranges,
general memrefs, unsupported integer widths, packed bit-vector operations,
unknown operations, and non-unitary modifier contents. Diagnostics are attached
to the relevant MLIR location. Dead side-effect-free arithmetic left behind by
the frontend may be omitted because it cannot affect the emitted program.

Practical programs with static qubit and bit indices round-trip through the
strict OpenQASM frontend. The exporter does not recognize or reverse the
frontend's runtime bounds-check and checked-arithmetic machinery. If this
machinery survives optimization, emission fails instead of producing a
potentially different program. Cast-containing output is standards-compliant,
but is not part of this strict round-trip subset until the input grammar gains
explicit type-cast syntax.

## Parser and semantic support

| Feature                                                  | Status                                | Restriction                                                                                                                                                  | Representative test                                                                                            |
| -------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Version declaration and profile selection                | Supported                             | Explicit OpenQASM 3.0 and 3.1 select the maintained OpenQASM 3 profile; versionless input uses the same mode, and later 3.x versions are rejected            | `PreservesExactAndOptionalVersionSemantics`                                                                    |
| `stdgates.inc`, `qelib1.inc`, and nested includes        | Supported with compatibility leniency | The two libraries keep distinct gate sets, but either spelling is accepted in either source mode                                                             | `PreservesStandardLibraryIdentity`, `AcceptsHybridOpenQASM2Libraries`                                          |
| Qubits, bits, and `bool`, `int`, `uint`, `float` scalars | Supported                             | Width-qualified integer and floating types are not yet supported                                                                                             | `RejectsUnsupportedIntegerDeclarations`                                                                        |
| Implicit and explicit outputs                            | Supported                             | Without an `output` declaration, every global classical variable is returned; otherwise only explicitly marked variables are returned, in declaration order  | `PreservesOrderedScalarAndRegisterOutputs`                                                                     |
| Lexical scope, assignment, constants, and conversions    | Supported                             | Mutable global values cannot be captured by gate definitions; resolved conversions are represented explicitly in the typed frontend                          | `TracksLexicalScopeAndEnclosingAssignments`, `RecordsResolvedConversionsInTypedExpressions`                    |
| `if`, inclusive `for`, and `while`                       | Supported                             | Gate bodies contain gate calls and loops over gate calls only                                                                                                | `EmitsStructuredLoopsWithCarriedMutableState`                                                                  |
| Expressions and scalar math functions                    | Supported                             | Boolean `&&` and `||` short-circuit; gate parameters and `pi`/`tau` use angle semantics; inverse trigonometric functions return angles                       | `RestrictsMathBuiltinsOnGateAngles`, `FoldsAndEmitsCeilingAndFloor`, `LowersShortCircuitBooleanEvaluation`     |
| Bit-vector builtins                                      | Supported                             | `popcount(bit[n])` and `rotl`/`rotr(bit[n], int)` require a fully initialized whole bit register; rotations preserve its exact width and may be nested       | `LowersTypedBitVectorBuiltins`, `InvalidatesPopcountIndexFactsOnBitMutation`                                   |
| Measurement                                              | Supported in statement contexts       | Measurement is accepted in declarations, assignments, legacy arrows, and targetless statements, not as a general expression; outputs are assigned separately | `RejectsMeasurementsInGeneralExpressions`                                                                      |
| Dynamic qubit and bit indexing                           | Supported                             | Target restrictions and the combined emission budget are listed below                                                                                        | `DispatchesDynamicQubitGatesWithStructuredControlFlow`                                                         |
| Physical and declared qubits                             | Supported by the frontend             | A partially constrained program may reference both; the current QC target rejects mixing static and dynamic allocation modes                                 | `AcceptsMixedPhysicalAndDeclaredQubits`, `RejectsMixedQubitAllocationAtTheQCTarget`                            |
| Primitive, broadcast, and custom gates                   | Supported                             | Recursive definitions and mismatched broadcast widths are rejected                                                                                           | `BroadcastsRegistersAlongsideScalarQubits`                                                                     |
| `inv`, `ctrl`, `negctrl`, and `pow` modifiers            | Supported                             | Target support differs below; modifier and custom-gate dependency depth are bounded                                                                          | `LowersDynamicPowerModifiersToQC`, `BoundsModifiersAndGateDependencies`                                        |
| `input` declarations, subroutines, and `extern`          | Recognized and rejected by the parser | `input`, `def`, `return`, and `extern` are reserved but are not in the implemented grammar                                                                   | `DiagnosesUnsupportedReservedFeatureSyntax`                                                                    |
| Calibration, timing, `duration`, and `stretch`           | Recognized and rejected by the parser | `defcalgrammar`, `cal`, `defcal`, `delay`, `durationof`, `duration`, and `stretch` are not implemented                                                       | `DiagnosesUnsupportedReservedFeatureSyntax`                                                                    |
| `array`, `complex`, `angle`, and aliases                 | Recognized and rejected by the parser | Aggregate, complex, angle, and `let` alias declarations have no typed representation yet                                                                     | `RejectsUnsupportedReservedWordsAsIdentifiers`                                                                 |
| `switch`, `break`, and `continue`                        | Recognized and rejected by the parser | These control-flow forms are reserved but are not in the implemented grammar                                                                                 | `RejectsUnsupportedReservedWordsAsIdentifiers`                                                                 |
| Bitwise and shift operators                              | Parsed and rejected semantically      | Explicitly sized `uint`, `bit`, or `angle` operands are required by the language and are not implemented                                                     | `RejectsInvalidProgramsAcrossSemanticFamilies`                                                                 |
| Frontend resource limits                                 | Diagnosed                             | Expression depth is limited to 256, block/modifier/custom-gate dependency depth to 64, and register elements and typed statements to 100000                  | `BoundsExpressionAndBlockDepth`, `BoundsRegisterStorageBeforeAllocation`, `BoundsModifiersAndGateDependencies` |

Syntax, semantic, and target diagnostics use MLIR's diagnostic engine at the
translation boundary. Diagnostics originating in nested includes retain the
included source location and the complete include call stack.

The frontend resolves promotions and assignments before QC emission. Runtime
floating-point-to-integer conversions lower to `arith.fptosi` or `arith.fptoui`,
which round toward zero. Values that cannot be represented by the destination
integer type, including non-finite values, produce poison rather than a checked
runtime failure. Integer arithmetic operations for which OpenQASM requires a
runtime precondition continue to emit explicit assertions.

Bit outputs use the classical-register representation from the QC dialect:
`bit[n]` is returned as `memref<nxi1>`, including `bit` as `memref<1xi1>`.
Non-output bits remain SSA values and do not allocate classical result storage.
Other scalar outputs retain their builtin MLIR scalar types. This keeps the
function signature in source declaration order through QC, QCO, and
reconstructed QC. Current QIR output recording covers returned bit registers
whose elements are assigned directly from measurements. Arbitrary
classical-valued bit outputs and scalar OpenQASM outputs remain
target-capability follow-ups.

Scalar `qubit` declarations lower to `qc.alloc`. Explicitly sized declarations
remain register allocations, so `qubit[1]` retains a one-element
`memref<1x!qc.qubit>` rather than being conflated with the scalar form.

## Translation and compiler support

The standard compiler path is OpenQASM to QC, optimized QCO, reconstructed QC,
and QIR. This is the primary acceptance contract. Serialization through `jeff`
is optional: a smaller positive corpus exercises it, while known incompatible
programs are accepted by the standard path and fail explicitly when QCO is
converted to `jeff`. Base refers to direct production of the QIR Base Profile.

| Feature                                         | Parse     | Semantics | QC                                               | Standard Adaptive QIR                | `jeff`                                                  | Base                                 | Restriction or rejection reason                                                                                                 | Representative test                                                                           |
| ----------------------------------------------- | --------- | --------- | ------------------------------------------------ | ------------------------------------ | ------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Primitive and custom gates                      | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Supported                            | Custom gates are expanded during QC emission                                                                                    | `broadcast_custom_gate`                                                                       |
| Gate arithmetic and math parameters             | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Supported                            | Scalar `pow()` is distinct from the gate modifier                                                                               | `math_parameters`                                                                             |
| Scalar `ceiling()` and `floor()`                | Supported | Supported | Supported through `math.ceil`/`math.floor`       | Supported                            | Supported in the positive corpus                        | Supported                            | Constant calls are folded during semantic analysis                                                                              | `runtime_scalar_rounding`                                                                     |
| `popcount()`, `rotl()`, and `rotr()`            | Supported | Supported | Supported through integer popcount and rotate    | Supported                            | Rejected at QCO-to-`jeff`                               | Not in the tested Base subset        | Bit zero is the packed least-significant bit; `jeff` cannot represent the retained integer popcount and funnel-shift operations | `bit_vector_builtins`                                                                         |
| Broadcast gates                                 | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Supported                            | Operands must have compatible widths                                                                                            | `broadcast_custom_gate`                                                                       |
| `inv`, `ctrl`, and `negctrl`                    | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Not in the tested Base subset        | Modifiers on custom gates that require structured control flow are rejected                                                     | `RejectsModifiersOnTransitivelyStructuredCustomGatesAtQCTarget`                               |
| `pow @`                                         | Supported | Supported | Supported                                        | Supported for canonicalizable bodies | Supported after canonicalization in the positive corpus | Supported for canonicalizable bodies | Dynamic exponents remain `qc.pow`; composite bodies that cannot yet be canonicalized fail at the downstream conversion boundary | `ChecksPowerExponentPrecisionAndNesting`, `custom-pow-hs`                                     |
| `if` and nested `if`/`for`                      | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Adaptive only                        | The Base corpus is intentionally straight-line                                                                                  | `nested_static_control_flow`                                                                  |
| Measurement-controlled `while`                  | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Adaptive only                        | Requires runtime classical control                                                                                              | `measurement_controlled_while`                                                                |
| Loop-carried mutable bit state                  | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Adaptive only                        | Carried bit state remains SSA values through QC, QCO, and reconstructed QC                                                      | `mutable_loop_state`                                                                          |
| Loop-carried numeric scalar state               | Supported | Supported | Supported                                        | Supported                            | Not in the positive corpus                              | Adaptive only                        | QCO-to-QC preserves classical arguments, results, yields, and conditions alongside reference-semantic qubits                    | `scalar_loop_state`, `PreservesTypeChangingClassicalWhileState`                               |
| Checked signed and wrapping unsigned arithmetic | Supported | Supported | Supported with runtime assertions                | Supported                            | Rejected at QCO-to-`jeff` when live                     | Not in the tested Base subset        | Signed overflow and invalid division are asserted; unsigned arithmetic wraps at 64 bits                                         | `checked_integer_state`                                                                       |
| Constant inclusive ranges                       | Supported | Supported | Supported with constant `scf.for` bounds         | Supported                            | Supported when the body is compatible                   | Adaptive only                        | Positive, negative, empty, singleton, non-divisible, and boundary ranges avoid runtime trip-count math                          | `UsesConstantBoundsForStaticInclusiveRanges`                                                  |
| Dynamic inclusive ranges                        | Supported | Supported | Supported with comparison-driven `scf.while`     | Supported                            | Rejected at QCO-to-`jeff`                               | Adaptive only                        | A dynamic zero step is asserted; iteration uses no division-based trip count                                                    | `dynamic_range`                                                                               |
| Dynamic indexing resolved by optimization       | Supported | Supported | Supported with bounds assertions                 | Supported                            | Supported when assertions fold away                     | Not in the tested Base subset        | Straight-line constants and equal-constant branch joins can be simplified before `jeff`                                         | `resolved_dynamic_index`, `equal_constant_index_join`                                         |
| General runtime and induction-variable indexing | Supported | Supported | Supported with bounds assertions                 | Supported                            | Rejected at QCO-to-`jeff`                               | Not in the tested Base subset        | The standard QIR path lowers the assertions; `jeff` cannot currently represent them                                             | `runtime_dynamic_index`, `induction_variable_index`                                           |
| Measurement, reset, and barrier                 | Supported | Supported | Supported                                        | Supported                            | Supported in the positive corpus                        | Measurement and barrier supported    | Reset is Adaptive-only                                                                                                          | `reset`, `barrier`                                                                            |
| Projected QC emission budget                    | Supported | Supported | Rejected above 10,000,000 constructed operations | Rejected                             | Rejected before `jeff`                                  | Rejected                             | One overflow-safe projection composes custom-gate expansion, dynamic dispatch, and linear bit-vector packing or unpacking work  | `ComposesDispatchAndCustomGateExpansionBudgets`, `BudgetsRepresentativeOperationConstruction` |

The integration suites use public compiler APIs. The broad corpus must complete
the standard pipeline and the smaller `jeff` corpus must round-trip. The
incompatible corpus records failures at `intoJeff()` so those limitations do not
reduce the set of programs accepted by QC or QIR.

Bit-register declaration initializers, casts, sized-`uint` overloads, bit-string
literals, and additional constant folding remain follow-up work. They are not
silently accepted by the bit-vector builtin subset above.
