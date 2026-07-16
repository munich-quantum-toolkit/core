# OpenQASM input

The compiler reads OpenQASM 3 through a staged lexer, parser, and semantic
analyzer, then emits the QC dialect directly. OpenQASM is an input language, not
an intermediate dialect: a successful translation contains QC and standard MLIR
operations only. Translation fails before returning a module when the QC target
cannot represent an accepted source feature.

## Parser and semantic support

| Feature | Status | Restriction | Representative test |
| --- | --- | --- | --- |
| OpenQASM 3.0 and 3.1 | Supported | Later 3.x versions are rejected | `PreservesExactAndOptionalVersionSemantics` |
| `stdgates.inc` and nested includes | Supported | Includes are expanded textually with bounded depth and source locations | `ExpandsNestedIncludesAtTheirSourceLocations` |
| Qubits, bits, and `bool`, `int`, `uint`, `float` scalars | Supported | Width-qualified integer and floating types are not yet supported | `RejectsUnsupportedIntegerDeclarations` |
| Lexical scope, assignment, and constants | Supported | Mutable global values cannot be captured by gate definitions | `TracksLexicalScopeAndEnclosingAssignments` |
| `if`, inclusive `for`, and `while` | Supported | Gate bodies contain gate calls and loops over gate calls only | `EmitsStructuredLoopsWithCarriedMutableState` |
| Expressions and scalar math functions | Supported | Operations are checked against the implemented scalar type rules | `AcceptsAllScalarOperatorsAndComparisonPredicates` |
| Dynamic qubit and bit indexing | Supported | Target restrictions and the combined emission budget are listed below | `DispatchesDynamicQubitGatesWithStructuredControlFlow` |
| Primitive, broadcast, and custom gates | Supported | Recursive definitions and mismatched broadcast widths are rejected | `BroadcastsRegistersAlongsideScalarQubits` |
| `inv`, `ctrl`, `negctrl`, and `pow` modifiers | Parsed and semantically checked | Target support differs below | `RejectsInvalidGateControlAndBroadcastShapes` |
| `input` declarations, subroutines, and `extern` | Recognized and rejected by the parser | `input`, `def`, `return`, and `extern` are reserved but are not in the implemented grammar | `DiagnosesUnsupportedReservedFeatureSyntax` |
| Calibration, timing, `duration`, and `stretch` | Recognized and rejected by the parser | `defcalgrammar`, `cal`, `defcal`, `delay`, `durationof`, `duration`, and `stretch` are not implemented | `DiagnosesUnsupportedReservedFeatureSyntax` |
| `array`, `complex`, `angle`, and aliases | Recognized and rejected by the parser | Aggregate, complex, angle, and `let` alias declarations have no typed representation yet | `RejectsUnsupportedReservedWordsAsIdentifiers` |
| `switch`, `break`, and `continue` | Recognized and rejected by the parser | These control-flow forms are reserved but are not in the implemented grammar | `RejectsUnsupportedReservedWordsAsIdentifiers` |
| Bitwise and shift operators | Parsed and rejected semantically | Explicitly sized `uint`, `bit`, or `angle` operands are required by the language and are not implemented | `RejectsInvalidProgramsAcrossSemanticFamilies` |

## Translation and compiler support

“Adaptive plus Jeff” means the tested public path from QC through optimized QCO,
Jeff byte serialization and deserialization, back to QC, and finally to Adaptive
QIR. Base refers to direct production of the QIR Base Profile.

| Feature | Parse | Semantics | QC | Adaptive plus Jeff | Base | Restriction or rejection reason | Representative test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Primitive and custom gates | Supported | Supported | Supported | Supported | Supported | Custom gates are expanded during QC emission | `broadcast_custom_gate` |
| Gate arithmetic and math parameters | Supported | Supported | Supported | Supported | Supported | Scalar `pow()` is distinct from the gate modifier | `math_parameters` |
| Broadcast gates | Supported | Supported | Supported | Supported | Supported | Operands must have compatible widths | `broadcast_custom_gate` |
| `inv`, `ctrl`, and `negctrl` | Supported | Supported | Supported | Supported | Not in the tested Base subset | Modifiers on custom gates that directly or transitively require structured control flow are rejected | `RejectsModifiersOnTransitivelyStructuredCustomGatesAtQCTarget` |
| `pow @` | Supported | Supported | Rejected | Rejected | Rejected | The QC dialect has no power modifier yet; translation reports the source location | `RejectsPowerAtTheQCTargetBoundary` |
| `if` and nested `if`/`for` | Supported | Supported | Supported | Supported | Adaptive only | The Base corpus is intentionally straight-line | `nested_static_control_flow` |
| Measurement-controlled `while` | Supported | Supported | Supported | Supported | Adaptive only | Requires runtime classical control | `measurement_controlled_while` |
| Loop-carried mutable bit state | Supported | Supported | Supported | Supported | Adaptive only | Carried state remains SSA values across QC, QCO, Jeff, and QIR | `mutable_loop_state` |
| Loop-carried mutable floating state | Supported | Supported | Supported | Supported | Adaptive only | The result is used after both `for` and `while` | `scalar_loop_state` |
| Non-folded checked integer arithmetic and ranges | Supported | Supported | Rejected | Rejected | Rejected | The complete path cannot preserve the required overflow and range assertions; constant-folded expressions remain supported | `RejectsCheckedIntegerArithmeticAtQCTarget` |
| Dynamic indexing resolved by optimization | Supported | Supported | Supported | Supported | Not in the tested Base subset | Straight-line constants and equal-constant branch joins are accepted | `resolved_dynamic_index`, `equal_constant_index_join` |
| Multi-iteration induction-variable indexing | Supported | Supported | Rejected | Rejected | Rejected | Static source bounds do not guarantee that Jeff eliminates the emitted `scf.for` | `RejectsMultiIterationInductionIndicesAtQCTarget` |
| General runtime dynamic indexing | Supported | Supported | Rejected | Rejected | Rejected | The complete compiler path cannot preserve the required bounds assertion through Jeff; QC emission reports the source location | `RejectsRuntimeDynamicIndicesAtQCTarget` |
| Measurement, reset, and barrier | Supported | Supported | Supported | Supported | Measurement and barrier supported | Reset is Adaptive-only | `reset`, `barrier` |
| Projected QC emission budget | Supported | Supported | Rejected above 100000 primitive applications | Rejected | Rejected | One overflow-safe projection composes custom-gate expansion with dynamic-dispatch multiplicity | `ComposesDispatchAndCustomGateExpansionBudgets` |

The integration tests use public compiler APIs and treat every stage as
required. They do not encode expected failures in the source corpus. Features
accepted by QC emission are required to pass the demonstrated compiler path.
Features that cannot retain their semantics through that path fail with a
source-located diagnostic before a QC module is returned.
