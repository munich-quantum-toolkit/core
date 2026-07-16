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
| `if`, inclusive `for`, and `while` | Supported | Gate bodies contain gate calls and loops over gate calls only | `PreservesImportedLoopAndDynamicIndexBehavior` |
| Expressions and scalar math functions | Supported | Operations are checked against the implemented scalar type rules | `EmitsAllScalarOperatorsAndComparisonPredicates` |
| Dynamic qubit and bit indexing | Supported | Structured dispatch is bounded to 4096 leaves | `DispatchesDynamicQubitGatesWithStructuredControlFlow` |
| Primitive, broadcast, and custom gates | Supported | Recursive definitions and mismatched broadcast widths are rejected | `BroadcastsRegistersAlongsideScalarQubits` |
| `inv`, `ctrl`, `negctrl`, and `pow` modifiers | Parsed and semantically checked | Target support differs below | `RejectsInvalidGateControlAndBroadcastShapes` |

## Translation and compiler support

“Adaptive plus Jeff” means the tested public path from QC through optimized QCO,
Jeff byte serialization and deserialization, back to QC, and finally to Adaptive
QIR. Base refers to direct production of the QIR Base Profile.

| Feature | Parse | Semantics | QC | Adaptive plus Jeff | Base | Restriction or rejection reason | Representative test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Primitive and custom gates | Supported | Supported | Supported | Supported | Supported | Custom gates are expanded during QC emission | `broadcast_custom_gate` |
| Gate arithmetic and math parameters | Supported | Supported | Supported | Supported | Supported | Scalar `pow()` is distinct from the gate modifier | `math_parameters` |
| Broadcast gates | Supported | Supported | Supported | Supported | Supported | Operands must have compatible widths | `broadcast_custom_gate` |
| `inv`, `ctrl`, and `negctrl` | Supported | Supported | Supported | Supported | Not in the tested Base subset | Structured custom-gate modifiers are rejected when QC cannot preserve them | `mixed_controls` |
| `pow @` | Supported | Supported | Rejected | Rejected | Rejected | The QC dialect has no power modifier yet; translation reports the source location | `RejectsPowerAtTheQCTargetBoundary` |
| `if` and nested `if`/`for` | Supported | Supported | Supported | Supported | Adaptive only | The Base corpus is intentionally straight-line | `nested_static_control_flow` |
| Measurement-controlled `while` | Supported | Supported | Supported | Supported | Adaptive only | Requires runtime classical control | `measurement_controlled_while` |
| Loop-carried mutable bit state | Supported | Supported | Supported | Supported | Adaptive only | Carried state remains SSA values across QC, QCO, Jeff, and QIR | `mutable_loop_state` |
| Dynamic indexing resolved by optimization | Supported | Supported | Supported | Supported | Not in the tested Base subset | Runtime bounds checks must be removable before Jeff serialization | `resolved_dynamic_index` |
| General runtime dynamic indexing | Supported | Supported | Supported | Not yet supported | Not yet supported | Jeff has no representation for the emitted runtime bounds assertion | `DispatchesDynamicQubitGatesWithStructuredControlFlow` |
| Measurement, reset, and barrier | Supported | Supported | Supported | Supported | Measurement and barrier supported | Reset is Adaptive-only | `reset`, `barrier` |

The integration tests use public compiler APIs and treat every stage as
required. They do not encode expected failures in the source corpus. Features
outside the full-pipeline column remain useful at the QC boundary, but callers
must not assume that every downstream format can represent them yet.
