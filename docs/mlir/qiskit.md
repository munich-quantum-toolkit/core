# Qiskit compatibility

MQT Core exposes two distinct Qiskit interfaces:

| Interface                                                | Supported Qiskit versions | Purpose                                                          |
| -------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------- |
| {doc}`QDMI backend <../qdmi/qdmi_backend>`               | 2.1 and newer             | Execute circuits through device-specific serializers.            |
| {doc}`MQT Compiler Collection <mqt_compiler_collection>` | `>=2.5.0,<2.6.0`          | Import and export circuits through the versioned native adapter. |

Install `mqt-core[qiskit]`. Direct compiler translation requires a version in
the narrower range above; the adapter checks it before inspecting a circuit.

## Circuit translation contract

Each output block owns one private Python circuit. Numeric instructions use a
borrowed C API view; symbolic gates, classical expressions, and control flow use
Python construction. Blocks share their parent's exact bits and lexical variable
captures. Parameters and parameter vectors are created once per export.

| Circuit feature                                                         | Import               | Export                             |
| ----------------------------------------------------------------------- | -------------------- | ---------------------------------- |
| Standard gates, constructible numeric modifiers, and global phase       | Supported            | Supported                          |
| Other finite numeric modifiers                                          | Supported            | Rejected                           |
| Measurement, reset, and barrier                                         | Supported            | Supported                          |
| Canonical named registers and leading loose bits                        | Supported            | Explicit registers                 |
| Custom Gates with finite, acyclic definitions                           | Reusable functions   | Custom Gates                       |
| Generic instructions with finite, acyclic definitions                   | Recursively expanded | Expanded operations                |
| Nested `if`/`else`, `for`, `while`, and `switch`                        | Supported            | Supported                          |
| Classical-bit and register conditions                                   | Supported            | Supported                          |
| Constant Boolean, `Uint` up to 64 bits, and `Float` expressions         | Supported            | Supported                          |
| Clbit and ClassicalRegister expression variables                        | Supported            | Supported                          |
| Fixed-width bitwise operations, comparisons, and bounded shifts         | Supported            | Supported                          |
| Direct complete-register comparisons wider than 64 bits                 | Supported            | Supported                          |
| Clbit, indexed-register, and whole-register `Store` assignments         | Supported            | Supported                          |
| Initialized local classical variables and enclosing captures            | Supported            | Native `expr.Var` and `Store`      |
| External runtime input variables                                        | Rejected             | Rejected                           |
| `break` in `for` and `while`, including nested conditionals/switches    | Supported            | Native `BreakLoopOp`               |
| `continue` in `for` and `while`, including nested conditionals/switches | Supported            | Preserved through SCF control flow |
| Free symbols and supported real parameter expressions                   | Supported            | Supported                          |
| Parameter-vector elements                                               | Supported            | Supported                          |
| Dense numeric unitaries up to eight qubits                              | Supported            | Supported                          |
| Register aliases or interleaved membership                              | Rejected             | Rejected                           |
| Transpiler layout metadata                                              | Accepted and ignored | Not emitted                        |

Classical-expression variables may refer to Clbits or ClassicalRegisters in the
containing circuit. This includes values used only by the condition or switch
target and not by a control-flow block. External runtime inputs remain
unsupported.

Private gate parameters bind by position and receive generated local names
during Qiskit export; their original names and grouping are not preserved.
Public program inputs still require explicit names. OpenQASM custom gates can
therefore use `QCProgram.from_openqasm_str(source).to_qiskit()` directly within
the supported subset below.

Export folds scalar expressions on a copy of the QC program. Constant
arithmetic, casts, and idempotent expressions can therefore disappear;
expression-tree shape is not preserved. Quantum-resource and classical-snapshot
canonicalization patterns are not applied, because they can change circuit width
or introduce scratch bits. Call `cleanup()` explicitly when those broader
transformations are wanted. Live free parameters retain their identities; unused
named program inputs remain unsupported.

Free symbols become named {code}`f64` program inputs. Parameter-vector elements
retain their grouping and index, preserving vector order and positional binding
across a round trip; similarly named standalone parameters remain standalone.
Elements used in different structured-control blocks are restored into one
shared vector for the complete circuit tree. Free parameter vectors and their
combined declared size in one translated circuit are each limited to 65,536
elements. Parameter-expression trees support at most 64 levels and 4,096 nodes.
Import and export support real addition, subtraction, multiplication, division,
power, negation, trigonometric and inverse trigonometric functions, exponential,
logarithm, absolute value, and real conjugation. Export also folds signed and
unsigned integer-to-float casts of constants. Other parameter-expression
functions are rejected. Lexically bound {code}`for`-loop induction parameters
are supported and remain distinct from free symbols. Parameterized
custom-instruction definitions are expanded after their symbols and expressions
are resolved. Definition expansion rejects missing definitions, cycles, operand
arity mismatches, nesting beyond 64 levels, and more than 10 million expanded
operations.

Structured-control export supports scalar results from {code}`scf.if` and
{code}`scf.index_switch`, carried scalar state in constant-range
{code}`scf.for`, and general two-region {code}`scf.while`. Ordinary conditions
retain native while-loop conditions. More general loops use a constant-true
`WhileLoopOp`, before-region instructions, a conditional `BreakLoopOp`, and
after-region instructions. The condition's values become the loop results on
exit and the after-region arguments on continuation. Edge updates preserve
parallel-assignment semantics, including swaps and unequal before/after tuples.

`break` and `continue` target the innermost loop. `continue` skips the remaining
body, advances a for-loop iterator, and then reevaluates the loop condition.
Both imports use SCF control flow, so export preserves behavior without
requiring the original jump statement to survive normalization.

Local scalar state covers Boolean values, integers of widths 1–64, and `f64`
using supported backend operations. It uses native `expr.Var` declarations,
`Store` assignments, and captures in nested circuits. Compiler variables do not
increase the classical-bit or register counts. Import accepts initialized local
variables and captures from enclosing supported circuits. External runtime
inputs remain unsupported. Runtime classical values cannot be used as symbolic
gate parameters.

Canonical short-circuit Boolean expressions retain their native expression form.
A live {code}`scf.for` induction value must reduce to an affine {code}`f64` gate
parameter. The exporter preserves one Qiskit parameter identity for that value
throughout its lexical body. An {code}`scf.index_switch` selector must be a
constant index or a supported Boolean/Uint expression converted with
{code}`arith.index_castui`. Switch labels must be nonnegative constants that fit
the target width.

Nested blocks may capture existing qubits, classical bits, and local variables
but may not allocate or release circuit resources. Control flow and classical
expressions may nest up to 64 levels, and classical expression trees may contain
at most 16,384 nodes (parameter-expression limits are unchanged). Integer values
use exact widths from 1 through 64. The only wider form is a direct unsigned
comparison between one complete `ClassicalRegister` and one same-width literal;
computed, packed, and signed wide values remain rejected. Both expression
conditions and tuple conditions such as `if_test((register, value))` support
this form. Tuple equalities with a value outside the register range become
false. Standard `arith.cmpi` handles every comparison: signed ordering is
encoded by XOR-biasing both operands' sign bits, including computed operands.
Casts preserve truncation and sign/zero extension. Bitwise operations, modular
arithmetic, integer selection, and shifts share these typed rules. Import guards
runtime shifts so overshifts produce zero; export preserves the guards.
Rotations and population count are expanded through the same bounded integer
lowering used by jeff. Unsupported operations, invalid widths, non-finite
constants, unsupported index uses, and dynamic for-loop bounds fail during
validation. Programs without classical outputs have a void entry function. For
compatibility, Qiskit export also ignores a lone constant-zero `i64` return.
Whole-register reads map to Qiskit `ClassicalRegister` expressions, and writes
map to atomic Qiskit `Store` operations. Indexed stores assume that their
runtime index is in bounds. The Qiskit C API does not expose `Store`, so the
adapter inspects and constructs that instruction through Qiskit's public Python
classes, as it already does for structured control flow. Internal entry-block
CBit storage becomes additional Qiskit registers, ordered before returned
registers; Qiskit exposes all circuit storage. OpenQASM remains the source
interchange path for arbitrary register widths.

Every public CBit output is exported as a Qiskit `ClassicalRegister`; an unnamed
allocation receives a collision-free `_mqt_cN` name. This preserves the CBit
register boundary and gives whole-register writes a valid Qiskit lvalue. Loose
input Clbits therefore round trip semantically, but not as loose output bits.

Conditions and switch targets may read a zero-initialized CBit register. An
undefined CBit may be read only after a definite write to that bit, and every
bit of an undefined returned register must be definitely initialized. Branches
intersect their initialization facts. A while loop's before region executes at
least once; its after region may execute zero times. The exporter saves
supported scalar snapshots in local variables when a later write, control-flow
edge, or region crossing prevents safe re-evaluation. It bounds expression depth
by saving intermediate runtime values. This policy does not depend on unused
control-flow results and remains valid after compiler cleanup. Reads wider than
64 bits remain subject to the snapshot checks.

Each exported measurement must write to one static public CBit in the same
block. Destinations may be reused; later measurements overwrite earlier values
in program order. Constants, unitary quantum operations (including barriers),
and resets may separate a measurement from its destination store. The exporter
keeps the measurement at its original position and writes the destination there.
Other intervening operations, including classical accesses and control flow, are
rejected because this earlier write may change the program's meaning. The
measurement result may feed supported classical expressions after that store.
Live measurement results are saved in local variables before later writes.
Deferred measurement expressions require an unchanged destination CBit.

Dense numeric unitaries remain explicit matrix operations during import and
export. Target compilation synthesizes supported one- and two-qubit matrices to
the target gate set. Dense unitary operations support at most eight qubits.
Qiskit import preserves inverse, numeric power, and closed-control modifiers on
dense-unitary operations. Export preserves inverse and closed-control modifiers.
Other powers require canonicalization or synthesis.

A circuit remains valid when {code}`circ.layout` is present. The importer
translates the circuit operations and deliberately does not preserve physical or
virtual layout metadata.

Names passed between Qiskit and the compiler must not contain NUL characters.
The importer checks names before native access. Arithmetic-progression loop
lists without jumps use range lowering. List loops with jumps and
variable-bearing switch cases use balanced dispatch. The 64-level nesting limit
also applies to generated SCF, and expansion limits account for duplicated
switch bodies.

Input validation finishes before an MLIR module is created. Generic output
validation finishes before Qiskit construction starts; the version-specific
adapter validates its constructed blocks before returning the top-level circuit.
Unsupported programs therefore fail without modifying the source object or
exposing a partial result.

The binding imports Qiskit only when circuit translation is requested. It
accepts versions in the registered {code}`>=2.5.0,<2.6.0` range and verifies the
native API version before reading a circuit.
