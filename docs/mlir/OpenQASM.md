# OpenQASM input and output

MQT Core accepts OpenQASM as a compiler input and can export structured programs
from the QC dialect.

The [OpenQASM specification](https://openqasm.com/index.html) defines the
language. This page describes the subset supported by MQT Core.

## Import OpenQASM

The frontend parses and validates the source before translating it directly to
QC. The C++ compiler API accepts strings and files:

```cpp
auto fromString = mlir::QCProgram::fromOpenQASMString(source);
auto fromFile = mlir::QCProgram::fromOpenQASMFile("program.qasm");
```

The lower-level `mlir::qc::translateOpenQASMToQC` importer accepts
`OpenQASMImportOptions`. Its `gatePolicy` field selects the gate policy, and
`maxOperations` limits the number of inserted QC operations (1,000,000,000 by
default). Exceeding the limit emits a diagnostic and returns no program.

Python provides the corresponding constructors:

```python
from mqt.core.mlir import QCProgram

from_string = QCProgram.from_openqasm_str(source)
from_file = QCProgram.from_openqasm_file("program.qasm")
qiskit_circuit = QCProgram.from_openqasm_str(source).to_qiskit()
```

Python convenience functions such as `compile_program` recognize OpenQASM source
strings by the `OPENQASM` header. For versionless source, construct a
`QCProgram` with `from_openqasm_str` first. The explicit constructors and
`.qasm` file imports accept versionless input directly.

`mqt-cc` recognizes `.qasm` files automatically. Use `--input-format=openqasm`
when the filename does not identify the format:

```console
mqt-cc program.qasm
mqt-cc --input-format=openqasm program.txt
```

The generated entry function is named `main`, or `mqt.entry` when a source gate
is named `main`. It is identified by `mqt.entry_point`, not by its name. The dot
keeps the fallback name outside OpenQASM's identifier syntax. Export preserves
legal custom gate names, including names beginning `_mqt_`.

The default output checkpoint is `--emit=qc`, which emits QC MLIR after the
default compiler pipeline. Use `--emit=qc-import` to inspect the imported QC
before cleanup or QCO optimization. MLIR is the representation shared by the QC,
QCO, and `jeff` dialects, so each output checkpoint names its dialect.

### Input support

| OpenQASM concept           | Support and restrictions                                                                                                                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Versions and includes      | Versionless input and versions 2.0, 3.0, and 3.1 are accepted within the supported subset below. `stdgates.inc`, `qelib1.inc`, and nested textual includes are supported.                                                                                          |
| Classical types            | `bit`, `bool`, `int`, `uint`, and `float` declarations are supported, including integer widths 1–64 and `float[64]`. Initialized compile-time `angle[N]` values support widths 1–52. Other sized scalar declarations, complex values, and aliases are unsupported. |
| Inputs                     | Global `input float name;` and `input float[64] name;` become named `f64` entry-point arguments in declaration order. Other input types and widths are diagnosed; names beginning `_mqt_` are reserved. Inputs are not implicit outputs.                           |
| Outputs                    | Explicit `output` declarations are preserved in source order. Without any explicit output, global scalars and bit registers become outputs.                                                                                                                        |
| Gates                      | Language gates, the standard libraries, custom gates, broadcasting, and `inv`, `ctrl`, `negctrl`, and `pow` modifiers are supported. Custom definitions remain private QC functions instead of being expanded at every use. Recursive custom gates are rejected.   |
| Quantum statements         | Measurement, reset, barrier, logical qubits, and physical qubits are supported. The QC translation rejects programs that mix logical allocation with physical qubits.                                                                                              |
| Expressions                | Scalar arithmetic, comparisons, Boolean expressions, and the supported math functions are type checked before translation. Initialized bit registers support `~`, `&`, `\|`, `^`, `<<`, `>>`, `popcount`, `rotl`, and `rotr`.                                      |
| Structured control         | `if`, `switch`, supported range-based `for`, and `while`. `break` exits the innermost enclosing loop; `continue` advances to its next iteration. Both may appear inside conditional and switch bodies.                                                             |
| Dynamic indexing           | Classical bit and array indices can be dynamic and must remain in bounds. A nonconstant qubit index must be a proven affine expression as described below.                                                                                                         |
| Classical arrays           | Global, fixed-size arrays of `bool`, `int`, `uint`, `float`, and `angle`, with up to seven dimensions and the scalar widths above.                                                                                                                                 |
| Unsupported language areas | Subroutines, `extern`, calibration and timing constructs, and other array forms are diagnosed.                                                                                                                                                                     |

The importer recognizes custom gates containing unitary gate calls and
constant-range `for` loops as unitary, including nested loops and their callers.
This classification is conservative: an unclassified gate may have a unitary
quantum action. A modifier on a custom gate requires that the gate and its
callees contain no loops.

Import accepts quantum register widths from 1 through 2⁶³ − 1 and bit-register
widths from 1 through 16,777,215 (MLIR's integer-width limit). Available memory,
later compiler stages, and [export limits](#export-limitations) also constrain
usable register sizes.

Declare a classical array with `array[int[8], 3] values = {1, 2, 3};`. Arrays
have one through seven nonnegative compile-time dimensions. Nested initializer
lists must match each dimension; for example,
`array[int, 2, 3] a = {{1, 2, 3}, {4, 5, 6}};`. Zero-length dimensions are
valid. Each dimension and the total classical array element count are limited to
100,000.

Whole-array initialization (`array[int, 2, 3] b = a;`) and assignment (`b = a;`)
copy values into independent storage. Leading scalar indices select subarrays:
`array[int, 3] row = a[0];`, `a[1] = row;`, and `a[0] = a[1];`. Source and
destination must have matching remaining dimensions and element types, including
their widths. All selected source elements must be initialized. Whole-array
arithmetic, compound assignments, and whole-array inputs or outputs are
unsupported. To pass or return array values, select individual entries as gate
parameters or scalar outputs. Gate definitions can query array dimensions but
cannot access array elements.

Use inclusive ranges to select entries in any dimension: `a[1:3]`, `a[0:2:4]`,
or `a[:, 1]`. Bounds and steps may be runtime integers. Negative bounds count
from the end; omitted endpoints use the dimension's ends in the step's
direction, so `a[:-1:]` reverses it. Ranges retain dimensions; scalar indices
remove them. Constant empty ranges, zero steps, and out-of-bounds endpoints are
rejected, except a full selection of a zero-length dimension. Overlapping
assignments copy the original source values.

Runtime ranges require in-bounds endpoints, nonzero steps that fit signed i64,
matching copy shapes, and a nonempty selection in the step's direction, except
for full selections of zero-length dimensions. These are runtime preconditions;
no assertions are emitted. Runtime-selected sources must be fully initialized,
and runtime-selected destinations establish no new definite-initialization
facts.

Use `a ++ b` to concatenate arrays or slices in an initializer or assignment.
Concatenation joins the first retained dimension; element types and all
remaining dimensions must match. Chained and parenthesized concatenations,
runtime ranges, and repeated operands are supported. Assignments such as
`a = a[2:] ++ a[:1];` preserve the original source values.

`sizeof(a)` returns the first dimension's length as a `uint`; `sizeof(a, d)`
selects a zero-based, compile-time integer dimension. Subarrays and ranges are
supported, including zero-length dimensions. The query does not read array
elements, so the array need not be initialized. The queried extent is constant
when known at compile time. A runtime-sized range, such as `sizeof(a[:i])`,
returns a runtime value and cannot set a fixed array size or a `const`
initializer. Its queried range has the same bounds and step preconditions as
array copies.

Iterate a rank-one array or slice with `for int x in values { ... }`, or
`for float x in matrix[row, :-1:] { ... }`. The local variable copies each
element in selection order; writes to later elements remain visible. Bounds are
evaluated once. Empty arrays skip the body; `break` and `continue` work as in
range loops. Selected elements must be initialized, or the whole array for
runtime selections. Loop variables support scalar numeric conversions and sized
integers. An `angle` loop variable must have at least the array element width.
Its assignments must be compile-time values.

Inputs share the parameter API with Qiskit programs. They remain symbolic until
bound, so a program can be inspected and transformed before choosing values:

```python
source = "OPENQASM 3.1; input float theta; qubit q; ry(theta) q;"
program = QCProgram.from_openqasm_str(source)
assert program.parameters == ["theta"]
bound = program.copy()
bound.bind_parameters({"theta": 0.5})
assert bound.parameters == []
```

Binding substitutes values in QC or QCO; a target that requires fixed gate
angles still needs compilation after binding for each value. Exporting an
unbound QC program writes `input float[64]` declarations and preserves their
names. OpenQASM carries names, but not Qiskit parameter identities or vector
grouping. QIR Base and Adaptive lowering require inputs to be bound first.

Sized `uint[N](bits)` and `int[N](bits)` casts accept an initialized `bit[N]`
register when the constant width is 1 through 64. Bit zero is the least
significant bit. Signed casts use two's-complement representation, with bit
`N - 1` as the sign bit. For wider registers, immediately narrow the value to at
most 64 bits, for example `uint[64](uint[150](bits))`. Narrowing retains the low
bits; the outer cast determines their signedness. Wide scalar variables and
arithmetic are unsupported.

`float(value)` and `float[64](value)` accept numeric and Boolean values.
Bit-string literals contain binary digits, optionally separated by underscores,
and must match the destination register width. Leading zeros count toward that
width; the rightmost digit initializes bit zero. For example,
`bit[6] b = "00_1101"` has value 13, and `b[3:-1:1]` has value `"011"`: the
first selected bit becomes bit zero of the slice. These are the OpenQASM
[bit-register conventions](https://openqasm.com/language/types.html#classical-bits-and-registers).

Textual includes use LLVM SourceMgr lookup: paths are tried relative to the
process working directory, then in the include directories supplied to
SourceMgr. They are not resolved relative to the including file. The built-in
standard libraries do not require files on disk.

Syntax and semantic diagnostics retain source locations and include stacks. The
importer diagnoses statically invalid inputs and emits no runtime assertions.
Runtime classical indices must be in bounds after negative-index wrapping, range
steps must be nonzero, and integer arithmetic powers require nonnegative
exponents. Integer `pow` modifier exponents must be exactly representable as
`f64`, the QC/QCO exponent type. These are program preconditions; violating them
has no guaranteed diagnostic or result.

Runtime integer arithmetic, including powers, uses machine-width promotion and
wraps modulo that width. Explicit integer casts truncate or extend to their
declared width. Compile-time invalid arithmetic is diagnosed. Runtime division
by zero remains undefined. Range endpoints and steps use the same integer
promotion before iteration.

OpenQASM 3 supports all six comparisons between fixed-width bit-register
expressions. Direct register comparisons use unsigned meaning. An exact-width
`int[N]` cast selects signed two's-complement interpretation before the
language's usual integer promotion. The frontend also accepts these conditions
in OpenQASM 2 as a compatibility extension; version-specific initialization and
gates are unchanged.

Runtime shift distances have unsigned interpretation. Overshifts produce zero.
The frontend checks the original distance before narrowing it and uses a safe
count even in the unselected shift. Constant distances fold without guards. The
same helper is used by Qiskit import.

For Qiskit-generated source, nonnegative constant operands of typed bitwise
expressions are accepted when they fit the unsigned operand width. Standalone
unsized constant bitwise expressions use the 64-bit machine width. This does not
give runtime signed integers an implicit unsigned interpretation.

For the same compatibility reason, a whole-register assignment accepts a
nonnegative integer constant that fits the register width. Use an exact-width
bit-string literal for strict OpenQASM source.

Scalar fixed-width angles hold compile-time values. An omitted angle width
resolves to 52 bits. Both `const angle[N]` and initialized `angle[N]`
declarations are accepted as write-once values. Initializers and angle casts
must be compile-time expressions. MQT Core supports float-to-angle conversion,
angle resizing, unary negation, addition and subtraction, multiplication and
division by nonnegative integer literals that fit the angle width, comparisons,
and `sin`, `cos`, and `tan`. Mixed-width angle operands promote to the wider
width. It uses round-to-nearest, ties-to-even for float conversion and
narrowing. Scalar angle reassignment, runtime fixed-width angle arithmetic,
bit-level angle operations, and angle inputs or outputs are unsupported. Angle
arrays use the same widths and quantization. Their indices can be dynamic, but
initializers and assigned values must be compile-time float or angle
expressions. Use `float(angles[i])` for arithmetic or mixed-type comparisons on
an entry's value in radians; this cast is an MQT Core extension to OpenQASM.

### Indexing and initialization

Read or assign array elements with one comma-separated integer index per
dimension (`a[1, 2]`). Negative indices count from the end of each dimension
(`a[-1, -1]` is the last element of the last row). The frontend diagnoses
constant out-of-range indices; runtime indices must stay in bounds in each
dimension.

The frontend accepts a nonconstant qubit index only when it proves that every
value is in the register and that operands of one gate or explicit barrier are
distinct. Proven expressions can contain constants, positive constant-step `for`
induction variables, known scalar values, negation, addition, subtraction,
multiplication by an integer constant, and value-preserving `int`/`uint` casts.
Assignments and control-flow joins preserve a scalar value only while its affine
form remains known. A nested loop bound can use proven induction variables from
enclosing loops. The proof treats an inclusive range as its full interval and
does not use the step's congruence.

The frontend normalizes negative indices relative to the register width when it
can prove their sign. It rejects measurement-derived values, indices whose sign
is unknown, nonlinear expressions, unsupported integer operators, and ranges
whose step is not known to be positive when their induction variable reaches a
qubit index. Mutations in repeating loops and unequal branch values invalidate
scalar facts. Branch conditions do not add proof facts. Classical bit and array
indexing and loops that do not index qubits keep their runtime behavior.

Register operands support inclusive slices `q[first:last]` and
`q[first:step:last]`, including negative indices and negative steps. Omitted
endpoints select the register ends in the step's direction. With the default
step, `q[:last]` means `q[0:last]` and includes `last`; `q[:2]` selects qubits
`0`, `1`, and `2`. Constant slices expand in selection order. Gates broadcast
over slices; a slice does not supply multiple control arguments to `ctrl(n)`.

Slice lengths and steps must be statically known. Bounds may use the same proven
affine expressions as scalar qubit indices. For example, `q[i:i]` selects
`q[i]`, and `q[i:i+1]` selects two qubits when both are in bounds. A slice
remains a register operand, so all register operands of a broadcast must have
the same length, including one-element slices. Measurement-selected ranges and
unproved lengths, bounds, or operand distinctness are rejected during analysis.
Slice lowering adds no runtime assertions or wider integer arithmetic.

Classical slices support fixed-width bit-vector expressions and assignments.
Their first selected bit becomes bit zero of the value. Assignments snapshot the
source before writing, so overlapping copies are safe. Constant selections use
scalar bit operations; selecting a whole register in order uses a register read
or write. Compound assignments to indexed bits or slices are not supported.

OpenQASM 3 requires each bit or array element to be initialized before it is
read. A read through a nonconstant index requires the whole source register or
array to be initialized. Such writes do not prove whole-register or whole-array
initialization, including measurement writes. Constant selections track
initialization per element.

For convenience, import also accepts three-part slices with an omitted final
bound and measurement between a scalar and a one-element register. Export
expands slices and emits scalar measurements that follow the OpenQASM
[range grammar](https://openqasm.com/grammar/index.html) and
[measurement types](https://openqasm.com/language/insts.html#measurement).

Bit registers use `!cbit.reg<N>` in QC. OpenQASM 2 initializes each register to
zero. Whole-register reads and writes lower to `cbit.read` and `cbit.write`.
Standard integer operations represent computation, including all comparisons:
`cbit.read` produces the snapshot, `arith.constant` the comparison constant, and
`arith.cmpi` determines signedness. CBit operations carry storage memory
effects. jeff legalization preserves native widths and promotes other widths up
to 64 to 8, 16, 32, or 64 bits, masking results to retain exact-width semantics.
Wider register-versus-constant comparisons remain supported; wider general
integer expressions are rejected. Integer-to-floating-point casts (for example,
using a runtime population count as a rotation angle) remain outside the jeff
subset. Explicit outputs and implicit global outputs are returned by the entry
function; internal CBit allocations are not outputs. Other scalar outputs use
builtin MLIR scalar types. Programs without classical outputs have a void entry
function; a returned integer zero is ordinary output data. A scalar `qubit`
lowers to `qc.alloc`, while `qubit[1]` remains a one-element qubit register.

## Export OpenQASM

The exporter prints validated QC and SCF operations. It writes to the requested
stream only after preparing the complete source.

Use the translation API with an owned MLIR module:

```cpp
#include "mqt/Target/OpenQASM/TranslateQCToOpenQASM3.h"

auto source = mlir::qc::translateQCToOpenQASM3(std::move(moduleOp));
if (mlir::failed(source)) {
  // An MLIR diagnostic describes the unsupported operation.
}
```

The compiler API consumes its input and returns an owned textual program. Copy
the input explicitly when it is needed by another compilation:

```cpp
auto qc = mlir::QCProgram::fromOpenQASMFile("input.qasm");
auto direct = qc->copy().intoOpenQASM3(); // Keep qc for the optimized path.
direct->write("direct.qasm");
auto reimported = mlir::runDefaultPipeline(
    mlir::CompilerInput{*direct}, mlir::ProgramFormat::QCImport);

auto optimized = mlir::runDefaultPipeline(
    mlir::CompilerInput{std::move(*qc)}, mlir::ProgramFormat::OpenQASM3);
```

Python exposes both forms:

```python
from mqt.core.mlir import OutputFormat, QCProgram, compile_program

qc = QCProgram.from_openqasm_file("input.qasm")
direct = qc.to_openqasm3()
print(direct.source)
direct.write("direct.qasm")

optimized = compile_program("input.qasm", output=OutputFormat.OPENQASM3)
optimized.write("optimized.qasm")
```

The command-line driver writes to standard output unless `-o` is given:

```console
mqt-cc input.qasm --emit=openqasm3
mqt-cc input.qasm --emit=openqasm3 -o optimized.qasm
```

The compiler-pipeline path performs target compilation when requested, runs the
QCO optimization pipeline, converts back to QC, and then exports. Calling
{py:meth}`~mqt.core.mlir.QCProgram.to_openqasm3` or
{code}`mlir::QCProgram::intoOpenQASM3` applies the QC cleanup pipeline but
bypasses that QCO optimization round trip. Both consume the input, including on
failure. In Python, set `copy=True` to preserve it. The low-level translation
API also consumes its module; only the output stream is unchanged on failure.

### Export and round-trip support

| QC or MLIR concept        | Export support                                                                                                                                                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qubits and classical bits | Logical and physical qubits, scalar qubit allocations, static rank-one qubit memrefs, and CBit registers. Logical qubit and CBit indices can be dynamic. Mapped programs require static physical qubits; indexed tensor loops must be specialized before export.    |
| Quantum operations        | Measurement, reset, barrier, deallocation, global phase, and QC unitary operations. The exporter uses standard gates where available; for example, `sxdg` becomes `inv @ sx` and `u` and `u2` use a shared helper that compensates the OpenQASM 3 `U` global phase. |
| Program parameters        | Named `f64` entry-point arguments become `input float[64]` declarations in argument order. Names must be unique, valid OpenQASM identifiers that do not conflict with standard gates or use the reserved `_mqt_` prefix.                                            |
| Reusable gates            | Private single-block functions with leading `f64` parameters followed by scalar qubit arguments and no results. Unitary functions, including statically bounded `for` loops over unitary gates, use `qc.call`. Other structured gate functions use `func.call`.     |
| Gate modifiers            | Nested `ctrl`, `inv`, and `pow`. A multi-operation modifier body with target qubits becomes a private generated gate.                                                                                                                                               |
| Scalar values             | Integers of widths 1–64, `f64`, and internal `index` values, including arithmetic, comparisons, Boolean operations, value-preserving casts, and supported math functions.                                                                                           |
| Constant tables           | Reads from non-empty constant rank-one `f64` tensors. Constant indices and splats become scalar values; other reads use switches with shared cases for equal values.                                                                                                |
| Structured control        | `scf.if`, `scf.index_switch`, signed `scf.for` with positive constant steps, and general two-region `scf.while`, including scalar arguments and results. Index switches use native `switch`, `case`, and `default`.                                                 |
| Results                   | Multiple scalar and bit-register outputs using the canonical type and naming rules below.                                                                                                                                                                           |

DD simulation and Adaptive QIR support array storage. Adaptive QIR uses native
stack storage, bounded by the frontend element limit to 800,000 bytes of array
payload per invocation, or 1,600,000 bytes including copy scratch storage.
Targets must support arrays and their element types; dynamic indices also
require loop support. See the
[QIR conversion contracts](Conversions.md#qir-conversions).

OpenQASM export supports fixed-size stack arrays, strided and overlapping
copies, and concatenation. Arrays support one to seven dimensions and up to
100,000 elements in total, including temporary arrays whose copies cannot be
folded into one assignment. Export preserves loaded values across writes,
retains integer bit patterns, and emits angles as floating-point radians. Dense
constant initialization becomes an array declaration initializer; it must write
a whole entry-block allocation before any other use. Reimport also requires the
combined register-and-array limit described above.

jeff export flattens fixed-size stack arrays in row-major order and carries
updates through control flow. Import uses reference storage with copies for live
old values; array-valued interfaces stay tensor-based. Lengths must fit signed
i32. Unresolved safety assertions are unsupported. Across mutating control flow,
live old array values and repeated operands require numeric literals or zero
arrays, which initialize independent storage. Loop-carried stack arrays must
keep their incoming storage.

Qiskit and Base QIR export require array storage to be eliminated. Compiler
cleanup promotes eligible arrays to scalar values and folds constants. With
LLVM/MLIR 23.1, promotion requires at most 16 elements, constant indices, and
supported uses. Call `program.cleanup()` before `to_qiskit()`; Qiskit export
itself only folds scalar expressions.

Cleanup also promotes local CBit registers of at most 256 bits to Boolean SSA
values. Accesses must use constant indices, whole-register writes, or
whole-register reads of at most 64 bits. Wider reads also participate when all
reads of the local register feed constant comparisons: cleanup evaluates these
comparisons bit by bit at the snapshot point, avoiding wide packed arithmetic.
Other wide reads retain storage. Branches and loops carry promoted values
through their results and block arguments. Returned registers, register
arguments, dynamic accesses, and opaque register users retain storage. Reads of
undefined bits before their first store remain undefined behavior; promotion may
choose zero for those values. Whole-register reads preserve snapshots, and
returned registers preserve their names and bit order. The promotion bound does
not limit output register widths.

Export retains private gate-shaped helper functions. Other defined helpers,
including functions with integer or register arguments or classical results, are
inlined using MLIR's inliner. Their control flow remains structured. Recursive
calls, non-gate helpers marked `no_inline`, and constructs outside the export
subset still produce diagnostics. Retained gates forward their `f64` parameters
directly to gates or other retained helpers. Helpers that compute with these
parameters, including `pow` exponents, are inlined because OpenQASM gate
parameters are angles. Constant expressions and calculations using loop indices
remain supported in retained gates. The exporter does not emit OpenQASM `def`
subroutines.

Signed and unsigned index casts preserve sign extension, zero extension, and
truncation, including truncation to one bit. Indices use 64 bits. Addition,
subtraction, and multiplication of index or 64-bit integer values use signed
expressions when range analysis proves they cannot overflow. This preserves
affine bounds through negative temporaries, including bounds of empty nested
loops. Other integer arithmetic keeps its modular unsigned representation.
Private integer temporaries use `uint[n]`; index temporaries use `int[64]`.
Assignments use OpenQASM's implicit numeric conversions. Explicit casts preserve
signed operations, integer widths, and conversions from bit registers. Scalar
outputs keep the canonical types below.

For loops preserve their exclusive upper bound when exported to an inclusive
OpenQASM range. Dynamic bounds are supported in the entry function and are
evaluated before the loop. An empty-range guard prevents underflow when
converting the upper bound. Loop-carried values retain their initial values when
the range is empty.

On import, inclusive integer ranges use 64-bit arithmetic, including unsigned
bounds, negative steps, `break`, and `continue`. The loop tests the unsigned
distance to the endpoint before continuing, so a final increment that wraps
cannot cause another iteration. Range lowering does not require wider integer
types that the exporter rejects.

Entry-function while loops use `while (true)` and a conditional `break`, so
condition-region expressions are evaluated once per iteration. Gate functions
retain direct `while (condition)` syntax because they cannot declare local
state. The before and after regions may have different argument counts and
types. Local variables preserve scalar state, exit values, and simultaneous
updates such as swaps. The condition and its forwarded values are evaluated
before any continuation updates.

For example, this terminating do-while form executes its body three times:

```openqasm3
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
int count = 0;
while (true) {
  x q;
  count += 1;
  if (!(count < 3)) { break; }
}
```

Import recognizes this form structurally and produces one `scf.while` with a
forwarding after region. No additional conditional is needed solely to exit.
Initialization facts include every reachable break: a value assigned before all
exits of this do-while form is initialized afterwards. An ordinary while loop
may execute zero times. Constant-true loops remain valid; translation does not
prove termination.

Resources must be allocated outside loops and conditionals. Unsupported scalar
types, operations, captures, or runtime gate parameters produce a
target-specific diagnostic without writing partial output.

The exporter writes an OpenQASM 3.1 version declaration and includes
`stdgates.inc`. Controlled gates use compact standard-library names such as
`cx`, `cz`, and `ccx` when available, and `ctrl` modifiers otherwise. Gates in
MQT Core's compatibility catalog, such as `r`, `rzz`, and `ecr`, receive
definitions under their catalog names. Strict consumers use those definitions.
MQT Core's default compatibility mode recognizes a definition with the catalog
name and signature and imports calls directly as the corresponding native QC
operation; the definition body is deliberately ignored. A same-name definition
with a mismatched signature is rejected. Strict mode always analyzes the custom
definition normally.

The compatibility spelling `u(theta, phi, lambda)` has the exact QC and Qiskit
U-gate matrix. The standardized `U`, `u2`, `u3`, and `cu3` spellings retain
their OpenQASM phase conventions; use `u3` for the legacy OpenQASM 2 U matrix.

The `_mqt_` prefix is reserved for generated composite-modifier gates,
temporaries, and collision-safe identifiers. Existing classical-register
allocation names are reused when valid and distinct from catalog gates; scalar
output names are generated deterministically. Scalar qubits and scoped classical
storage reuse valid, nonconflicting source-name hints from MLIR `NameLoc`
locations. Explicit parameter and register names take precedence over these
hints. If locations are discarded, exporters generate names for those resources;
parameter binding and output labels remain unchanged.

Output types follow a deliberately small canonical mapping:

| QC result                         | OpenQASM output |
| --------------------------------- | --------------- |
| Returned `!cbit.reg<N>`           | `bit[N]`        |
| `i1` produced directly by measure | `bit`           |
| Other `i1`                        | `bool`          |
| `i64` or `index`                  | `int`           |
| Other integers of 2–63 bits       | `uint[N]`       |
| `f64`                             | `float`         |

Outputs preserve function-result order, including mixed scalar and register
results and constant-zero integers. Returning the same register more than once
is diagnosed because OpenQASM outputs cannot preserve that aliasing. Programs
without results keep generated classical temporaries in a local scope to avoid
implicit outputs. Unused measurement results need no temporary.

Measurements write directly to a named register when their result has one store
and that store can occur at the measurement without crossing a conflicting
classical access. This also preserves registers for grouped measurements after
routing. Measurement order is unchanged; other cases retain temporary bits.

Import and export do not preserve `uint`, fixed-angle spelling or width,
scalar-versus-one-element bit spelling, or scalar output names. Integer
computations use explicit `int[N]`/`uint[N]` casts, so signedness is chosen by
each MLIR operation rather than inferred from its source register. Truncation,
sign/zero extension, arithmetic, bitwise operations, comparisons, shifts, and
integer selection are supported. Selection uses a fixed-width bit mask and does
not allocate a temporary register. The frontend accepts the casts and
expressions emitted by the exporter, including Boolean/integer conversions.

### Export limitations

Export requires one defined entry function whose arguments are named `f64`
inputs. Functions that remain after helper inlining must be private, defined
gate functions with leading `f64` parameters followed by scalar qubit arguments
and no results. Gate functions may contain supported scalar expressions, quantum
operations, calls, and loops. Measurement, reset, barrier, allocation, classical
storage, conditionals, switches, and `arith.select` are rejected in gate
functions. For-loop bounds in gate functions must remain constant. Gate while
loops require a pure condition region and no loop-carried values.

The exporter rejects arbitrary CFGs, multi-block SCF regions, recursive or
unresolved calls, dynamic for-loop steps, unsigned `scf.for` comparisons,
general memrefs, unsupported integer widths, unknown operations, and non-unitary
content inside modifier regions. CBit loads, stores, whole-register reads and
writes, fixed-width bitwise operations, dynamic bit indices, SCF results, and
loop-carried values are supported in the entry function. Multi-operation
modifier bodies must have a target qubit and cannot capture additional qubits
from an enclosing scope. Their generated gate parameters carry evaluated gate
angles; supporting integer calculations remain at the call site. A `pow` nested
inside a generated composite gate requires a constant exponent. A runtime
exponent applied to the whole composite remains at its call site.

OpenQASM export supports arbitrary bit-register widths for bitwise operations,
unsigned comparisons, `popcount`, `rotl`, and `rotr`. Scalar arithmetic, integer
casts, signed comparisons, and logical shifts require widths of at most 64 bits.
Rotation counts must be constant or represented by at most 64 bits, optionally
zero-extended to the register width. Qiskit interoperability uses the common
subset described in the Python compiler documentation.

The exporter stores used scalar expressions and bit-register reads in local
variables at their definition. These variables preserve snapshots across later
writes and nested regions and prevent repeated expansion of shared expressions.
Wide bit-vector expressions use local bit registers. Zero-initialized registers
use one exact-width bit-string initializer. Constants with more than 64 active
bits use hexadecimal literals. Shift interpretation is determined by the MLIR
operation, not by the history of its operands. Arithmetic right shifts are
encoded with unsigned bitwise operations and explicit sign-bit biasing.

Inline expressions, including those in gate functions, have a nesting limit of
256 and an expansion budget of 4,096 values per expression. The total width of
classical registers, including wide snapshots, is limited to 1,048,576 bits.
Import limits affine proofs to 256 levels and 4,096 distinct expressions per
proof and QC emission to 1,000,000,000 inserted operations. Textual expansion is
limited to 100,000,000 statements and 1,000,000 file-include expansions,
including empty files. Standard-library includes count as statements. Include
nesting is limited to 64 levels. Each expanded register operand or barrier is
limited to 100,000,000 references, checked before allocation. Exceeding any
bound produces a diagnostic and no program.

The exporter rejects unsupported operations, including explicit `cf.assert`
operations and live poison values. It does not silently discard them. Programs
with supported qubit indices, dynamic bit indices, and integer/Boolean casts can
be exported and parsed again through the strict frontend.
Integer-to-floating-point conversions support this round trip. Compile-time
floating-point-to-integer conversions remain outside the input subset.
Floating-point `!=` uses unordered-or-not-equal semantics, including NaNs;
ordered-not-equal MLIR comparisons are rejected.
