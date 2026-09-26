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

The default output checkpoint is `--emit=qc`, which emits QC MLIR after the
default compiler pipeline. Use `--emit=qc-import` to inspect the imported QC
before cleanup or QCO optimization. MLIR is the representation shared by the QC,
QCO, and `jeff` dialects, so each output checkpoint names its dialect.

### Input support

| OpenQASM concept           | Support and restrictions                                                                                                                                                                                                                                         |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Versions and includes      | Versionless input and versions 2.0, 3.0, and 3.1 are accepted within the supported subset below. `stdgates.inc`, `qelib1.inc`, and nested textual includes are supported.                                                                                        |
| Classical types            | `bit`, `bool`, `int`, `uint`, and `float` declarations are supported, including integer widths 1–64. Initialized compile-time `angle[N]` values support widths 1–52. Other sized scalar declarations, complex values, and aliases are not yet supported.         |
| Outputs                    | Explicit `output` declarations are preserved in source order. Without any explicit output, global scalars and bit registers become outputs.                                                                                                                      |
| Gates                      | Language gates, the standard libraries, custom gates, broadcasting, and `inv`, `ctrl`, `negctrl`, and `pow` modifiers are supported. Custom definitions remain private QC functions instead of being expanded at every use. Recursive custom gates are rejected. |
| Quantum statements         | Measurement, reset, barrier, logical qubits, and physical qubits are supported. The QC translation rejects programs that mix logical allocation with physical qubits.                                                                                            |
| Expressions                | Scalar arithmetic, comparisons, Boolean expressions, and the supported math functions are type checked before translation. Initialized bit registers support `~`, `&`, `\|`, `^`, `<<`, `>>`, `popcount`, `rotl`, and `rotr`.                                    |
| Structured control         | `if`, `switch`, supported range-based `for`, and `while`. `break` exits the innermost enclosing loop; `continue` advances to its next iteration. Both may appear inside conditional and switch bodies.                                                           |
| Dynamic indexing           | Classical bit and array indices can be dynamic and must remain in bounds. A nonconstant qubit index must be a proven affine expression as described below.                                                                                                       |
| Classical arrays           | Global, fixed-size arrays of `bool`, `int`, `uint`, `float`, and `angle` with up to seven dimensions, as described below.                                                                                                                                        |
| Unsupported language areas | Subroutines, `extern`, calibration and timing constructs, and input declarations are diagnosed.                                                                                                                                                                  |

Sized `uint[N](bits)` and `int[N](bits)` casts accept an initialized `bit[N]`
register when the constant width is 1 through 64. Bit zero is the least
significant bit. Signed casts use two's-complement representation, with bit
`N - 1` as the sign bit.

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
by zero remains undefined.

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

Fixed-width angles are a compile-time input feature. An omitted angle width
resolves to 52 bits. Both `const angle[N]` and initialized `angle[N]`
declarations are accepted as write-once values. Initializers and angle casts
must be compile-time expressions. MQT Core supports float-to-angle conversion,
angle resizing, unary negation, addition and subtraction, multiplication and
division by nonnegative integer literals that fit the angle width, comparisons,
and `sin`, `cos`, and `tan`. Mixed-width angle operands promote to the wider
width. It uses round-to-nearest, ties-to-even for float conversion and
narrowing. Runtime angle state, reassignment, bit-level angle operations, and
angle inputs or outputs are not supported. Angle arrays support runtime indexing
of compile-time entries as described below.

### Classical arrays

OpenQASM 3 declarations such as `array[int[8], 3] values = {1, 2, 3};` allocate
mutable storage. Arrays must be global and can have up to seven dimensions; each
dimension must be a non-negative compile-time integer; zero extents are
supported. Each array and the combined number of array and register elements are
limited to 100,000. Integer element widths range from 1 through 64; `float` and
`float[64]` use double precision. Nested initializer lists must match every
declared dimension. For example, `array[int, 2, 3] a = {{1, 2, 3}, {4, 5, 6}};`
creates two rows of three elements. Without an initializer, elements are
undefined.

Element reads, assignments, and compound assignments accept constant or runtime
integer indices, with one comma-separated index per dimension (`a[1, 2]`).
Negative indices count from the end of each dimension (`a[-1, -1]` is the last
element of the last row). Constant out-of-range indices are diagnosed; dynamic
indices must remain in bounds in each dimension. A static read requires that
element to be initialized. A dynamic read requires every element to be
initialized; writing a dynamic index does not establish definite initialization.

Array initialization (`array[int, 2, 3] b = a;`) and assignment (`b = a;`) copy
values into independent storage. A prefix of scalar indices selects a subarray:
for a two-dimensional `a`, `array[int, 3] row = a[i];` copies one row, while
`a[i] = row;` or `a[i] = a[j];` replaces one row. Indices can be runtime or
negative values and have the same bounds preconditions as element access. Shapes
and element types, including widths, must match. Static copies require only the
selected source elements initialized and mark the selected destination elements
initialized. A runtime source index requires its whole array initialized; a
runtime destination index does not establish new initialization facts. Copies
also preserve quantized angle values. Array arithmetic and compound assignments
are not supported.

Copies also accept inclusive ranges in any dimension: `a[1:3]`, `a[0:2:4]`, or
`a[:, 1]`. Bounds and steps may be runtime integers. Negative bounds count from
the end; a negative step reverses the selection. Omitted endpoints select the
dimension's ends in the step's direction, so `a[:-1:]` reverses it. Ranges
retain their dimensions, while scalar indices remove them. Statically empty
ranges, zero steps, and out-of-bounds endpoints are rejected. Overlapping
assignments such as `a[1:] = a[:3];` for a five-element array copy the original
source values. Runtime ranges require in-bounds endpoints, nonzero steps, a
direction that produces a nonempty range, and matching copy shapes. Steps must
fit in a signed 64-bit integer. These are runtime preconditions; no assertions
are emitted. Runtime-selected sources require the whole source array
initialized; runtime-selected destinations establish no new
definite-initialization facts.

Use `a ++ b` to concatenate arrays or slices in an initializer or assignment.
Concatenation joins the first retained dimension; element types and all
remaining dimensions must match. Chained and parenthesized concatenations,
runtime ranges, and repeated operands are supported. Assignments such as
`a = a[2:] ++ a[:1];` preserve the original source values.

`sizeof(a)` returns the first dimension's length as a `uint`; `sizeof(a, d)`
selects a zero-based, compile-time integer dimension. Subarrays and ranges are
supported: `sizeof(a[0])` gives a matrix's row length. The query does not read
array elements, so the array need not be initialized. Use it in declarations or
loop bounds, such as `[0:sizeof(a)-1]`. A statically known extent is a
compile-time constant. A runtime-sized range, such as `sizeof(a[:i])`, returns a
runtime value and cannot set a fixed array size or a `const` initializer. Its
queried range has the same bounds and step preconditions as array copies.

Iterate a one-dimensional array or slice with `for int x in values { ... }`. The
loop variable is local and holds a copy of each element; assigning to it does
not change the array. Elements are read in selection order, so writes to later
elements are visible in later iterations. Slice bounds are evaluated once before
the loop. Empty arrays execute no iterations; `break` and `continue` work as in
range loops. Multidimensional arrays require a rank-one selection, such as
`for float x in matrix[row, :-1:] { ... }`.

Loop variables support the array element types and scalar numeric conversions,
including sized integers and `float[64]`. An `angle` loop variable requires an
angle array with no wider precision; assignments to it retain the compile-time
angle restriction below. Iteration requires the selected elements initialized,
or the whole array for runtime selections.

Angle arrays use the same widths and compile-time quantization as scalar angle
declarations. Values in initializer lists and assignments to individual entries
must be compile-time float or angle expressions, but the element index can be
dynamic. Use `float(angles[i])` for arithmetic on an entry's value in radians.
This angle-to-float cast is an intentional extension to OpenQASM. Runtime
fixed-width angle arithmetic and mixed-type comparisons are diagnosed. For
example:

```openqasm3
OPENQASM 3.0;
include "stdgates.inc";
array[angle[32], 3] angles = {0.0, pi / 2, pi};
qubit q;
for int i in [0:2] {
  ry(angles[i]) q;
}
```

Arrays lower to typed MLIR `memref.alloca` stack storage. The aggregate element
limit bounds the array payload to 800,000 bytes per entry-point invocation.
Overlapping copies reuse at most one equally sized scratch allocation per array,
so the combined payload is at most 1,600,000 bytes, independent of loop
iterations. Compiler cleanup uses upstream SROA and mem2reg to remove eligible
storage. Arrays whose storage and computations disappear can also export to Base
QIR, jeff, and OpenQASM.

Remaining arrays work through QC/QCO and Adaptive QIR conversion. Adaptive QIR
uses native LLVM stack storage, with no host C allocation or assertion runtime.
The emitted capability flags include arrays and element types, and
conservatively require loop support for residual dynamic indices. See the
[QIR 2.1 array contract](https://github.com/qir-alliance/qir-spec/blob/2.1/specification/Memory_Management.md#array-support).

OpenQASM export preserves fixed-size stack arrays, element reads and writes, and
array copies, including strided slices and overlapping assignments. It captures
loaded values before later writes and emits copies as array assignments. The
exporter normalizes views on a copy of the IR; it does not change the input
module. Integer storage preserves its bit patterns, and angle storage exports as
floating-point radians.

jeff export flattens fixed-size stack arrays in row-major order and preserves
element access, copies, and updates through structured control flow. It reuses
the classical-register SSA conversion and lowers copies to loops. Integer
elements use jeff's supported widths; non-native widths retain their bit
semantics. Import restores internal reference storage, copying only when an old
array value remains live. Unresolved runtime assertions cannot be exported.

Arrays are internal storage, not implicit outputs; assign selected elements to
scalar or bit outputs when needed. Array outputs and runtime angle conversion
are not yet supported. Classical subroutines and array-reference parameters
remain separate work. Gate definitions cannot capture mutable arrays; pass
selected entries as gate parameters instead.

### Qubit indices and classical registers

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
scalar facts. Branch conditions do not add proof facts. Classical bit indexing
and loops that do not index qubits keep their runtime behavior.

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

Reads through nonconstant classical indices require the whole source register to
be initialized. Such writes do not prove whole-register initialization,
including measurement writes. Constant selections track initialization per bit.

For convenience, import also accepts three-part slices with an omitted final
bound and measurement between a scalar and a one-element register. Export
expands slices and emits scalar measurements that follow the OpenQASM
[range grammar](https://openqasm.com/grammar/index.html) and
[measurement types](https://openqasm.com/language/insts.html#measurement).

Bit registers use `!cbit.reg<N>` in QC. OpenQASM 2 initializes each register to
zero. OpenQASM 3 leaves each register undefined until a statement writes it. A
static read requires its bit to be initialized. A dynamic read requires the
whole register to be initialized; writing one dynamic index does not prove that
a later dynamic read accesses an initialized bit. Whole-register reads and
writes lower to `cbit.read` and `cbit.write`. Standard integer operations
represent computation, including all comparisons: `cbit.read` produces the
snapshot, `arith.constant` the comparison constant, and `arith.cmpi` determines
signedness. CBit operations carry storage memory effects. jeff legalization
preserves native widths and promotes other widths up to 64 to 8, 16, 32, or 64
bits, masking results to retain exact-width semantics. Wider
register-versus-constant comparisons remain supported; wider general integer
expressions are rejected. Integer-to-floating-point casts (for example, using a
runtime population count as a rotation angle) remain outside the jeff subset.
Explicit outputs and implicit global outputs are returned by the entry function;
internal CBit allocations are not outputs. Other scalar outputs use builtin MLIR
scalar types. Programs without classical outputs have a void entry function; a
returned integer zero is ordinary output data. A scalar `qubit` lowers to
`qc.alloc`, while `qubit[1]` remains a one-element qubit register.

## Export OpenQASM

The exporter prints validated QC and SCF operations. The translation is
failure-atomic: it prepares the complete source before writing to the requested
stream.

Use the translation API for a `ModuleOp`:

```cpp
#include "mqt/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"

auto source = mlir::qc::translateQCToOpenQASM3(moduleOp);
if (mlir::failed(source)) {
  // An MLIR diagnostic describes the unsupported operation.
}
```

The compiler API returns an owned textual program:

```cpp
auto qc = mlir::QCProgram::fromOpenQASMFile("input.qasm");
auto direct = qc->toOpenQASM3(); // Export without QCO optimization.
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
{code}`mlir::QCProgram::toOpenQASM3` applies the QC cleanup pipeline but
bypasses that QCO optimization round trip.

### Export and round-trip support

| QC or MLIR concept        | Export support                                                                                                                                                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qubits and classical bits | Logical and physical qubits, scalar qubit allocations, static rank-one qubit memrefs, and CBit registers. Logical qubit and CBit indices can be dynamic. Mapped programs require static physical qubits; indexed tensor loops must be specialized before export.    |
| Quantum operations        | Measurement, reset, barrier, deallocation, global phase, and QC unitary operations. The exporter uses standard gates where available; for example, `sxdg` becomes `inv @ sx` and `u` and `u2` use a shared helper that compensates the OpenQASM 3 `U` global phase. |
| Reusable gates            | Private functions with leading `f64` parameters followed by scalar qubit arguments and no results. Straight-line `mqt.unitary` functions use `qc.call`; loop-containing gate functions use `func.call`.                                                             |
| Gate modifiers            | Nested `ctrl`, `inv`, and `pow`. A multi-operation modifier body with target qubits becomes a private generated gate.                                                                                                                                               |
| Scalar values             | Integers of widths 1–64, `f64`, and internal `index` values, including arithmetic, comparisons, Boolean operations, value-preserving casts, and supported math functions.                                                                                           |
| Constant tables           | Reads from non-empty constant rank-one `f64` tensors. Constant indices and splats become scalar values; other reads use switches with shared cases for equal values.                                                                                                |
| Structured control        | `scf.if`, `scf.index_switch`, signed `scf.for` with positive constant steps, and general two-region `scf.while`, including scalar arguments and results. Index switches use native `switch`, `case`, and `default`.                                                 |
| Results                   | Multiple scalar and bit-register outputs using the canonical type and naming rules below.                                                                                                                                                                           |

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
target-specific diagnostic. Export preserves the source module and buffers
output until it succeeds.

The exporter writes an OpenQASM 3.1 version declaration and includes
`stdgates.inc`. Gates in MQT Core's compatibility catalog, such as `r`, `rzz`,
and `ecr`, receive definitions under their catalog names. Strict consumers use
those definitions. MQT Core's default compatibility mode recognizes a definition
with the catalog name and signature and imports calls directly as the
corresponding native QC operation; the definition body is deliberately ignored.
A same-name definition with a mismatched signature is rejected. Strict mode
always analyzes the custom definition normally.

The `_mqt_` prefix is reserved for generated composite-modifier gates,
temporaries, and collision-safe identifiers. Existing classical-register
allocation names are reused when valid and distinct from catalog gates; scalar
output names are generated deterministically.

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

Export requires one defined, argument-free entry function. Additional functions
must be private, defined gate functions with leading `f64` parameters followed
by scalar qubit arguments and no results. Gate functions may contain supported
scalar expressions, quantum operations, calls, and loops. Measurement, reset,
barrier, allocation, classical storage, conditionals, switches, and
`arith.select` are rejected in gate functions. For-loop bounds in gate functions
must remain constant. Gate while loops require a pure condition region and no
loop-carried values.

The exporter rejects arbitrary CFGs, multi-block SCF regions, recursive or
unresolved calls, dynamic qubit indices, dynamic for-loop steps, unsigned
`scf.for` comparisons, general memrefs, unsupported integer widths, unknown
operations, and non-unitary content inside modifier regions. CBit loads, stores,
whole-register reads and writes, fixed-width bitwise operations, dynamic bit
indices, SCF results, and loop-carried values are supported in the entry
function. Multi-operation modifier bodies must have a target qubit and cannot
capture additional qubits from an enclosing scope.

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
use one exact-width bit-string initializer. Shift interpretation is determined
by the MLIR operation, not by the history of its operands. Arithmetic right
shifts are encoded with unsigned bitwise operations and explicit sign-bit
biasing.

Inline expressions, including those in gate functions, have a nesting limit of
256 and an expansion budget of 4,096 values per expression. The total width of
classical registers, including wide snapshots, is limited to 1,048,576 bits.
Import limits affine proofs to 256 levels and 4,096 distinct expressions per
proof and QC emission to 1,000,000,000 inserted operations. Textual expansion is
limited to 100,000,000 statements and 1,000,000 file-include expansions,
including empty files. Standard-library includes count as statements. Include
nesting is limited to 64 levels. Exceeding any bound produces a diagnostic and
no program.

The exporter rejects unsupported operations, including explicit `cf.assert`
operations and live poison values. It does not silently discard them. Programs
with supported qubit indices, dynamic bit indices, and integer/Boolean casts can
be exported and parsed again through the strict frontend.
Integer-to-floating-point conversions support this round trip. Compile-time
floating-point-to-integer conversions remain outside the input subset.
Floating-point `!=` uses unordered-or-not-equal semantics, including NaNs;
ordered-not-equal MLIR comparisons are rejected.
