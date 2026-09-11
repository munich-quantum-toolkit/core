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
`maxOperations` limits the number of inserted QC operations (10,000,000 by
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

| OpenQASM concept           | Support and restrictions                                                                                                                                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Versions and includes      | Versionless input and versions 2.0, 3.0, and 3.1 are accepted within the supported subset below. `stdgates.inc`, `qelib1.inc`, and nested textual includes are supported.                                                                                                 |
| Classical types            | `bit`, `bool`, `int`, `uint`, and `float` declarations are supported, including integer widths 1–64. Initialized compile-time `angle[N]` values support widths 1–52. Other sized numeric declarations, general arrays, complex values, and aliases are not yet supported. |
| Outputs                    | Explicit `output` declarations are preserved in source order. Without any explicit output, global classical variables become outputs.                                                                                                                                     |
| Gates                      | Language gates, the standard libraries, custom gates, broadcasting, and `inv`, `ctrl`, `negctrl`, and `pow` modifiers are supported. Custom definitions remain private QC functions instead of being expanded at every use. Recursive custom gates are rejected.          |
| Quantum statements         | Measurement, reset, barrier, logical qubits, and physical qubits are supported. The QC translation rejects programs that mix logical allocation with physical qubits.                                                                                                     |
| Expressions                | Scalar arithmetic, comparisons, Boolean expressions, and the supported math functions are type checked before translation. Initialized bit registers support `~`, `&`, `\|`, `^`, `<<`, `>>`, `popcount`, `rotl`, and `rotr`.                                             |
| Structured control         | `if`, `switch`, supported range-based `for`, and `while`. `break` exits the innermost enclosing loop; `continue` advances to its next iteration. Both may appear inside conditional and switch bodies.                                                                    |
| Dynamic indexing           | Classical bit indices can be dynamic and receive runtime bounds checks. A nonconstant qubit index must be a proven affine expression as described below.                                                                                                                  |
| Unsupported language areas | Subroutines, `extern`, calibration and timing constructs, input declarations, and arbitrary arrays are diagnosed.                                                                                                                                                         |

Sized `uint[N](bits)` and `int[N](bits)` casts accept an initialized `bit[N]`
register when the constant width is 1 through 64. Bit zero is the least
significant bit. Signed casts use two's-complement representation, with bit
`N - 1` as the sign bit.

`float(value)` and `float[64](value)` accept numeric and Boolean values.
Bit-string literals contain binary digits, optionally separated by underscores,
and must match the destination register width.

Textual includes use LLVM SourceMgr lookup: paths are tried relative to the
process working directory, then in the include directories supplied to
SourceMgr. They are not resolved relative to the including file. The built-in
standard libraries do not require files on disk.

Syntax and semantic diagnostics retain source locations and include stacks.
Classical-index bounds and integer-power preconditions are represented
explicitly in QC. Runtime integer arithmetic uses machine-width promotion and
wraps modulo that width; explicit integer casts truncate or extend to their
declared width. Compile-time invalid arithmetic is diagnosed. Runtime division
by zero remains undefined. Dynamic-index checks are supported by compiler/QIR
paths but remain outside the source-export subset.

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
angle inputs or outputs are not supported.

The frontend accepts a nonconstant qubit index only when it proves that every
value is in the register and that operands of one gate or explicit barrier are
distinct. Proven expressions can contain constants, positive constant-step `for`
induction variables, known scalar values, negation, addition, subtraction,
multiplication by an integer constant, and value-preserving `int`/`uint` casts.
Assignments and control-flow joins preserve a scalar value only while its affine
form remains known. A nested loop bound can use proven induction variables from
enclosing loops. The proof treats an inclusive range as its full interval and
does not use the step's congruence.

The frontend normalizes constant negative indices relative to the register
width. It rejects measurement-derived values, nonconstant negative indices,
nonlinear expressions, unsupported integer operators, and ranges whose step is
not known to be positive when their induction variable reaches a qubit index.
Mutations in repeating loops and unequal branch values invalidate scalar facts.
Branch conditions do not add proof facts. Classical bit indexing and loops that
do not index qubits keep their runtime behavior.

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
| Qubits and classical bits | Logical and physical qubits, scalar qubit allocations, static rank-one qubit memrefs, and CBit registers. Qubit memory indices must resolve statically. CBit indices can be dynamic.                                                                                |
| Quantum operations        | Measurement, reset, barrier, deallocation, global phase, and QC unitary operations. The exporter uses standard gates where available; for example, `sxdg` becomes `inv @ sx` and `u` and `u2` use a shared helper that compensates the OpenQASM 3 `U` global phase. |
| Reusable gates            | Private functions with leading `f64` parameters followed by scalar qubit arguments and no results. Straight-line `mqt.unitary` functions use `qc.call`; loop-containing gate functions use `func.call`.                                                             |
| Gate modifiers            | Nested `ctrl`, `inv`, and `pow`. A multi-operation modifier body with target qubits becomes a private generated gate.                                                                                                                                               |
| Scalar values             | Integers of widths 1–64, `f64`, and internal `index` values, including arithmetic, comparisons, Boolean operations, value-preserving casts, and supported math functions.                                                                                           |
| Structured control        | `scf.if`, `scf.index_switch`, signed `scf.for` with positive constant steps, and general two-region `scf.while`, including scalar arguments and results. Index switches use native `switch`, `case`, and `default`.                                                 |
| Results                   | Multiple scalar and bit-register outputs using the canonical type and naming rules below.                                                                                                                                                                           |

For loops preserve their exclusive upper bound when exported to an inclusive
OpenQASM range. Dynamic bounds are supported in the entry function and are
evaluated before the loop. An empty-range guard prevents underflow when
converting the upper bound. Loop-carried values retain their initial values when
the range is empty.

On import, dynamic ranges with signed bounds and a positive signed constant step
use 64-bit arithmetic when their bodies contain no `break` or `continue`. The
loop tests the unsigned distance to the inclusive endpoint before continuing, so
a final increment that wraps cannot cause another iteration. This supports round
trips without restricting the signed range of the endpoints. Other range forms
can still require wider arithmetic or runtime checks that the exporter rejects.

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
proof and QC emission to 10,000,000 inserted operations. Textual expansion is
limited to 1,000,000 statements and 1,000,000 file-include expansions, including
empty files. Standard-library includes count as statements. Include nesting is
limited to 64 levels. Exceeding any bound produces a diagnostic and no program.

The exporter does not reconstruct the runtime checks created for dynamic indices
or checked integer arithmetic. Surviving assertions, checked-index control flow,
or live poison values cause an explicit diagnostic. Programs with static qubit
and bit indices and supported integer/Boolean casts can be exported and parsed
again through the strict frontend. Integer-to-floating-point conversions support
this round trip. Compile-time floating-point-to-integer conversions remain
outside the input subset. Floating-point `!=` uses unordered-or-not-equal
semantics, including NaNs; ordered-not-equal MLIR comparisons are rejected.
Programs that rely on the input safety machinery must continue through another
output path such as QIR.
