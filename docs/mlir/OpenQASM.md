# OpenQASM input and output

MQT Core accepts OpenQASM as a compiler input and can export structured programs
from the QC dialect.

The [OpenQASM specification](https://openqasm.com/index.html) defines the
language. This page describes the subset supported by MQT Core.

## Import OpenQASM

The frontend parses and validates the source before translating it directly to
QC. The C++ compiler API accepts strings and files:

```cpp
auto fromString = mlir::QCProgram::fromQASMString(source);
auto fromFile = mlir::QCProgram::fromQASMFile("program.qasm");
```

Python provides the corresponding constructors:

```python
from mqt.core.mlir import QCProgram

from_string = QCProgram.from_qasm_str(source)
from_file = QCProgram.from_qasm_file("program.qasm")
```

`mqt-cc` recognizes `.qasm` files automatically. Use `--input-format=qasm` when
the filename does not identify the format:

```console
mqt-cc program.qasm
mqt-cc --input-format=qasm program.txt
```

### Input support

| OpenQASM concept           | Support and restrictions                                                                                                                                                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Versions and includes      | Versionless input and versions 3.0 and 3.1 use the maintained OpenQASM profile. `stdgates.inc`, `qelib1.inc`, and nested textual includes are supported.                                                                                           |
| Classical types            | Unsized `bit`, `bool`, `int`, `uint`, and `float` declarations are supported. Initialized compile-time `angle[N]` values support widths 1 through 52. Other sized numeric declarations, arrays, complex values, and aliases are not yet supported. |
| Outputs                    | Explicit `output` declarations are preserved in source order. Without any explicit output, global classical variables become outputs.                                                                                                              |
| Gates                      | Language gates, the standard libraries, custom gates, broadcasting, and `inv`, `ctrl`, `negctrl`, and `pow` modifiers are supported. Recursive custom gates are rejected.                                                                          |
| Quantum statements         | Measurement, reset, barrier, logical qubits, and physical qubits are supported. The QC target rejects programs that mix logical allocation with physical qubits.                                                                                   |
| Expressions                | Scalar arithmetic, comparisons, Boolean expressions, and the supported math functions are type checked before translation. Initialized bit registers support `~`, `&`, `\|`, `^`, `<<`, `>>`, `popcount`, `rotl`, and `rotr`.                      |
| Structured control         | `if`, inclusive `for`, `while`, and `switch` lower to SCF operations. Switch controls and case labels must be integers; labels must be unique constant expressions.                                                                                |
| Dynamic indexing           | Classical bit indices can be dynamic and receive runtime bounds checks. A nonconstant qubit index must be a proven affine expression as described below.                                                                                           |
| Unsupported language areas | Subroutines, `extern`, calibration and timing constructs, input declarations, arbitrary arrays, `break`, and `continue` are diagnosed.                                                                                                             |

Sized `uint[N](bits)` and `int[N](bits)` casts accept an initialized `bit[N]`
register when the constant width is 1 through 64. Bit zero is the least
significant bit. Signed casts use two's-complement representation, with bit
`N - 1` as the sign bit.

Syntax and semantic diagnostics retain source locations and include stacks.
Runtime integer preconditions and classical-index bounds are represented
explicitly in QC. This safety machinery is supported by the normal compiler and
QIR paths, but it is intentionally outside the export subset described below.

OpenQASM 3 supports all six comparisons between fixed-width bit-register
expressions. Direct register comparisons use unsigned meaning. An exact-width
`int[N]` cast selects signed two's-complement ordering. OpenQASM 2 retains its
equality-only register condition.

Runtime bit-register shift distances must be unsigned and less than the register
width. A scalar distance must have `uint` type. A bit-register expression of at
most 64 bits is interpreted as unsigned in this context for compatibility with
Qiskit output. The compiler folds larger constant distances to zero but assumes
that a nonconstant distance is in range. This range contract keeps the QC,
OpenQASM, and Qiskit representations identical without guarded shift operations.

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
zero. OpenQASM 3 leaves each register undefined until a statement writes it.
Whole-register reads and writes lower to `cbit.read` and `cbit.write`. Standard
integer operations represent fixed-width bitwise expressions. Direct
register-versus-constant comparisons lower to `cbit.cmp` and use unsigned
integer meaning. Comparisons of an exact-width `int[N]` or `uint[N]` register
cast with an in-range constant preserve the selected signed or unsigned meaning.
The jeff output path lowers `cbit.cmp`, but jeff cannot represent the arbitrary
fixed-width integers used by general `cbit.read` and `cbit.write` expressions.
Explicit outputs and implicit global outputs are returned by the entry function;
internal CBit allocations are not outputs. Other scalar outputs use builtin MLIR
scalar types. A scalar `qubit` lowers to `qc.alloc`, while `qubit[1]` remains a
one-element qubit register.

## Export OpenQASM

The exporter prints validated QC and SCF operations. The translation is
failure-atomic: it prepares the complete source before writing to the requested
stream.

Use the translation API for a `ModuleOp`:

```cpp
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"

auto source = mlir::qc::translateQCToOpenQASM3(moduleOp);
if (mlir::failed(source)) {
  // An MLIR diagnostic describes the unsupported operation.
}
```

The compiler API returns an owned textual program:

```cpp
auto qc = mlir::QCProgram::fromQASMFile("input.qasm");
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

qc = QCProgram.from_qasm_file("input.qasm")
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

| QC or MLIR concept        | Export support                                                                                                                                                                                                               |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qubits and classical bits | Logical and physical qubits, scalar qubit allocations, static rank-one qubit memrefs, and CBit registers. Qubit memory indices must resolve statically. CBit indices can be dynamic.                                         |
| Quantum operations        | Measurement, reset, barrier, deallocation, global phase, and QC unitary operations. The exporter uses standard gates where available; for example, `sxdg` becomes `inv @ sx` and `u2` uses the standard compatibility alias. |
| Gate modifiers            | Nested `ctrl`, `inv`, and `pow`. A multi-operation modifier body with target qubits becomes a private generated gate.                                                                                                        |
| Scalar values             | `i1`, `i64`, `f64`, and internal `index` values, including arithmetic, comparisons, Boolean operations, value-preserving casts, and supported math functions.                                                                |
| Structured control        | Result-free `scf.if` and `scf.index_switch`, constant-range `scf.for` without iterated state, and zero-state expression-based `scf.while`. Index switches use native `switch`, `case`, and `default` statements.             |
| Results                   | Multiple scalar and bit-register outputs using the canonical type and naming rules below.                                                                                                                                    |

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
| `f64`                             | `float`         |

A lone constant-zero `i64` result is treated as the frontend's status return and
is not emitted. Import and export do not preserve `uint`, fixed-angle spelling
or width, scalar-versus-one-element bit spelling, or scalar output names.
Unsigned constants therefore normalize to `int`. Generic scalar operations whose
signedness affects their meaning, such as unsigned division, comparison, or
conversion, are rejected instead of being approximated. Integer sign extension
and truncation are also rejected because OpenQASM scalar casts have different
value semantics. The `cbit.cmp` operation is the narrow exception and retains
signed or unsigned register semantics.

Emitted scalar casts use unsized standard OpenQASM conversion syntax. The MQT
Core frontend does not yet support these runtime casts, so cast-containing
output is outside the current MQT strict round-trip subset.

### Export limitations

Export accepts exactly one defined, argument-free function. It rejects calls,
arbitrary CFGs, multi-block SCF regions, dynamic qubit indices or ranges,
general memrefs, unsupported integer widths, unknown operations, and non-unitary
content inside modifier regions. CBit loads, stores, whole-register reads and
writes, fixed-width bitwise operations, and dynamic indices are supported. SCF
results, loop-carried values, nonempty `scf.yield`, and `arith.select` are
outside the export subset. Multi-operation modifier bodies must have a target
qubit and cannot capture additional qubits from an enclosing scope.

The OpenQASM path additionally supports arbitrary bit-register widths,
`popcount`, `rotl`, and `rotr`. Qiskit interoperability uses the common subset
described in the Python compiler documentation.

The exporter inlines a whole-register read only in the block that contains the
read and only when no later write to that register precedes the expression use.
It rejects stale and cross-region snapshots instead of reading newer register
state. A dynamic shift distance must retain provably unsigned provenance: a
bit-register expression of at most 64 bits or a bit-vector scalar such as
`popcount`. Signless scalar MLIR values are rejected because they cannot be
emitted as OpenQASM `uint` without changing their type.

Export accepts an expression nesting depth of at most 256 and an expansion
budget of 4,096 values per expression. The total width of classical registers is
limited to 1,048,576 bits.

The exporter does not reconstruct the runtime checks created for dynamic indices
or checked integer arithmetic. Surviving assertions, checked-index control flow,
or live poison values cause an explicit diagnostic. Programs with static qubit
and bit indices and without scalar casts can be exported and parsed again
through the strict frontend. Programs that rely on the input safety machinery
must continue through another output path such as QIR.

:::{important}
The compiler removes dead code. A circuit that only prepares a state has no
observable effect and may be removed by optimization. Measure the relevant
qubits and return the results when compiling a program for execution.
:::
