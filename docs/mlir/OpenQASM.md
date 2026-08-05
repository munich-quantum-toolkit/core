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

| OpenQASM concept           | Support and restrictions                                                                                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Versions and includes      | Versionless input and versions 3.0 and 3.1 use the maintained OpenQASM profile. `stdgates.inc`, `qelib1.inc`, and nested textual includes are supported.                                        |
| Classical types            | Unsized `bit`, `bool`, `int`, `uint`, and `float` declarations are supported. Width-qualified numeric types, arrays, complex values, and aliases are not yet supported.                         |
| Outputs                    | Explicit `output` declarations are preserved in source order. Without any explicit output, global classical variables become outputs.                                                           |
| Gates                      | Language gates, the standard libraries, custom gates, broadcasting, and `inv`, `ctrl`, `negctrl`, and `pow` modifiers are supported. Recursive custom gates are rejected.                       |
| Quantum statements         | Measurement, reset, barrier, logical qubits, and physical qubits are supported. The QC target rejects programs that mix logical allocation with physical qubits.                                |
| Expressions                | Scalar arithmetic, comparisons, Boolean expressions, and the supported math functions are type checked before translation. `popcount`, `rotl`, and `rotr` operate on initialized bit registers. |
| Structured control         | `if`, inclusive `for`, `while`, and `switch` lower to SCF operations. Switch controls and case labels must be integers; labels must be unique constant expressions.                             |
| Dynamic indexing           | Dynamic qubit and bit indices are supported on input. The generated QC includes bounds checks and structured dispatch where needed.                                                             |
| Unsupported language areas | Subroutines, `extern`, calibration and timing constructs, input declarations, arbitrary arrays, `break`, and `continue` are diagnosed.                                                          |

Syntax and semantic diagnostics retain source locations and include stacks.
Runtime integer preconditions and dynamic-index bounds are represented
explicitly in QC. This safety machinery is supported by the normal compiler and
QIR paths, but it is intentionally outside the export subset described below.

Bit outputs use `memref<nxi1>` in QC, including scalar `bit` as `memref<1xi1>`.
Other scalar outputs use builtin MLIR scalar types. A scalar `qubit` lowers to
`qc.alloc`, while `qubit[1]` remains a one-element qubit register.

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
| Qubits and classical bits | Logical and physical qubits, scalar allocations, and static rank-one qubit or `i1` memrefs. Memory indices must resolve statically.                                                                                          |
| Quantum operations        | Measurement, reset, barrier, deallocation, global phase, and QC unitary operations. The exporter uses standard gates where available; for example, `sxdg` becomes `inv @ sx` and `u2` uses the standard compatibility alias. |
| Gate modifiers            | Nested `ctrl`, `inv`, and `pow`. A multi-operation modifier body becomes a private generated gate.                                                                                                                           |
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
| `memref<Nxi1>`                    | `bit[N]`        |
| `i1` produced directly by measure | `bit`           |
| Other `i1`                        | `bool`          |
| `i64` or `index`                  | `int`           |
| `f64`                             | `float`         |

A lone constant-zero `i64` result is treated as the frontend's status return and
is not emitted. Import and export do not preserve `uint`, angle spelling,
scalar-versus-one-element bit spelling, or scalar output names. Unsigned
constants therefore normalize to `int`. Operations whose signedness affects
their meaning, such as unsigned division, comparison, or conversion, are
rejected instead of being approximated. Integer sign extension and truncation
are also rejected because OpenQASM scalar casts have different value semantics.

Emitted scalar casts use standard OpenQASM conversion syntax. The MQT Core
frontend does not yet parse that syntax, so cast-containing output is outside
the current MQT strict round-trip subset.

### Export limitations

Export accepts exactly one defined, argument-free function. It rejects calls,
arbitrary CFGs, multi-block SCF regions, dynamic indices or ranges, general
memrefs, unsupported integer widths, packed bit-vector operations, unknown
operations, and non-unitary content inside modifier regions. SCF results,
loop-carried values, nonempty `scf.yield`, and `arith.select` are outside the
export subset.

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
