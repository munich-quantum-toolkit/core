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

| OpenQASM concept           | Support and restrictions                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Versions and includes      | Versionless input and versions 3.0 and 3.1 use the maintained OpenQASM profile. `stdgates.inc`, `qelib1.inc`, and nested textual includes are supported.                                                                                                                                                                                                                                                                                                                                    |
| Classical types            | `bit`, `bool`, `int`, `uint`, `float`, and fixed-width `angle[N]` declarations are supported. `angle` and `uint` accept widths from 1 through 64; unsized values use 64 bits. `float[64]` is accepted explicitly. Other width-qualified numeric types, arrays, complex values, and aliases are not yet supported.                                                                                                                                                                           |
| Inputs and outputs         | Scalar `input` declarations and explicit `output` declarations are preserved in source order. Without any explicit output, global classical variables become outputs.                                                                                                                                                                                                                                                                                                                       |
| Gates                      | Language gates, the standard libraries, custom gates, broadcasting, and `inv`, `ctrl`, `negctrl`, and `pow` modifiers are supported. Recursive custom gates are rejected.                                                                                                                                                                                                                                                                                                                   |
| Quantum statements         | Measurement, reset, barrier, logical qubits, and physical qubits are supported. The QC target rejects programs that mix logical allocation with physical qubits.                                                                                                                                                                                                                                                                                                                            |
| Expressions                | Scalar arithmetic, comparisons, Boolean expressions, and the supported math functions are type checked before translation. Fixed-width angles use their OpenQASM unsigned-ring operations, promotion, comparison, trigonometric-input, and cast rules. Equal-width `bit[N]` and `uint[N]`/`angle[N]` casts are supported, as is single-bit rvalue indexing of integer and angle scalars. `popcount`, `rotl`, and `rotr` support initialized bit registers and scalar `uint`/`angle` values. |
| Structured control         | `if`, inclusive `for`, `while`, and `switch` lower to SCF operations. Switch controls and case labels must be integers; labels must be unique constant expressions.                                                                                                                                                                                                                                                                                                                         |
| Dynamic indexing           | Dynamic qubit and bit indices are supported on input. The generated QC includes bounds checks and structured dispatch where needed.                                                                                                                                                                                                                                                                                                                                                         |
| Unsupported language areas | Subroutines, `extern`, calibration and timing constructs, arbitrary arrays, scalar bit slices and indexed scalar assignment, `break`, and `continue` are diagnosed.                                                                                                                                                                                                                                                                                                                         |

Syntax and semantic diagnostics retain source locations and include stacks.
Runtime integer preconditions and dynamic-index bounds are represented
explicitly in QC. This safety machinery is supported by normal QC/QCO compiler
paths, but it is intentionally outside the OpenQASM export and strict QIR
subsets described below.

Bit outputs use `memref<nxi1>` in QC, including scalar `bit` as `memref<1xi1>`.
Other scalar outputs use builtin MLIR scalar types. A scalar `qubit` lowers to
`qc.alloc`, while `qubit[1]` remains a one-element qubit register.

### Fixed-width angles

An OpenQASM `angle[N]` stores an unsigned `N`-bit pattern representing a
multiple of $2\pi/2^N$. MQT Core keeps the source kind and width in the typed
frontend, then lowers its storage and arithmetic to builtin signless `iN`,
`arith`, and `scf` constructs.

QC and QCO gate parameters and the QIR QIS ABI remain `f64` radians. Canonical
integer-to-radian bridges are inserted only at gate and angle-aware math uses.
Float-to-angle conversion decomposes the source binary64 value and evaluates the
quotient against the exact binary64 representation of $2\pi$ with integer
arithmetic. This preserves the specified modulo and nearest, ties-to-even result
through 64 angle bits, including when the source has a large exponent. Angle
widening inserts zero least-significant bits; narrowing uses the same
ties-to-even rule. Widths above 64 are rejected because the existing gate and
QIR boundary is binary64.

The scalar forms of `popcount`, `rotl`, and `rotr` preserve fixed-width bit
semantics for `uint[N]` and `angle[N]`. They lower to `math.ctpop` and LLVM
funnel-shift operations. Whole `bit` registers continue to use their existing
packed-register lowering.

MQT Core always uses the OpenQASM-permitted ties-to-even narrowing behavior.
Pragmas selecting truncation instead of rounding are not supported.

The compiler records scalar input and output kind and source name in a
namespaced function argument/result attribute; the builtin integer type carries
the width. Together they distinguish an `angle[N]` interface from a `uint[N]`
interface while it traverses QC and QCO.

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

| QC or MLIR concept        | Export support                                                                                                                                                                                                                                                                      |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qubits and classical bits | Logical and physical qubits, scalar allocations, and static rank-one qubit or `i1` memrefs. Memory indices must resolve statically.                                                                                                                                                 |
| Quantum operations        | Measurement, reset, barrier, deallocation, global phase, and QC unitary operations. The exporter uses standard gates where available; for example, `sxdg` becomes `inv @ sx` and `u2` uses the standard compatibility alias.                                                        |
| Gate modifiers            | Nested `ctrl`, `inv`, and `pow`. A multi-operation modifier body with target qubits becomes a private generated gate.                                                                                                                                                               |
| Scalar values             | `i1`, arbitrary supported `iN`, `f64`, and internal `index` values, including signed and unsigned arithmetic, comparisons, bitwise operations, shifts, casts, and supported math functions. Canonical angle conversion and resize sequences emit official `angle[N]` casts.         |
| Structured control        | Single-block `scf.if`, `scf.index_switch`, constant-range `scf.for`, and expression-based `scf.while`. Supported scalar results and loop-carried state become typed OpenQASM temporaries with staged updates. Index switches use native `switch`, `case`, and `default` statements. |
| Results                   | Multiple scalar and bit-register outputs using the canonical type and naming rules below.                                                                                                                                                                                           |

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
| Other `iN`                        | `uint[N]`       |
| `f64`                             | `float`         |

A lone constant-zero `i64` result is treated as the frontend's status return and
is not emitted. Scalar interface metadata preserves `input` and `output`
`angle[N]` and `uint[N]` declarations across the QC/QCO round trip. Without that
metadata, `i64` results retain the established `int` default and narrower
integers emit as `uint[N]`.

Emitted scalar casts use standard OpenQASM conversion syntax. The MQT Core
frontend reparses the scalar casts used by the angle and unsigned-integer
subset. The round-trip contract is semantic: generated source may express erased
internal angle state as an equivalent `uint[N]` bit computation followed by an
exact `bit[N]` to `angle[N]` conversion.

### Export limitations

Export accepts exactly one defined function; scalar arguments require the
OpenQASM interface metadata produced by the frontend. It rejects calls,
arbitrary CFGs, multi-block SCF regions, dynamic indices or ranges, general
memrefs, integer widths above 64, packed bit-vector operations, unknown
operations, and non-unitary content inside modifier regions. SCF regions must
contain one block. `scf.for` requires constant bounds and a positive constant
step. An `scf.while` condition region must be side-effect free and forward its
carried state unchanged; updates belong in the loop body. Carried values are
limited to the supported scalar export types. General `arith.select` is outside
the export subset, although the canonical guarded sequence for an OpenQASM
dynamic shift is recognized and collapsed. Multi-operation modifier bodies must
have a target qubit and cannot capture additional qubits from an enclosing
scope.

The exporter recognizes the canonical nonzero guard around unsigned division and
modulo and reconstructs the corresponding source operator. It does not
reconstruct other runtime checks created for dynamic indices or checked integer
arithmetic. Other surviving assertions, checked-index control flow, or live
poison values cause an explicit diagnostic. Programs with static qubit and bit
indices and scalar casts from the documented subset can be exported and parsed
again through the strict frontend. Programs that rely on the remaining input
safety machinery must continue through another output path such as QIR.

:::{important}
The compiler removes dead code. A circuit that only prepares a state has no
observable effect and may be removed by optimization. Measure the relevant
qubits and return the results when compiling a program for execution.
:::
