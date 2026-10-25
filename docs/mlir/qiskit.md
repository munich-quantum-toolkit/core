# Qiskit compatibility

MQT Core exposes two distinct Qiskit interfaces:

| Interface                                                | Supported Qiskit versions | Purpose                                                          |
| -------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------- |
| {doc}`QDMI backend <../qdmi/qdmi_backend>`               | 2.1 and newer             | Execute circuits through device-specific serializers.            |
| {doc}`MQT Compiler Collection <mqt_compiler_collection>` | `>=2.5.0,<2.7.0`          | Import and export circuits through the versioned native adapter. |

Install `mqt-core[qiskit]`. Direct compiler translation requires a version in
the narrower range above; the adapter checks it before inspecting a circuit.

## QPY files

Use {py:mod}`qiskit.qpy` with the compiler's Qiskit conversion methods to
exchange {term}`QPY` files:

```python
from qiskit import qpy
from mqt.core.mlir import QCProgram

with open("input.qpy", "rb") as stream:
    (circuit,) = qpy.load(stream)

program = QCProgram.from_qiskit(circuit)
circuit = program.to_qiskit()

with open("output.qpy", "wb") as stream:
    qpy.dump(circuit, stream, version=17)
```

The tuple unpacking requires exactly one circuit. QC and QCO programs both
provide `to_qiskit()`; pass `copy=True` to preserve the source or `target=` to
map compiler site IDs through a target. The
[circuit translation contract](#circuit-translation-contract) defines which
circuit features survive conversion.

QPY version 17 works with Qiskit 2.5 and 2.6 and enables their native writer.
Load only trusted files. To preserve an existing destination on serialization
failure, serialize to `io.BytesIO` before opening the destination.

## Compiler targets

Pass a Qiskit `BackendV2` or `Target` to
{py:func}`~mqt.core.mlir.compile_program` to compile for its gates, topology,
and advertised standard control flow. The default result is a `QuantumCircuit`
with physical qubits; its layout records placement and routing of logical input
qubits. An explicit `output` selects another compiler representation. Omit
`target` for target-independent optimization, which preserves the input's
program family and may remove idle qubits. See the
[Qiskit example](mqt_compiler_collection.md#use-qiskit-circuits-directly).

For explicit compiler pipelines or repeated compilation,
{py:meth}`~mqt.core.mlir.CompilerTarget.from_qiskit` creates an independent
snapshot of gates and topology:

```python
from qiskit.providers.fake_provider import GenericBackendV2
from mqt.core.mlir import CompilerTarget

backend = GenericBackendV2(3, basis_gates=["sx", "x", "rz", "cx"])
target = CompilerTarget.from_qiskit(backend)
```

A backend supplies the default target name; pass `name` to override it. Qiskit
standard gates, measurement, reset, and their ordered qubit placements become
native capabilities. Each parameter slot can be unrestricted or fixed to one
finite value. Multiple instructions of the same gate, such as `RXGate(pi / 2)`
and `RXGate(pi)`, retain their distinct target names and placements. The
compiler uses canonical gate names; `to_qiskit(target=target)` selects the
target name by gate, parameters, and ordered sites. This also preserves legacy
spellings such as `u1` and `u3`.

Routing connectivity is undirected and comes from the retained two-qubit gates.
Gate applicability keeps its original qubit order. The target needs a known,
positive qubit count and a connected routing graph. Global phase is always
allowed as circuit metadata.

By default, unrepresentable gates are omitted with a warning. This includes
custom gates, unsupported angle bounds, renamed measurement or reset, open
controls, and controlled gates other than CX, CY, and CZ. Bounded RXX, RYY, RZX,
and RZZ gates are retained with the conservative interval `[0, pi/2]` when the
target accepts that entire interval; see
{doc}`target compilation <target_compilation>`. Delay, barrier, classical
control flow, and operations with no applicable qubits are omitted without a
warning. Pass `operation_names` to retain a subset; every selected name must be
representable. Calibration, timing, and scheduling data are not copied. A
`CompilerTarget` also omits backend control-flow support. For Qiskit output,
this snapshot permits all control-flow forms supported by the exporter.

Target-native synthesis requires a common single-qubit basis across sites and a
common entangler across routing edges. For explicit pipelines, export with
`program.to_qiskit(target=target)` to map compiler site IDs to physical qubits.

## Circuit translation contract

The compiler imports circuit operations without changing the source circuit.
Export creates a new circuit and preserves parameter identities across nested
control flow. Both `QCProgram.to_qiskit()` and `QCOProgram.to_qiskit()` consume
the source, including when export fails. Pass `copy=True` to preserve it.

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
| Constant Boolean, `Uint`, and `Float` expressions                       | Supported            | Supported                          |
| Clbit and ClassicalRegister expression variables                        | Supported            | Supported                          |
| Fixed-width bitwise operations, comparisons, and bounded shifts         | Supported            | Supported                          |
| Clbit, indexed-register, and whole-register `Store` assignments         | Supported            | Supported                          |
| Initialized local classical variables and enclosing captures            | Supported            | Native `expr.Var` and `Store`      |
| External runtime input variables                                        | Rejected             | Rejected                           |
| `break` in `for` and `while`, including nested conditionals/switches    | Supported            | Native `BreakLoopOp`               |
| `continue` in `for` and `while`, including nested conditionals/switches | Supported            | Preserved through SCF control flow |
| Free symbols and supported real parameter expressions                   | Supported            | Supported                          |
| Parameter-vector elements                                               | Supported            | Supported                          |
| Dense numeric unitaries up to eight qubits                              | Supported            | Supported                          |
| Register aliases or interleaved membership                              | Rejected             | Rejected                           |
| Complete transpiler layout metadata                                     | Preserved            | Reconstructed when still valid     |

Classical-expression variables may refer to Clbits or ClassicalRegisters in the
containing circuit. This includes values used only by the condition or switch
target and not by a control-flow block. External runtime inputs remain
unsupported.

Private gate parameters bind by position and receive generated local names
during Qiskit export; their original names and grouping are not preserved.
Public program inputs still require explicit names. OpenQASM custom gates can
therefore use `QCProgram.from_openqasm_str(source).to_qiskit()` directly within
the supported subset below.

Export folds scalar expressions in the consumed QC program. Constant arithmetic,
casts, and idempotent expressions can therefore disappear; expression-tree shape
is not preserved. Quantum-resource and classical-snapshot canonicalization
patterns are not applied, because they can change circuit width or introduce
scratch bits. Call `cleanup()` explicitly when those broader transformations are
wanted.

Imported free parameters and vectors retain their identities through QC/QCO
conversion, optimization, and MLIR serialization, so exported circuits can be
bound with the original Qiskit objects. OpenQASM 3 export preserves names for
free parameters that are valid OpenQASM identifiers, but does not preserve
Qiskit identities or vector grouping. QIR does not preserve those identities.
Unused entry parameters are omitted from Qiskit output. When binding a
dictionary that includes removed parameters, use
`assign_parameters(values, strict=False)`. Custom gate definitions must use
every formal parameter.

Free symbols become named {code}`f64` program inputs. Parameter-vector elements
retain their grouping and index, preserving vector order and positional binding
across a round trip; similarly named standalone parameters remain standalone.
Elements used in different structured-control blocks are restored into one
shared vector for the complete circuit tree. Free parameter vectors and their
combined declared size in one translated circuit are each limited to 65,536
elements. Parameter-expression trees support at most 64 levels and 32,768 nodes.
Import balances addition and subtraction chains before applying the depth limit.
Fusing runtime rotations can expand these expressions. For repeated numerical
runs, bind parameters in Core before export to avoid Qiskit's symbolic binding
cost and stack limits on long addition chains. The limits above bound conversion
work; they do not bound Qiskit's internal expression depth.

Import and export support real addition, subtraction, multiplication, division,
power, negation, trigonometric and inverse trigonometric functions, exponential,
logarithm, absolute value, and real conjugation. Export also folds signed and
unsigned integer-to-float casts of constants. Other parameter-expression
functions are rejected. Lexically bound {code}`for`-loop induction parameters
are supported and remain distinct from free symbols. Parameterized
custom-instruction definitions are expanded after their symbols and expressions
are resolved. Definition expansion rejects missing definitions, cycles, operand
arity mismatches, nesting beyond 64 levels, and more than 10 million expanded
operations. Permutation patterns lower directly to SWAPs in Core, including
inside nested definitions and gate modifiers. Other array-valued custom
parameters are represented by the circuit definition rather than scalar program
inputs. Operations without a supported definition are rejected with a Python
exception; arbitrary Python parameter objects are not preserved on export.

Controlled-gate import supports multiple targets and open controls. Open
controls use X conjugation from the gate's circuit definition;
`AnnotatedOperation` control modifiers must be closed.

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

Local scalar state covers Boolean values, fixed-width integers, and `f64` using
supported backend operations. It uses native `expr.Var` declarations, `Store`
assignments, and captures in nested circuits. Compiler variables do not increase
the classical-bit or register counts. Import accepts initialized local variables
and captures from enclosing supported circuits. External runtime inputs remain
unsupported. Runtime classical values cannot be used as symbolic gate
parameters.

Canonical short-circuit Boolean expressions retain their native expression form.
A live {code}`scf.for` induction value must reduce to an affine {code}`f64` gate
parameter. The exporter preserves one Qiskit parameter identity for that value
throughout its lexical body. An {code}`scf.index_switch` selector must be a
constant index or a supported Boolean/Uint expression converted with
{code}`arith.index_castui`. Switch selectors have at most 64 bits. Labels must
be nonnegative constants that fit the target width.

Nested blocks may capture existing qubits, classical bits, and local variables
but may not allocate or release qubits. Scoped CBit allocations become circuit
registers; zero-initialized storage is reset on each block entry. Control flow
and classical expressions may nest up to 64 levels, and classical expression
trees may contain at most 16,384 nodes. Integer expressions retain their exact
width, including values wider than 64 bits. Expression conditions and tuple
conditions such as `if_test((register, value))` support wide registers. Tuple
equalities with a value outside the register range become false. Standard
`arith.cmpi` handles every comparison: signed ordering is encoded by XOR-biasing
both operands' sign bits, including computed operands. Casts preserve truncation
and sign/zero extension. Bitwise operations, modular arithmetic, integer
selection, and shifts share these typed rules. Import guards runtime shifts so
overshifts produce zero; export preserves the guards. Rotations and population
count are expanded through the same bounded integer lowering used by jeff.
Unsupported operations, invalid widths, non-finite constants, unsupported index
uses, and dynamic for-loop bounds fail during validation. Programs without
classical outputs use a void entry function on import and export. Boolean entry
results become one-bit output registers in function-result order. Numeric scalar
results, including a constant-zero `i64`, cannot be represented as Qiskit output
bits. Each returned CBit register must be distinct. Whole-register reads map to
Qiskit `ClassicalRegister` expressions, and writes map to atomic Qiskit `Store`
operations. Indexed reads and writes accept constant indices or unsigned integer
expressions converted with {code}`arith.index_castui`; runtime indices must be
in bounds. Internal CBit storage becomes additional Qiskit registers, ordered
before returned registers; Qiskit exposes all circuit storage. Other formats and
execution backends may support a narrower set of integer widths.

Every public CBit output is exported as a Qiskit `ClassicalRegister`; an unnamed
allocation receives a collision-free `_mqt_cN` name. This preserves the CBit
register boundary and gives whole-register writes a valid Qiskit lvalue. Loose
input Clbits therefore round trip semantically, but not as loose output bits.

Qiskit circuit storage starts at zero. Reading an undefined CBit before its
first write is undefined behavior. The exporter preserves scalar snapshots,
including wide register reads, across later writes and control-flow boundaries.

Measurements keep their program order and write to static Qiskit Clbits. The
exporter reuses a CBit destination when its store can move next to the
measurement without crossing conflicting classical accesses. Otherwise, it
allocates a one-bit scratch register, exposed as additional circuit storage.
Other uses of the result read a snapshot, including classical expressions,
copies to dynamic indices, and stores in branches. Later measurements may reuse
a destination; saved snapshots retain the earlier result.

Dense numeric unitaries remain explicit matrix operations during import and
export. Target compilation synthesizes supported one- and two-qubit matrices to
the target gate set. Dense unitary operations support at most eight qubits.
Qiskit import preserves inverse, numeric power, and closed-control modifiers on
dense-unitary operations. Export preserves inverse and closed-control modifiers.
Other powers require canonicalization or synthesis.

### Transpiler layouts

A complete Qiskit `TranspileLayout` maps program qubits to device qubits and
records routing. Import stores those numeric maps as `mqt.layout` on the program
module; export constructs a Qiskit layout from them. Copies, MLIR serialization,
and QC/QCO conversions preserve the metadata. Other compiler transformations
clear it.

```python
program = QCProgram.from_qiskit(transpiled_circuit)
restored = program.to_qco().to_qc().to_qiskit()
```

Target compilation and compilation to a Qiskit circuit require inputs without an
attached layout. Use the conversion methods above to preserve an existing
layout. The numeric mapping does not retain Qiskit input register names or input
ancilla labels; incomplete layouts are unsupported. See {doc}`MQT dialect <MQT>`
for the metadata schema and {doc}`target compilation <target_compilation>` for
placement.

Names passed between Qiskit and the compiler must not contain NUL characters.
Loop lists with jumps and switch cases with variables are supported within the
64-level nesting and definition-expansion limits. Unsupported programs fail
without exposing a partial result.

The binding imports Qiskit only when circuit translation is requested. It
accepts versions in the registered {code}`>=2.5.0,<2.7.0` range and verifies the
native API version before reading a circuit.

Qiskit 2.6 uses borrowed C API instruction views and reads numeric gate
parameters and global phases natively. Both adapters construct circuits,
registers, and empty control-flow blocks through the C API. Tuple conditions
become C++ equality trees directly; their public Python values remain
authoritative because native conditions can be stale after a condition edit.
Symbolic expressions retain the Python fallback because the C API does not
expose their expression trees or parameter identities. Custom Python gates and
classical expressions also retain their Python fallbacks. The new native
custom-operation API defines operations; it does not inspect existing Python
gate definitions. Delay instructions remain unsupported because the compiler
does not represent their timing semantics.
