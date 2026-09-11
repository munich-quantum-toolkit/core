---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Follow a program through the compiler

Why does the compiler use more than one representation of a quantum program? How
can you tell whether an optimization changed its meaning? This workshop answers
these questions by compiling small programs, inspecting the intermediate
representation (IR), and checking their results.

You need basic Python and quantum computing knowledge, but no MLIR experience.
Allow **60–90 minutes** for the three notebooks, including the experiments:

1. **This notebook:** read QC and QCO, apply an optimization, and check it
   (about 25 minutes).
2. **{doc}`getting_started_control_flow`:** follow registers, loops, and
   measurement results (20–25 minutes).
3. **{doc}`getting_started_targets`:** explain routing and native gates, then
   execute a compiled program (25–35 minutes).

For a first execution without inspecting IR, start with
{doc}`the QPE walkthrough <../getting_started>`. The
{doc}`compiler guide <mqt_compiler_collection>` documents the interfaces and
options used here; this workshop explains
*why the output looks the way it does*.

## Run the notebook

Use an installation with the **MQT Core v4 compiler interface** described in
{doc}`../installation`. Before v4 is released, use a current development build;
a stable v3 package does not provide this interface. The Python package includes
the compiler and local simulation used here. You do not need a separate MLIR
installation when using a compatible wheel.

The figures use Qiskit's visualization tools. Direct compiler translation
currently supports **Qiskit 2.5.x**; see {doc}`qiskit`. Install
`qiskit[visualization]~=2.5.0` and JupyterLab in the same Python environment if
they are not already available.

{download}`Download this notebook <../_build/jupyter_execute/mlir/GettingStarted.ipynb>`
and run its cells from top to bottom in JupyterLab. Each notebook has its own
setup. On this website, the cells and figures show results executed during the
documentation build; edit and rerun the downloaded notebook to experiment.

```{code-cell} ipython3
import numpy as np
from IPython.display import Code, display

from mqt.core.mlir import (
    OutputFormat,
    QCProgram,
    build_functionality,
    compile_program,
    simulate,
)
```

## Start with a program you can predict

A Hadamard followed by a controlled-X prepares the Bell state
$(|00\rangle + |11\rangle)/\sqrt{2}$ from $|00\rangle$. Our program deliberately
has **three** Hadamards. Since $H^2 = I$, it should prepare the same state.

We begin with two individual qubits. Registers and measurements will come in the
next notebook, so they do not obscure the first transformation.

```{code-cell} ipython3
source = """OPENQASM 3.0;
include "stdgates.inc";
qubit a;
qubit b;
h a;
h a;
h a;
cx a, b;
"""

qc = QCProgram.from_openqasm_str(source)
qc.to_qiskit().draw("mpl")
```

The circuit is read from left to right. The controlled-X acts on `b`, controlled
by `a`. Predict which gates the compiler can remove before running a pass.

## Read the imported QC

The compiler first **translates** OpenQASM into MLIR. MLIR is a framework for
representing and transforming programs. A *dialect* groups related operations
and types: `qc.h` is the Hadamard operation in MQT's QC dialect, while
`func.func` defines a function in MLIR's `func` dialect.

```{code-cell} ipython3
display(Code(qc.ir, language="mlir"))
```

Read the output in this order:

- `module` contains the program. Its `func.func @main()` has the
  `mqt.entry_point` attribute identifying the entry point.
- `qc.alloc` allocates a qubit initialized to $|0\rangle$. Each `%...` name
  identifies an **SSA value**: it is defined once, rather than reassigned.
- `!qc.qubit` is the type of a qubit reference. All three `qc.h` operations use
  the same reference. The reference stays the same while the quantum state
  changes. This is **reference semantics**.
- `qc.ctrl` applies the operations in its body under quantum control. The body's
  `%arg...` names are its arguments; here the body applies X to the target
  qubit.
- Braces delimit **regions**, which contain blocks of operations. A block ends
  with a **terminator**: `qc.yield` ends the controlled body, and `return` ends
  the function. `qc.dealloc` releases the allocated qubits.

The printed SSA names are chosen for readability. They are not stable IDs and
can change after a transformation.

## Make quantum data flow explicit with QCO

QC is convenient for import and export: a gate can refer to an existing qubit.
For optimization, it helps to express which operation produces the quantum value
used by the next operation. Convert QC to **QCO** to see this change:

```{code-cell} ipython3
qco = qc.to_qco(copy=True)
display(Code(qco.ir, language="mlir"))
```

Each `qco.h` now consumes one qubit value and produces a new one. Its result
feeds the next Hadamard. QCO uses **value semantics** to make that dependency
explicit. A value represents a qubit at that point in the computation; the IR
does not store a simulated statevector.

```{figure} ../_static/mlir/quantum-data-flow.svg
:alt: QC reuses one qubit reference; QCO connects three Hadamards through successive qubit values. Removing two Hadamards reconnects the remaining value flow.
:width: 100%

The same gates expressed through references and through quantum value flow. The
names in this diagram are explanatory, not a snapshot of the printed IR.
```

QCO enforces **linear semantics**: each quantum SSA value has exactly one use. A
gate transfers the value onward; it does not leave an old value available for a
second operation. `qco.ctrl` transfers both the control and target values, even
though a controlled-X does not flip its control. The final `qco.sink` operations
consume the values at the end of their lifetimes.

This rule lets a rewrite follow the quantum data flow directly. It is not a
claim that neighboring lines always act on the same qubit: follow their operands
and results.

## Run one optimization and check its meaning

A **pass** inspects or transforms IR. The `canonicalize` pass applies registered
rules that simplify operations, including cancellation of adjacent Hadamards on
the same qubit. It can apply several such rules; it is not a Hadamard-only pass.

Predict the gate count, then run it on a copy:

```{code-cell} ipython3
optimized = qco.copy()
optimized.run_pass_pipeline("canonicalize")
final_qc = optimized.to_qc(copy=True)

before = qc.num_gates()
after = final_qc.num_gates()
assert (before, after) == (4, 2)
print(f"Gate count: {before} → {after}")
final_qc.to_qiskit().draw("mpl")
```

The two cancelled Hadamards were redundant. But fewer gates alone do not prove
correctness. For this small, unitary program, compare the **full matrices**:

```{code-cell} ipython3
original_matrix = build_functionality(qco)
optimized_matrix = build_functionality(optimized)
np.testing.assert_allclose(optimized_matrix, original_matrix, rtol=0, atol=1e-12)
print(
    f"Largest matrix-entry difference: {np.max(np.abs(optimized_matrix - original_matrix)):.2e}"
)
```

This checks the transformation on every input state, including phase, within
floating-point tolerance. Merely obtaining the same measurement counts from
$|00\rangle$ would be a weaker check. Full matrices grow exponentially; use this
experiment for these two qubits, not as a general large-program validator.

Now inspect the output state from $|00\rangle$:

```{code-cell} ipython3
state = simulate(optimized)
expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
np.testing.assert_allclose(state, expected, rtol=0, atol=1e-12)
for index, amplitude in enumerate(state):
    print(f"|{index:02b}>: {amplitude.real:+.3f}{amplitude.imag:+.3f}j")
```

The vector uses basis order $|b\,a\rangle$, with the higher-index qubit on the
left. Only $|00\rangle$ and $|11\rangle$ have nonzero amplitudes, each
$1/\sqrt{2}$. Their measurement probabilities are $1/2$.

## Locate this pass in the full compiler

We selected one pass to explain its effect. Ordinary `compile_program` calls
coordinate frontend preparation, conversion, optimization, and output lowering.

```{figure} ../_static/mlir/compiler-workshop-pipeline.svg
:alt: OpenQASM is imported into QC, converted to QCO, optimized, and converted back to QC for export. Target compilation adds placement, routing, and native synthesis before emission. Submission is a separate step.
:width: 100%

Inspection checkpoints in the compiler. A target's capabilities and the chosen
output format also determine the required compilation steps.
```

The following calls expose four checkpoints. The first two do not run the
configured QCO optimization pipeline, though import and conversion can perform
their own preparation and cleanup.

```{code-cell} ipython3
imported = compile_program(source, output=OutputFormat.QC_IMPORT)
converted = compile_program(source, output=OutputFormat.QCO)
automatic = compile_program(source, output=OutputFormat.QCO_OPTIMIZED)
compiled = compile_program(source, output=OutputFormat.QC)
np.testing.assert_allclose(
    build_functionality(automatic), original_matrix, rtol=0, atol=1e-12
)
print(f"Imported QC gates: {imported.num_gates()}")
print(f"Final QC gates: {compiled.num_gates()}")
```

Inspect the optimized QCO and final QC below. The conversion back to reference
semantics preserves the optimized program.

```{code-cell} ipython3
:tags: [hide-output]
display(Code(automatic.ir, language="mlir"))
display(Code(compiled.ir, language="mlir"))
```

Use `qco_pipeline="canonicalize"` to replace the default QCO optimization
pipeline in a target-independent compilation. It does not remove the compiler's
required preparation and output stages. See the
[compiler guide](mqt_compiler_collection.md#run-passes-explicitly) for composing
passes; the workshop's target compilation uses the coordinated target pipeline.

## Keep Python ownership separate from quantum semantics

`copy=True` above preserves the source program so that we can compare stages.
Without it, conversions between MLIR-backed program objects
**consume the source module**. This avoids an implicit copy of a potentially
large program.

```{code-cell} ipython3
temporary = qc.copy()
converted_temporary = temporary.to_qco()
assert not temporary.is_valid
assert converted_temporary.is_valid and qc.is_valid
print(f"Source still owns a module: {temporary.is_valid}")
print(f"Result owns a module: {converted_temporary.is_valid}")
```

`is_valid` reports whether the Python object still owns its module. It is not an
IR-verification method or a check of algorithmic correctness. Python module
ownership and QCO's linear quantum values are different concepts.

High-level `compile_program` preserves typed inputs by default; `inplace=True`
allows it to consume them. Recreate inputs or use copies when comparing results
interactively.

## Experiment: which Hadamards cancel?

First predict the effect of zero through four consecutive Hadamards. The cell
checks each variant and prints its remaining gate count:

```{code-cell} ipython3
for hadamards in range(5):
    variant = source.replace("h a;\nh a;\nh a;", "h a;\n" * hadamards)
    original = QCProgram.from_openqasm_str(variant).to_qco()
    simplified = original.copy()
    simplified.run_pass_pipeline("canonicalize")
    np.testing.assert_allclose(
        build_functionality(simplified),
        build_functionality(original),
        rtol=0,
        atol=1e-12,
    )
    gates = simplified.to_qc(copy=True).num_gates()
    assert gates == 1 + hadamards % 2
    print(f"{hadamards} Hadamards → {gates} total gates")
```

:::{dropdown} Explain the result
An even number of Hadamards acts as the identity; an odd number acts as one
Hadamard. The controlled-X remains in both cases. Although it leaves
$|00\rangle$ unchanged, it acts on other possible inputs, so removing it would
change the program's full unitary.
:::

Now replace the three Hadamards with **H, Z, H**. Can you still cancel the two
Hadamards across Z? Predict the final state before running:

```{code-cell} ipython3
intervening = source.replace("h a;\nh a;\nh a;", "h a;\nz a;\nh a;")
intervening_qco = QCProgram.from_openqasm_str(intervening).to_qco()
simplified = intervening_qco.copy()
simplified.run_pass_pipeline("canonicalize")
np.testing.assert_allclose(
    build_functionality(simplified),
    build_functionality(intervening_qco),
    rtol=0,
    atol=1e-12,
)
np.testing.assert_allclose(simulate(simplified), [0, 0, 0, 1], rtol=0, atol=1e-12)
print(simulate(simplified))
```

:::{dropdown} Explain the result
$HZH = X$, not $Z$. Starting from $|00\rangle$, X flips `a`, and the
controlled-X then flips `b`, producing $|11\rangle$. Removing the Hadamards
across Z would be incorrect. A different rewrite may use the complete identity
$HZH = X$; this is not cancellation of adjacent inverse gates.
:::

Continue with {doc}`getting_started_control_flow` to see how quantum values move
through registers and control flow. For operation definitions and implementation
guidance, use the {doc}`QC`, {doc}`QCO`, and {doc}`development` references.

```{toctree}
:hidden:

getting_started_control_flow
getting_started_targets
```
