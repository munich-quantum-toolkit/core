---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Follow registers and control flow

A quantum program can keep a register, repeat a block, and use a measurement to
choose what happens next. How does the compiler represent these features while
preserving QCO's linear quantum values?

This is the second part of the {doc}`compiler workshop <GettingStarted>`. Allow
**20–25 minutes**. The notebook runs independently; use the same v4-capable MQT
Core installation, Qiskit 2.5.x visualization tools, and Jupyter environment
described in the [workshop setup](GettingStarted.md#run-the-notebook).

{download}`Download this notebook <../_build/jupyter_execute/mlir/getting_started_control_flow.ipynb>`.

```{code-cell} ipython3
from IPython.display import Code, display
from qiskit.visualization import plot_distribution

from mqt.core.mlir import OutputFormat, QCProgram, compile_program, sample
```

## Grow a Bell state into a GHZ state

For $n$ qubits, a GHZ state is $(|0\cdots0\rangle + |1\cdots1\rangle)/\sqrt{2}$.
Apply H to the first qubit, then a controlled-X from that qubit to each of the
others.

Use a **register and loop** instead of writing every controlled-X by hand.
Change `width` from 3 to 2 or 4 and rerun the following cells. Predict both the
loop's iteration count and the possible measurement outcomes.

```{code-cell} ipython3
width = 3
shots = 4096
seed = 17
assert 2 <= width <= 4

ghz_source = f"""OPENQASM 3.0;
include "stdgates.inc";
qubit[{width}] q;
bit[{width}] result;
h q[0];
for int i in [1:{width - 1}] {{
    cx q[0], q[i];
}}
result = measure q;
"""

ghz_qc = QCProgram.from_openqasm_str(ghz_source)
```

OpenQASM ranges include both endpoints: `[1:2]` visits 1 and 2. The loop
therefore applies `width - 1` controlled-X gates during execution. The last
statement measures each qubit into the corresponding output bit.

## Inspect the register in QC

Read the allocation, the loads inside the loop, and the measurement stores:

```{code-cell} ipython3
display(Code(ghz_qc.ir, language="mlir"))
```

`memref<3x!qc.qubit>` holds three qubit references in the default example.
`memref.load` retrieves one of those references; a quantum gate operates on the
referenced qubit. This is the register counterpart of scalar QC reference
semantics.

The output has type `!cbit.reg<3>`. **CBit** represents classical-bit registers
shared by QC and QCO. `qc.measure` produces an `i1` measurement result;
`cbit.store` writes it into the register that `main` returns. The imported
register's undefined initializer is safe here because the program writes every
bit before returning it.

`scf.for` is MLIR's structured counted loop. Its upper bound is **exclusive**,
so the compiler translates OpenQASM's inclusive range accordingly. An
`arith.constant` creates a classical constant; `index` is MLIR's type for
indexing and iteration in this loop.

## Follow the register through QCO

Predict what must change: if each quantum value has exactly one use, how can
successive loop iterations work on the same register?

```{code-cell} ipython3
ghz_qco = ghz_qc.to_qco(copy=True)
display(Code(ghz_qco.ir, language="mlir"))
```

Read the quantum register's path through the program:

1. `qtensor.alloc` creates a `tensor<3x!qco.qubit>`. **QTensor** represents a
   collection of linear quantum values.
2. `qtensor.extract` transfers a qubit out of the tensor. It returns both the
   qubit and the remaining tensor, whose extracted slot is unavailable until
   filled again.
3. A gate consumes the extracted qubit and produces its successor.
4. `qtensor.insert` puts that successor back and returns an updated tensor.
5. `scf.for` carries the tensor through `iter_args`. The body receives the
   current tensor as a block argument and returns its successor with
   `scf.yield`. The loop result holds the tensor after the final iteration.

Extraction does not copy a quantum state. Both the extracted qubit and the
remaining tensor follow linear semantics. A loop must carry its quantum state
through the iteration arguments rather than repeatedly capturing an earlier
value from outside the body.

`qco.measure` produces **two results**: the post-measurement qubit value and the
classical outcome. The qubit must still be inserted back into its tensor; the
classical result is stored in the CBit register. Classical values do not follow
the quantum exactly-one-use rule.

## Distinguish a loop body from executed gates

The inspection methods count gate operations in the entry-point IR. A loop body
is counted once, regardless of its trip count:

```{code-cell} ipython3
static_gates = ghz_qc.num_gates()
static_cx = ghz_qc.num_two_qubit_gates()
assert static_gates == 2 and static_cx == 1
print(f"Static IR: {static_gates} gates, including {static_cx} controlled-X")
print(
    f"This program executes: {1 + width - 1} gates, including {width - 1} controlled-X"
)
```

The runtime count here follows directly from a loop with constant bounds. It is
not a general estimate provided by `num_gates()`. Branches and other loops can
make runtime work depend on measurement outcomes or classical inputs.

Compare the same program after explicit loop unrolling:

```{code-cell} ipython3
unrolled = ghz_qco.copy()
unrolled.unroll_quantum_loops()
unrolled_qc = unrolled.to_qc()
assert unrolled_qc.num_two_qubit_gates() == width - 1
print(f"Controlled-X operations after unrolling: {unrolled_qc.num_two_qubit_gates()}")
unrolled_qc.to_qiskit().draw("mpl")
```

Unrolling duplicates the body for the known iterations. It changes the
representation, not the algorithm. Keeping loops can keep IR compact; some
output formats or target-address requirements need the iterations exposed. The
[target notebook](getting_started_targets.md) explains how output and hardware
constraints guide compilation.

## Check the observable result

The QCO sampler interprets supported structured programs directly. With 4096
shots, approximately half the results should be all zeros and half all ones:

```{code-cell} ipython3
ghz_counts = sample(ghz_qco, shots=shots, seed=seed)
zeros, ones = "0" * width, "1" * width
assert set(ghz_counts) == {zeros, ones}
assert sum(ghz_counts.values()) == shots
assert abs(ghz_counts[zeros] / shots - 0.5) < 0.05
plot_distribution(ghz_counts, title=f"{width}-qubit GHZ outputs")
```

The keys describe the returned classical register, with its highest-index bit on
the left. The sampler reports measured outputs; it does not display a
statevector. A fixed nonzero seed makes this experiment reproducible, but
statistical checks should not depend on a particular histogram.

:::{dropdown} Experiment: change the register width
Try `width = 2` and `width = 4` in the parameter cell and rerun the GHZ section.
The structured body still contains one controlled-X, but it executes once or
three times. The histogram still has two outcomes, now with two or four bits.
:::

These checks execute both variants during the documentation build:

```{code-cell} ipython3
for test_width in (2, 4):
    variant = ghz_source.replace(f"[{width}]", f"[{test_width}]").replace(
        f"[1:{width - 1}]", f"[1:{test_width - 1}]"
    )
    counts = sample(variant, shots=shots, seed=seed)
    assert set(counts) == {"0" * test_width, "1" * test_width}
    assert sum(counts.values()) == shots
    assert abs(counts["0" * test_width] / shots - 0.5) < 0.05
    print(f"Width {test_width}: {counts}")
```

## Use a measurement to choose the next gate

A **classical correction** prepares zero even when the first measurement is
random. Start with H, measure, and apply X only if the outcome is 1. What should
a second measurement return?

```{code-cell} ipython3
feedback_source = """OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit outcome;
h q;
outcome = measure q;
if (outcome) {
    x q;
}
outcome = measure q;
"""

feedback_qc = QCProgram.from_openqasm_str(feedback_source)
feedback_qc.to_qiskit().draw("mpl")
```

The output register is overwritten by the second measurement. The first result
controls the correction but is not a second returned output bit.

Inspect the branch in QCO:

```{code-cell} ipython3
feedback_qco = feedback_qc.to_qco(copy=True)
display(Code(feedback_qco.ir, language="mlir"))
```

`qco.if` receives a classical condition and transfers the post-measurement qubit
into the selected branch. The true branch applies X; the false branch forwards
the qubit unchanged. Each branch yields the qubit value that execution should
use next. The second measurement consumes the branch result.

| Operation  | Condition             | What happens                                                           |
| ---------- | --------------------- | ---------------------------------------------------------------------- |
| `qco.ctrl` | Quantum control qubit | Applies a coherent controlled operation, without measuring the control |
| `qco.if`   | Classical value       | Executes one branch and carries its quantum values to the result       |

Generic `scf.if` can represent classical branching, but quantum branches use
`qco.if` to express these ownership transfers. SCF loops carry quantum values
through their iteration arguments, as in the GHZ example.

```{code-cell} ipython3
corrected_counts = sample(feedback_qco, shots=shots, seed=seed)
assert corrected_counts == {"0": shots}
plot_distribution(corrected_counts, title="Final measurement after correction")
```

Every shot returns zero. Use sampling for this experiment: the convenience
`simulate()` function that returns a dense statevector does not support
mid-circuit feedback. See the {doc}`DD guide <../dd_package>` for lower-level
simulation interfaces.

:::{dropdown} Experiment: remove the correction
Delete the `if` block, or run the variant below. The second measurement repeats
the first outcome because nothing changes the measured qubit in between. Its
outcome is random across shots, even though the two measurements within a shot
agree.
:::

```{code-cell} ipython3
uncorrected_source = feedback_source.replace("if (outcome) {\n    x q;\n}\n", "")
uncorrected_counts = sample(uncorrected_source, shots=shots, seed=seed)
assert set(uncorrected_counts) == {"0", "1"}
assert sum(uncorrected_counts.values()) == shots
assert abs(uncorrected_counts["0"] / shots - 0.5) < 0.05
plot_distribution(
    [corrected_counts, uncorrected_counts], legend=["Correction", "No correction"]
)
```

Continue with {doc}`getting_started_targets` to compile against explicit
hardware constraints. For larger structured examples, explore
[iterative QPE](../getting_started.md#standard-versus-iterative-qpe) and
[repeat until success](../benchmarks.md#repeat-until-success). The
{doc}`QTensor`, {doc}`CBit`, and {doc}`OpenQASM` references describe the
operations and supported input forms in detail.
