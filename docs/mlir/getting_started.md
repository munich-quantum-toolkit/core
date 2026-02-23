# Getting Started with MLIR in MQT Core

MQT Core compiles quantum circuits through an MLIR-based pipeline: programs are lowered to QC IR, then to QCO for optimization, and finally to QIR or other backends. This guide gets you from zero to writing and reading that IR yourself—so you can debug, extend, or plug into the compiler instead of treating it as a black box.

No MLIR experience is required. By the end you will have built a Bell-state circuit in QC, seen the same program in QCO, and used loops and conditionals with qubits while respecting linear types.

**What you'll do:**

- Set up a build that includes the MLIR compiler and QC/QCO dialects
- Write a minimal "Hello Quantum" program and a Bell-state circuit in QC
- Understand QC (reference semantics) vs QCO (value semantics) with a running example
- Use `scf.for` and `scf.if` with qubits and uphold SSA and linear types in QCO

## The Setup

You need a build of MQT Core that includes the MLIR compiler and the QC/QCO dialects.

- **Install MQT Core:** Follow {doc}`installation <../installation>`. For Python-only use, `uv pip install mqt.core` or `pip install mqt.core` is enough. To work with MLIR (run the pipeline, run tests, or inspect IR), build from source; see "Building from Source for Performance" and "Development Setup" in that page.

- **Build with MLIR:** Clone the repo, configure with CMake, and build. The default configuration builds the MLIR pipeline and the compiler unit tests (e.g. the executable in `mlir/unittests/Compiler/`). Run `cmake --build build` to get binaries and tests.

- **Big picture:** The overview {doc}`MLIR in the MQT <index>` explains how QC, QCO, and conversions (e.g. to QIR) fit together. This tutorial is the hands-on companion: we write IR and reason about it; the reference docs define every op in detail.

## "Hello Quantum" in QC: Your First IR Module

In QC, a qubit is a **reference** (type `!qc.qubit`): gates like `qc.h` or `qc.x` modify it in place. That matches the usual "circuit with wires and gates" mental model. QC is the dialect you get when importing from Qiskit, OpenQASM, or the MQT Core `QuantumComputation` type; it is also the dialect the pipeline turns into QCO for optimization.

We'll embed QC inside a normal MLIR module (using `func` and, later, `arith`). Everything you see below is valid MLIR that the compiler understands.

### Single-qubit "Hello Quantum"

The smallest useful program: allocate one qubit, put it in superposition with Hadamard, measure it, then deallocate.

```text
module {
  func.func @hello_single_qubit() {
    %q = qc.alloc : !qc.qubit

    qc.h %q : !qc.qubit

    %m = qc.measure %q : !qc.qubit -> i1

    qc.dealloc %q : !qc.qubit

    return
  }
}
```

**Reading the IR:** `qc.alloc` creates a qubit; `qc.h` and `qc.x` take a qubit and modify it in place (no new SSA value). `qc.measure` returns a classical `i1`; `qc.dealloc` releases the qubit. The same `%q` is used for every gate—that's reference semantics.

:::{tip}
In QC, one name can refer to the same qubit across many operations. The compiler tracks the *state* of that qubit as it flows through the circuit.
:::

### From one qubit to a Bell state

Next we prepare a **Bell pair** \(\frac{|00\rangle + |11\rangle}{\sqrt{2}}\): Hadamard on the first qubit, then a **controlled-X (CNOT)** with the first qubit as control and the second as target. In QC, controlled gates are written with `qc.ctrl(control) { body }`: the body is the operation that runs only when the control is \(\lvert 1 \rangle\).

```text
module {
  func.func @bell_pair() {
    %q0 = qc.alloc : !qc.qubit
    %q1 = qc.alloc : !qc.qubit

    qc.h %q0 : !qc.qubit

    qc.ctrl(%q0) {
      qc.x %q1 : !qc.qubit
    } : !qc.qubit

    %c0 = qc.measure("c", 2, 0) %q0 : !qc.qubit -> i1
    %c1 = qc.measure("c", 2, 1) %q1 : !qc.qubit -> i1

    qc.dealloc %q0 : !qc.qubit
    qc.dealloc %q1 : !qc.qubit

    return
  }
}
```

- Inside `qc.ctrl(%q0) { ... }`, the body sees the target qubit(s); `%q0` is the control and is not modified.
- `qc.measure("c", 2, 0)` and `("c", 2, 1)` record results into a 2-bit classical register `"c"`. The returned `i1` values are perfectly correlated (00 or 11) for an ideal Bell state.

## QC vs QCO: Reference vs Value Semantics

The same quantum operations can be represented in two ways in MLIR:

| | QC | QCO |
|---:|:---|:---|
| **Semantics** | Reference: gates modify qubits in place | Value: each op consumes inputs, produces new qubit values |
| **Type** | `!qc.qubit` | `!qco.qubit` |
| **Use case** | Import/export, human-readable circuits | Optimizations, DAG-style analysis |
| **SSA** | One name can be reused along the circuit | Every gate introduces new SSA values |

QC is described in {doc}`QC <QC>`, QCO in {doc}`QCO <QCO>`. The compiler converts QC → QCO so that optimization passes see explicit dataflow; then it can convert back or lower to QIR. Conceptually the pipeline looks like this:

```text
  Your circuit / QuantumComputation
           |
           v
  QC (reference semantics)  ←  import from Qiskit, OpenQASM, etc.
           |
           v
  QCO (value semantics)     ←  optimizations run here
           |
           v
  QIR or other backend      ←  target-specific code
```

### Same Bell state in QC (reference style)

We already wrote this above; here it is again so you can compare. Notice `%q0` and `%q1` are used repeatedly—the *same* references are updated by each gate.

```text
func.func @bell_qc() {
  %q0 = qc.alloc : !qc.qubit
  %q1 = qc.alloc : !qc.qubit

  qc.h %q0 : !qc.qubit

  qc.ctrl(%q0) {
    qc.x %q1 : !qc.qubit
  } : !qc.qubit

  %c0 = qc.measure("c", 2, 0) %q0 : !qc.qubit -> i1
  %c1 = qc.measure("c", 2, 1) %q1 : !qc.qubit -> i1

  qc.dealloc %q0 : !qc.qubit
  qc.dealloc %q1 : !qc.qubit
  return
}
```

### Same Bell state in QCO (value style)

In QCO, every gate **consumes** its qubit operand(s) and **produces** new qubit value(s). You never reuse a qubit value after it's been "used" by an operation—that's **linear typing**. The circuit is the same; only the IR representation changes.

```text
module {
  func.func @bell_qco() {
    %q0_0 = qco.alloc : !qco.qubit
    %q1_0 = qco.alloc : !qco.qubit

    %q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit

    %q0_2, %q1_1 = qco.ctrl(%q0_1) targets(%t0 = %q1_0) {
      %t0_1 = qco.x %t0 : !qco.qubit -> !qco.qubit
      qco.yield %t0_1
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    %q0_3, %c0 = qco.measure("c", 2, 0) %q0_2 : !qco.qubit
    %q1_2, %c1 = qco.measure("c", 2, 1) %q1_1 : !qco.qubit

    qco.dealloc %q0_3 : !qco.qubit
    qco.dealloc %q1_2 : !qco.qubit

    return
  }
}
```

**What to notice:**

- Qubit type is `!qco.qubit`; each op has explicit output types, e.g. `qco.h %q_in : !qco.qubit -> !qco.qubit`.
- `qco.ctrl` consumes control and target qubits and returns new values for both; the body must end with `qco.yield` passing the transformed target(s).
- `qco.measure` returns both the (collapsed) qubit and the classical bit.

Value semantics make dataflow explicit and allow optimization passes to reason about dependencies.

### How the compiler turns QC into QCO

The pipeline has a **QC → QCO** conversion pass. It keeps the logical circuit but rewrites it so every mutation becomes an explicit "consume old value, produce new value." For example, this QC snippet:

```text
%q = qc.alloc : !qc.qubit
qc.h %q : !qc.qubit
qc.x %q : !qc.qubit
```

becomes this in QCO (same physics, different IR):

```text
%q0 = qco.alloc : !qco.qubit
%q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
%q2 = qco.x %q1 : !qco.qubit -> !qco.qubit
```

Optimization passes then work on this SSA form without worrying about aliasing.

## Advanced Constructs: Functions, Control Flow, and Linear Types

Beyond straight-line circuits you can use multiple functions, loops (`scf.for`), and conditionals (`scf.if` or `qco.if`). With QCO qubits you must respect **SSA and linear types**: each `!qco.qubit` is defined once and consumed exactly once. When control flow splits or merges, qubits must be passed in as block arguments and **yielded** out so that no qubit is duplicated or dropped.

:::{important}
In QCO, every path through a region that touches a qubit must **yield** that qubit (or a value derived from it). Use `scf.yield` in `scf.for` / `scf.if` and `qco.yield` in `qco.ctrl` / `qco.inv` / `qco.if`. Entry block arguments carry qubits into the region; the yield carries them out.
:::

### Multiple functions with QC/QCO

You can split a program into functions that allocate and return qubits. The caller receives the qubit values and is responsible for deallocating them (or passing them on). Below, `@prepare_bell` builds a Bell pair and returns both qubits and the two classical measurement bits; `@use_bell` calls it and then deallocates the qubits.

```text
module {
  func.func @prepare_bell() -> (!qco.qubit, !qco.qubit, i1, i1) {
    %q0_0 = qco.alloc : !qco.qubit
    %q1_0 = qco.alloc : !qco.qubit

    %q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit

    %q0_2, %q1_1 = qco.ctrl(%q0_1) targets(%t0 = %q1_0) {
      %t0_1 = qco.x %t0 : !qco.qubit -> !qco.qubit
      qco.yield %t0_1
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    %q0_3, %c0 = qco.measure("c", 2, 0) %q0_2 : !qco.qubit
    %q1_2, %c1 = qco.measure("c", 2, 1) %q1_1 : !qco.qubit

    return %q0_3, %q1_2, %c0, %c1 : !qco.qubit, !qco.qubit, i1, i1
  }

  func.func @use_bell() {
    %q0, %q1, %c0, %c1 = call @prepare_bell() : () -> (!qco.qubit, !qco.qubit, i1, i1)

    qco.dealloc %q0 : !qco.qubit
    qco.dealloc %q1 : !qco.qubit

    return
  }
}
```

Returning qubits from a function makes ownership explicit: the caller owns the returned values and must deallocate or pass them on.

### Loops with scf.for and QCO

To loop over a qubit (e.g. apply a gate repeatedly), pass the qubit into the loop as an **iteration argument** (`iter_args`) and **yield** the updated qubit from the body. The loop result is the qubit value after the last iteration. Never use a qubit from outside the loop inside the body without passing it through `iter_args`—that would violate SSA and linearity.

```text
module {
  func.func @loop_h(%n : index) -> !qco.qubit {
    %q0 = qco.alloc : !qco.qubit
    %zero = arith.constant 0 : index
    %one  = arith.constant 1 : index

    %q_final = scf.for %i = %zero to %n step %one
                iter_args(%q_iter = %q0 : !qco.qubit)
                -> !qco.qubit {
      %q_next = qco.h %q_iter : !qco.qubit -> !qco.qubit
      scf.yield %q_next : !qco.qubit
    }

    return %q_final : !qco.qubit
  }
}
```

Here `%q_iter` is the qubit "at the start of this iteration"; we apply `qco.h` to get `%q_next` and `scf.yield %q_next` so the next iteration gets that value. Each iteration has a single chain of ownership.

### Branching with scf.if and QCO

With `scf.if`, **both** the then and else regions must yield a value of the same type. So if you pass a qubit into an `scf.if`, one branch might apply a gate and yield the new qubit, the other might yield the qubit unchanged—but both must yield exactly one value. Example: apply X only when a classical condition is true:

```text
module {
  func.func @conditional_x(%q_in : !qco.qubit, %cond : i1) -> !qco.qubit {
    %q_out = scf.if %cond -> !qco.qubit {
      %q_then = qco.x %q_in : !qco.qubit -> !qco.qubit
      scf.yield %q_then : !qco.qubit
    } else {
      scf.yield %q_in : !qco.qubit
    }

    return %q_out : !qco.qubit
  }
}
```

The then branch applies `qco.x` and yields the result; the else branch yields the input qubit unchanged. The `if` op's result is the single live qubit after the branch. For conditionals that are specifically about qubits, you can also use **`qco.if`**, which takes a condition and qubit operands and passes the qubits into both regions as block arguments; each branch ends with `qco.yield` of the (possibly transformed) qubits. See the {doc}`QCO dialect <QCO>` for the exact syntax.

### Combining loops and branches

You can nest `scf.if` inside `scf.for` (or the other way around) as long as every path yields the right values. Below, the loop carries both a qubit and a classical condition; in each iteration we optionally apply X to the qubit depending on the condition, then flip the condition and yield both for the next iteration.

```text
module {
  func.func @adaptive_loop(%steps : index, %initial_cond : i1)
              -> (!qco.qubit, i1) {
    %q0 = qco.alloc : !qco.qubit
    %zero = arith.constant 0 : index
    %one  = arith.constant 1 : index

    %q_final, %cond_final = scf.for %i = %zero to %steps step %one
                              iter_args(%q_iter = %q0 : !qco.qubit,
                                        %cond_iter = %initial_cond : i1)
                              -> (!qco.qubit, i1) {
      %q_new = scf.if %cond_iter -> !qco.qubit {
        %q_then = qco.x %q_iter : !qco.qubit -> !qco.qubit
        scf.yield %q_then : !qco.qubit
      } else {
        scf.yield %q_iter : !qco.qubit
      }

      %cond_new = arith.xori %cond_iter, arith.constant 1 : i1

      scf.yield %q_new, %cond_new : !qco.qubit, i1
    }

    return %q_final, %cond_final : !qco.qubit, i1
  }
}
```

The rule is the same everywhere: all live qubit (and classical) state that crosses an iteration or branch boundary goes through `iter_args` or `scf.if` operands, and is returned via `scf.yield`.

## From QC to QCO and Beyond

The full pipeline does the following:

1. **Input:** A high-level circuit (e.g. MQT Core `QuantumComputation`) or a QC module. Translation from circuits to QC is done by `translateQuantumComputationToQC` or similar.
2. **QC:** Optional canonicalization and cleanup on QC.
3. **QC → QCO:** Conversion to value semantics. See {doc}`Conversions <Conversions>` for the conversion passes.
4. **QCO:** Optimization passes run on QCO (and possibly further lowering).
5. **Output:** QCO can be converted back to QC or lowered to QIR or another backend.

The flow diagram at the start of the [QC vs QCO](#qc-vs-qco-reference-vs-value-semantics) section summarizes this.

**Next steps:**

- Run the pipeline yourself: the test `mlir/unittests/Compiler/test_compiler_pipeline.cpp` runs the compiler on many small programs and compares results to reference QC and QIR. Use it as a template to feed your own QC modules through the pipeline and inspect the IR after each stage.
- Read the dialect references: {doc}`QC <QC>` and {doc}`QCO <QCO>` list every operation and type; {doc}`Conversions <Conversions>` documents the conversion passes.
- Try modifying the examples in this guide (e.g. add a gate, add a loop iteration) and run them through the compiler to see how QC and QCO change.
