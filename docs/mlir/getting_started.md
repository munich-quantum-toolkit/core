# Getting Started with MLIR in MQT Core

## What Is MLIR?

Do you know how many levels of abstraction a quantum program passes through before it reaches hardware—high-level intent, gate sequences, hardware-specific pulses—and how each level needs its own operations and types?
Building a separate IR for each level, and keeping them all consistent, is painful and error-prone.
**MLIR** (Multi-Level Intermediate Representation) solves exactly this: it is a single extensible framework—developed as part of the LLVM project—in which you define your own *dialects* (modular vocabularies of operations and types) and compose them freely inside one module.

The core concept in MLIR is the **dialect**: a named collection of operations, types, and attributes that together define one level of abstraction in a compiler pipeline.
For example, a dialect might represent high-level quantum circuits, a lower-level gate-level IR, or even classical control flow.
Dialects can be mixed freely inside a single MLIR module, which makes it straightforward to gradually lower a program from a high-level description to machine code.

An **Intermediate Representation (IR)** is a data structure—usually human-readable text or a binary format—that a compiler uses to represent a program between the source language (e.g. Python or OpenQASM) and the final output (e.g. native quantum instructions or QIR).
Inspecting the IR at each stage is one of the most powerful tools for understanding and debugging a compiler pipeline.

## MLIR in MQT Core

MQT Core includes an evolving MLIR-based submodule for representing and compiling quantum programs.
This submodule is separate from the rest of MQT Core and is not yet used outside of the `mlir/` directory.

Two custom dialects are defined:

- **QC** (Quantum Circuit): uses *reference semantics*.
  A qubit is a `!qc.qubit` reference; gates like `qc.h` or `qc.x` modify the qubit in place without returning a new value.
  QC is the import/export dialect—it is the natural representation when translating from Qiskit, OpenQASM, or the MQT Core `QuantumComputation` type.

- **QCO** (Quantum Circuit Optimized): uses *value semantics*.
  Every gate consumes its qubit operands and produces new qubit values, making dataflow explicit.
  QCO is designed for optimization passes that benefit from a static single-assignment (SSA) dataflow graph.

The compiler converts QC to QCO so that optimization passes see explicit dataflow.
To output results, QCO is first converted back to QC, which can then be lowered to QIR or other backends.

More detail on both dialects is available in the {doc}`MLIR in the MQT index <index>`.

**What you will do in this guide:**

- Build a minimal "Hello Quantum" program in QC IR.
- Understand the Bell-state circuit in QC.
- See the same Bell-state in QCO and understand the difference between reference and value semantics.
- Follow how the compiler converts QC to QCO.

## The Setup

To work with MLIR in MQT Core (run the pipeline, run tests, or inspect IR), you need to build from source.
Follow the {doc}`installation guide <../installation>`, specifically the "Building from Source for Performance" and "Development Setup" sections.
The default CMake configuration builds the MLIR pipeline and the compiler unit tests (e.g. the executable in `mlir/unittests/Compiler/`).
Run `cmake --build build` to get binaries and tests.

The {doc}`MLIR in the MQT index <index>` explains how QC, QCO, and conversions (e.g. to QIR) fit together.
This tutorial is the hands-on companion: we write IR and reason about it; the reference docs define every operation in detail.

## "Hello Quantum" in QC: Your First IR Module

In QC, a qubit is a **reference** (type `!qc.qubit`): gates like `qc.h` or `qc.x` modify it in place.
That matches the usual "circuit with wires and gates" mental model.
QC is the dialect you get when importing from Qiskit, OpenQASM, or the MQT Core `QuantumComputation` type; it is also the dialect the pipeline converts into QCO for optimization.

A valid QC module wraps quantum code inside a regular MLIR `module` and `func.func`.
The entry-point function must return an `i64`, use the `entry_point` passthrough attribute, and return a constant `0` to signal success.
Qubits are allocated with `qc.alloc("name", register_size, index)` and released with `qc.dealloc`.

### Single-qubit "Hello Quantum"

The smallest useful program: allocate one qubit, put it in superposition with Hadamard, measure it, then deallocate.

```text
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %c0_i64 = arith.constant 0 : i64

    %q = qc.alloc("q", 1, 0) : !qc.qubit

    qc.h %q : !qc.qubit

    %m = qc.measure("c", 1, 0) %q : !qc.qubit -> i1

    qc.dealloc %q : !qc.qubit

    return %c0_i64 : i64
  }
}
```

**Reading the IR:**
`qc.alloc("q", 1, 0)` allocates the first qubit of a register named `"q"` of size 1.
`qc.h` takes a qubit reference and modifies it in place—no new SSA value is introduced.
`qc.measure("c", 1, 0)` reads the qubit into bit 0 of a 1-bit classical register named `"c"` and returns an `i1`.
`qc.dealloc` releases the qubit.
The same `%q` is used for every gate—that is reference semantics.

:::{tip}
In QC, one name can refer to the same qubit across many operations.
The compiler tracks the *state* of that qubit as it flows through the circuit.
:::

### From one qubit to a Bell state

Next we prepare a **Bell pair** \(\frac{|00\rangle + |11\rangle}{\sqrt{2}}\): Hadamard on the first qubit, then a **controlled-X (CNOT)** with the first qubit as control and the second as target.
In QC, controlled gates are written with `qc.ctrl(control) { body }`: the body is the operation that runs only when the control is \(\lvert 1 \rangle\).

```text
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %c0_i64 = arith.constant 0 : i64

    %q0 = qc.alloc("q", 2, 0) : !qc.qubit
    %q1 = qc.alloc("q", 2, 1) : !qc.qubit

    qc.h %q0 : !qc.qubit

    qc.ctrl(%q0) {
      qc.x %q1 : !qc.qubit
    } : !qc.qubit

    %c0 = qc.measure("c", 2, 0) %q0 : !qc.qubit -> i1
    %c1 = qc.measure("c", 2, 1) %q1 : !qc.qubit -> i1

    qc.dealloc %q0 : !qc.qubit
    qc.dealloc %q1 : !qc.qubit

    return %c0_i64 : i64
  }
}
```

- Inside `qc.ctrl(%q0) { ... }`, the body sees the target qubit(s); `%q0` is the control and is not modified by the body.
- `qc.measure("c", 2, 0)` and `qc.measure("c", 2, 1)` record results into a 2-bit classical register `"c"`.
  The returned `i1` values are perfectly correlated (00 or 11) for an ideal Bell state.

## QC vs QCO: Reference vs Value Semantics

The same quantum operations can be represented in two ways in MLIR:

|               | QC                                       | QCO                                                       |
| ------------: | :--------------------------------------- | :-------------------------------------------------------- |
| **Semantics** | Reference: gates modify qubits in place  | Value: each op consumes inputs, produces new qubit values |
|      **Type** | `!qc.qubit`                              | `!qco.qubit`                                             |
|  **Use case** | Import/export, human-readable circuits   | Optimizations, DAG-style analysis                         |
|       **SSA** | One name can be reused along the circuit | Every gate introduces new SSA values                      |

QC is described in {doc}`QC <QC>`, QCO in {doc}`QCO <QCO>`.
The compiler converts QC → QCO so that optimization passes see explicit dataflow.
To produce output, QCO is converted back to QC, which can then be lowered to QIR or another backend.
Conceptually the pipeline looks like this:

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
  QC (reference semantics)  ←  convert back before lowering
           |
           v
  QIR or other backend      ←  target-specific code
```

### Bell state in QCO (value style)

In QCO, every gate **consumes** its qubit operand(s) and **produces** new qubit value(s).
You never reuse a qubit value after it has been consumed by an operation—that is **linear typing**.
The circuit is the same as the Bell-state example above; only the IR representation changes.

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

Value semantics make dataflow explicit and allow optimization passes to reason about dependencies without worrying about aliasing.

### How the compiler turns QC into QCO

The pipeline has a **QC → QCO** conversion pass.
It keeps the logical circuit but rewrites it so every mutation becomes an explicit "consume old value, produce new value."
For example, this QC snippet:

```text
%q = qc.alloc("q", 1, 0) : !qc.qubit
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

## Advanced Constructs: Multiple Functions

Real circuits are rarely a single flat sequence of gates.
Just like classical programs benefit from breaking logic into functions, quantum circuits can be structured the same way.
In QC and QCO you can define multiple `func.func` functions in the same module, call them with `func.call`, and pass qubits across function boundaries.

### Factoring the Bell-pair preparation into a helper

In QCO, a function that produces qubits simply returns them as part of its result types.
The caller is then responsible for consuming or deallocating those returned qubits—ownership is explicit in the type signature.

```text
module {
  // Helper: allocate a Bell pair and return both unmeasured qubits.
  func.func @prepare_bell() -> (!qco.qubit, !qco.qubit) {
    %q0_0 = qco.alloc : !qco.qubit
    %q1_0 = qco.alloc : !qco.qubit

    %q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit

    %q0_2, %q1_1 = qco.ctrl(%q0_1) targets(%t0 = %q1_0) {
      %t0_1 = qco.x %t0 : !qco.qubit -> !qco.qubit
      qco.yield %t0_1
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    return %q0_2, %q1_1 : !qco.qubit, !qco.qubit
  }

  // Caller: obtain the Bell pair, measure, and release the qubits.
  func.func @measure_bell() {
    %q0, %q1 = func.call @prepare_bell() : () -> (!qco.qubit, !qco.qubit)

    %q0_m, %c0 = qco.measure("c", 2, 0) %q0 : !qco.qubit
    %q1_m, %c1 = qco.measure("c", 2, 1) %q1 : !qco.qubit

    qco.dealloc %q0_m : !qco.qubit
    qco.dealloc %q1_m : !qco.qubit

    return
  }
}
```

**What to notice:**

- `@prepare_bell` returns `(!qco.qubit, !qco.qubit)`—two live qubit values.
  The caller receives them and owns them from that point on.
- The measurement and deallocation happen in `@measure_bell`, not in `@prepare_bell`.
  This mirrors how you would factor a state-preparation subroutine in a classical program.
- Every qubit that crosses a function boundary appears in the function's return type.
  If you forget to return or dealloc a qubit, the IR verifier will reject the module.

This same pattern works in QC: functions can allocate qubits and pass qubit references to callees.
Because QC uses reference semantics, the caller holds the `!qc.qubit` references throughout and passes them as arguments rather than returning new values.

## From QC to QCO and Back

The full pipeline does the following:

1. **Input:** A high-level circuit (e.g. MQT Core `QuantumComputation`) or a QC module.
   Translation from circuits to QC is done by `translateQuantumComputationToQC` or similar.
2. **QC:** Optional canonicalization and cleanup on QC.
3. **QC → QCO:** Conversion to value semantics.
   See {doc}`Conversions <Conversions>` for the conversion passes.
4. **QCO:** Optimization passes run on QCO.
5. **QCO → QC:** Conversion back to reference semantics before lowering.
6. **Output:** QC is lowered to QIR or another backend.

**Next steps:**

- Run the pipeline yourself: `mlir/unittests/Compiler/test_compiler_pipeline.cpp` runs the compiler on a wide variety of programs and compares results to reference QC and QIR output.
  Use it as a template to feed your own QC modules through the pipeline and inspect the IR after each stage.
- Read the dialect references: {doc}`QC <QC>` and {doc}`QCO <QCO>` list every operation and type; {doc}`Conversions <Conversions>` documents the conversion passes.
- Try modifying the examples in this guide (e.g. add a gate) and run them through the compiler to see how QC and QCO change.
