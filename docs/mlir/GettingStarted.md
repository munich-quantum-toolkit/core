# Getting Started

The Multi-Level Intermediate Representation (MLIR) project is an extensive framework to build compilers for heterogenous hardware. We, the maintainers of the Munich Quantum Toolkit (MQT), explore MLIR for quantum compilation. That is, given an intermediate representation (IR) - a description - of a quantum computation, transform this representation to one that is efficiently executable on a target architecture.

**Installation**

If you haven't already, make sure to visit the [installation](https://mqt.readthedocs.io/projects/core/en/latest/installation.html) page which describes how to setup the project (including MLIR) correctly.

## Understanding Quantum IR

### Dynamic And Static Allocation

The fundamental computational unit in quantum computing is the _qubit_. Consequently, at some point a quantum computation needs to allocate (and subsequently deallocate) qubits.

```mlir
/// file: dynamic-allocation.mlir
module {
    func.func @main() {
        %q0 = qc.alloc : !qc.qubit
        qc.dealloc %q0 : !qc.qubit

        func.return
    }
}
```

An allocation and deallocation defines the start and end of a qubit's logical scope or availability within a program, respectively. The snippet above allocates a dynamic qubit. That is, a qubit without a specific hardware location. To target a specific hardware qubit, we use the `static` operation and provide the respective hardware location.

```mlir
/// file: static-allocation.mlir
module {
    func.func @main() {
        %q0 = qc.static 42 : !qc.qubit
        qc.dealloc %q0 : !qc.qubit

        func.return
    }
}
```

### (Interlude: MLIR Concepts)

```{note}
If you are already familiar with the fundamental concepts of MLIR, you may skip this section.
```

The short snippets above contain many fundamental concepts of MLIR.

- **Dialects**: A dialect groups operations (`alloc`, `dealloc`) and types (`qubit`) under a common namespace (`qc`). The example above combines built-in dialects with custom dialects. The [`builtin`](https://mlir.llvm.org/docs/Dialects/Builtin/) dialect provides the `module` operation (the `builtin.` is usually omitted) and the [`func`](https://mlir.llvm.org/docs/Dialects/Func/) dialect contains operations to define and call functions. The custom [`qc`](./QC.md) (_"quantum circuit"_) dialect is defined in the MQT and extends the built-in ones with the necessary functionality for quantum computing.
- **SSA Values**: Operations can consume (_"operands"_) and produce (_"results"_) values. For instance, `qc.alloc` produces the value `q0`, while `qc.dealloc` consumes it. Furthermore, values in MLIR adhere to the static single-assignment (SSA) principle, where each variable is assigned exactly once and never reassigned.
- **Regions and Blocks**: To represent hierarchical structures, operations may contain _"regions"_. A region consists of one to many _"blocks"_ which again contain operations. For instance, the `module` operation contains one region consisting of one block that contains the `func.func` operation. A block optionally requires a _"terminator"_ that defines the end of the current block. The `func.return` operation is such a terminator. The following figure visualizes the connection between operations, regions, and blocks succinctly.

TODO

### Gates And Measurements

Once the qubits are allocated, we can start applying gates to implement quantum algorithms. In the following snippet we construct the first Bell state by applying a Hadamard and controlled-X gate. Subsequently, we measure both qubits which produces two values of the datatype `i1` - the MLIR-equivalent of a boolean.

```mlir
/// file: bell.mlir
module {
    func.func @main() {
        %q0 = qc.alloc : !qc.qubit
        %q1 = qc.alloc : !qc.qubit

        qc.h %q0 : !qc.qubit
        qc.ctrl(%q0) {
            qc.x %q1 : !qc.qubit
        } : !qc.qubit

        %c0 = qc.measure %q0 : !qc.qubit -> i1
        %c1 = qc.measure %q1 : !qc.qubit -> i1

        qc.dealloc %q0 : !qc.qubit
        qc.dealloc %q1 : !qc.qubit

        func.return
    }
}
```

The Hadamard and X gates are of course not the only ones supported. A list of all supported unitaries can be found in the [documentation](./QC.md#operations).

### Modifiers

Alongside operations representing unitary gates, the QC dialect also provides modifier operations. For instance, in the snippet above, the `qc.ctrl` operation is used to represent the controlled-X operation. Thanks to modifiers, we can represent arbitrary (multi-)controlled gates without having to explicitly define them.

```mlir
module {
    func.func @main() {
        // ... (IR above)
        qc.ctrl(%q0, %q2) {
            qc.s %q1 : !qc.qubit
        } : !qc.qubit, !qc.qubit
        // (IR below) ...
    }
}
```

Another example of a modifier is the `qc.inv` operation, which inverts (complex conjugate transpose) a unitary.

```mlir
module {
    func.func @main() {
        // ... (IR above)
        qc.inv {
            qc.s %q0 : !qc.qubit
        }
        // (IR below) ...
    }
}
```

### Reusable Components

## Optimizing Quantum IR

By combining built-in dialects and the QC dialect we can implement quantum algorithms in MLIR. This section outlines how to use our compiler driver to optimize quantum programs.

<!-- I think the idea of "external" and "internal" is good, but i don't like the wording with "interface"-->

### External Interface

The following command executes the compiler and performs a series of optimizations on the given quantum program. Files using the OpenQASM format will automatically be translated into the QC dialect.

```console
$ mqt-cc [options] <input .mlir/.qasm file>
```

For example, running `mqt-cc` on the first code snippet of this tutorial yields the following IR.

```console
$ mqt-cc dynamic-allocation.mlir
module {
  func.func @main() {
    return
  }
}
```

What happened? Because there are no unitary operations between the allocation and deallocation of the qubit, the `RemoveAllocDeallocPair` canonicalization pattern matches and removes the unused qubit from the program.

### Internal Interface

Internally, the optimizations are performed on the [`qco`](./QCO.md) (_"quantum circuit optimization"_) dialect. While the QC dialect is great for exchanging with other formats (such as OpenQASM), the QCO dialect is specifically designed for optimizations.

The following IR describes the construction of the first Bell state (and subsequent measurement) in the QCO dialect. Each unitary operation consumes and produces SSA values and each SSA value is used at most once (_"linear typing"_). Semantically, a qubit SSA value in the QCO dialect represents the state of the qubit (_"value semantics"_) whereas in the QC dialect a qubit SSA value references a qubit (_"reference semantics"_).

```mlir
/// file: bell-qco.mlir
module {
    func.func @main() {
        %q0_0 = qco.alloc : !qco.qubit
        %q1_0 = qco.alloc : !qco.qubit

        %q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
        %q0_2, %q1_1 = qco.ctrl(%q0_1) targets (%arg0 = %q1_0) {
            %q0_2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
            qco.yield %q0_2
        } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

        %q0_3, %c0 = qco.measure %q0_2 : !qco.qubit
        %q1_2, %c1 = qco.measure %q1_1 : !qco.qubit

        qco.dealloc %q0_3 : !qco.qubit
        qco.dealloc %q1_2 : !qco.qubit
    }
}
```

The following figure illustrates the data-flow graph of the IR above. Thanks to the QCO dialect, the dependencies between operations become immediately apparent. For example, the controlled-X gate depends on the Hadamard gate because it consumes the `q0_1` qubit SSA value. Moreover, MLIR provides the necessary functionality to efficiently traverse the data-flow graph and thus the circuit.

```{image} ../_static/qco-dataflow.svg
:width: 55%
:align: center
```

Quantum IR in the QCO dialect can be quite complex. Writing it by hand is certainly a errorprone task. Fortunately, you don't have to. The compiler driver's interface accepts and produces quantum IR in the QC dialect. Under the hood, it transforms it to the QCO dialect, performs the optimizations, and transforms it back to the QC dialect. That's also why we refer to the QC dialect as interface dialect. The following figure depicts the interplay between the two dialects illustratively.

```{image} ../_static/compilation-pipeline.svg
:width: 35%
:align: center
```

To print the quantum IR after each step in the compilation pipeline, you can supply the `--record-intermediates` option to the compiler driver.

```console
$ mqt-cc --record-intermediates <input .mlir/.qasm file>
```

## Emitting Low-Level Quantum IR

TODO

## Summary
