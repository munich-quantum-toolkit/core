# Getting Started

The Multi-Level Intermediate Representation (MLIR) project is an extensive framework to build compilers for heterogenous hardware. We, the maintainers of the Munich Quantum Toolkit (MQT), explore MLIR for quantum compilation. That is, given an intermediate representation (IR) - a description - of a quantum computation, transform this representation to one that is efficiently executable on a target architecture.

There is problem, however: Getting started with the MLIR framework is not an easy task. Navigating through the overwhelming amount of online resources can already seem like a daunting task. Choosing the right ones, even more so. This heavily impedes open-source contribution. Almost everyone can write Python code, but most do not know the intricacies of MLIR. (TODO: Different motivation - just use the compiler - not contribute to it)

This tutorial attempts to lower the barrier to contribute to our open-source quantum compilation stack. Towards that end, we guide you through the compilation of a quantum program. Alongside, we outline the fundamental data-structures and utilities that support this process.

**Installation**

If you haven't already, make sure to visit the [installation](https://mqt.readthedocs.io/projects/core/en/latest/installation.html) page which describes how to setup the project (including MLIR) correctly.

## Understanding And Writing Quantum IR

### Dynamic and Static Allocation

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

#### Interlude: MLIR Concepts

The snippets above cover already a lot of relevant MLIR concepts.

One of the most important ones are _dialects_. A dialect groups operations (`alloc`, `dealloc`) and types (`qubit`) under a common namespace (`qc`). The example above combines built-in dialects with custom dialects. The [`builtin`](https://mlir.llvm.org/docs/Dialects/Builtin/) dialect provides the `module` operation (the `builtin.` is usually omitted) and the [`func`](https://mlir.llvm.org/docs/Dialects/Func/) dialect contains operations to define and call functions. The custom [`qc`](https://mqt.readthedocs.io/projects/core/en/latest/mlir/QC.html) (_"quantum circuit"_) dialect is defined in the MQT and extends the built-in ones with the necessary functionality for quantum computing.

Operations can consume (_"operands"_) and produce (_"results"_) values. For instance, `qc.alloc` produces the value `q0`, while `qc.dealloc` consumes it. Furthermore, values in MLIR adhere to the static single-assignment (SSA) principle, where each variable is assigned exactly once and never reassigned.

Moreover, some operations contain others. For example, the `module` operation contains the `func.func` operation. In MLIR these nested structures are represented by _regions_ and _blocks_. The following figure visualizes the connection between operations, regions, and blocks succinctly.

```
┌──────────────────────┐
│ Operation            │
├──────────────────────┤
│┌────────────────────┐│
││ Region             ││
│├────────────────────┤│
││ ┌─────────────────┐││
││ │ Block           │││
││ │┌───────────────┐│││
││ ││Operation      ││││
││ │└───────────────┘│││
││ └─────────────────┘││
││ ┌─────────────────┐││
││ │ Block           │││
││ └─────────────────┘││
│└────────────────────┘│
└──────────────────────┘
```

In the snippet above, the `module` operation has one region with exactly one block. Inside this block is the `func.func` op, which again has one region with a single block. Finally, this inner block contains the quantum operations.

TODO: What's up with the func.return, better visualization (connect with IR above)

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

## Optimizing Quantum IR
