# Getting Started

The Multi-Level Intermediate Representation (MLIR) project is an extensive framework to build compilers for heterogenous hardware. We, the maintainers of the Munich Quantum Toolkit (MQT), explore MLIR for quantum compilation. That is, given a description of a quantum computation, transform this representation to one that is efficiently executable on a target architecture.

There is problem, however: Getting started with the MLIR framework is not an easy task. Navigating through the overwhelming amount of online resources can already seem like a daunting task. Choosing the right ones, even more so. This heavily impedes open-source contribution. Almost everyone can write Python code, but most do not know the intricacies of MLIR. (TODO: Different motivation - just use the compiler - not contribute to it)

This tutorial attempts to lower the barrier to contribute to our open-source quantum compilation stack. Towards that end, we guide you through the compilation of a quantum program. Alongside, we outline the fundamental data-structures and utilities that support this process.

**Installation**

If you haven't already, make sure to visit the [installation](https://mqt.readthedocs.io/projects/core/en/latest/installation.html) page which describes how to setup the project (including MLIR) correctly.

## "Hello World"

The fundamental computational unit in quantum computing is the _qubit_. Consequently, at some point a quantum computation needs to allocate (and subsequently deallocate) qubits.

```mlir
/// file: allocation.mlir
module {
    func.func @main() {
        %q0 = qc.alloc : !qc.qubit
        qc.dealloc %q0 : !qc.qubit

        func.return
    }
}
```

The tiny snippet above already covers a lot of relevant MLIR concepts.

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

As of now, our quantum program doesn't compute anything. Let's change that.

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

In the updated code snippet, we allocate a second qubit and construct the first Bell state by applying a Hadamard gate and a controlled NOT gate. Finally, we measure and deallocate both qubits.

Sometimes (e.g. when writing unit tests) it can be useful to programmatically build programs. For that purpose, the `QCProgramBuilder` exists. The C++ snippet that follows constructs the above quantum computation programmatically.

```cpp
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

void bell(MLIRContext* context) {
    qc::QCProgramBuilder builder(context);
    builder.initialize();

    const auto q0 = builder.allocQubit();
    const auto q1 = builder.allocQubit();

    builder.h(q0);
    builder.cx(q0, q1);

    const auto c0 = builder.measure(q0);
    const auto c1 = builder.measure(q1);

    builder.dealloc(q0);
    builder.dealloc(q1);

    // Automatically adds the module and entry point function.
    [[maybe_unused]] auto moduleOp = builder.finalize();
}
```

## The Optimization Dialect

## Advanced Concepts
