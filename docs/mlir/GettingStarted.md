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

Operations can consume (_"operands"_) and produce (_"results"_) values. For instance, the `qc.alloc` operation produces the value `q0`, while the `qc.dealloc` operation consumes it. Furthermore, values in MLIR adhere to the static single-assignment (SSA) principle, where each variable is assigned exactly once and never reassigned.

Moreover, purely from a visual perspective, one can notice that the snippet above contains nested structures. For example, the `module` operation contains the `func.func` operation.

## The Optimization Dialect

## Advanced Concepts
