# Getting Started

The Multi-Level Intermediate Representation (MLIR) project is an extensive framework to build compilers for heterogenous hardware. We, the maintainers of the Munich Quantum Toolkit (MQT), explore MLIR for quantum compilation. That is, given a description of a quantum computation, transform this representation to one that is efficiently executable on a target architecture.

There is problem, however: Getting started with the MLIR framework is not an easy task. Navigating through the overwhelming amount of online resources can already seem like a daunting task. Choosing the right ones, even more so. This heavily impedes open-source contribution. Almost everyone can write Python code, but most do not know the intricacies of MLIR.

This tutorial attempts to lower the barrier to contribute to our open-source quantum compilation stack. Towards that end, we guide you through the compilation of a quantum program. Alongside, we outline the fundamental data-structures and utilities that support this process.

#### Installation

If you haven't already, make sure to visit the [installation](https://mqt.readthedocs.io/projects/core/en/latest/installation.html) page, which describes how to setup the project (including MLIR) correctly.

## "Hello World"

```mlir
module {
    func.func @main() { attributes = ["entry_point"] }{

    }
}
```

## The Optimization Dialect

## Advanced Concepts
