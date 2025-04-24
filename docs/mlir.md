# MLIR in MQT

This part of MQT explores the capabilities of the Multi-Level Intermediate Representation (MLIR) in the context of compilation for quantum computing.
We define multiple dialects, each with its dedicated purpose.
For example, the MQTOpt dialect is designed for optimization of quantum programs and features value-semantics.
Accompanying the dialects, we provide a set of transformations on each dialect and conversions between dialects.

:::{note}
This page is a work in progress.
The content is not yet complete and may be subject to change.
Contributions are welcome.
See the [contribution guidelines](contributing.md) for more information.
:::

## FAQ

### How to print the textual representation of an operation?

During debugging, it can be handy to dump the textual representation of an operation to stdout.
This can be done using the `dump` method of the operation either in the evaluation field of the debugger or in the code.
It prints the textual representation of the operation to the standard output.

```c++
op->dump();
```
