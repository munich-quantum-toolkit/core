# MLIR in the MQT

This part of the MQT explores the capabilities of the Multi-Level Intermediate Representation (MLIR) in the context of compilation for quantum computing.

We define multiple dialects, each with its dedicated purpose:

- The {doc}`MQTRef dialect <MQTRef>` uses reference semantics and is designed as a compatibility dialect that simplifies translations from and to existing languages such as Qiskit, OpenQASM, or QIR.

- The {doc}`MQTOpt dialect <MQTOpt>` uses value semantics and is mainly designed for running optimizations.

Both dialects define various transformation passes.

For intercompatibility, we provide {doc}`conversions <Conversions>` between dialects.
So far, this comprises a conversion from MQTOpt to MQTRef and one from MQTRef to MQTOpt.

```{toctree}
:maxdepth: 2

MQTRef
MQTOpt
Conversions
```

:::{note}
This page is a work in progress.
The content is not yet complete and subject to change.
Contributions are welcome.
See the {doc}`contribution guide <contributing>` for more information.
:::

## Development

Building the MLIR library requires LLVM version 21.0 or later.
Our CI pipeline on GitHub continuously builds and tests the MLIR library on Linux, macOS, and Windows.
To access the latest build logs, visit the [GitHub Actions page](https://github.com/munich-quantum-toolkit/core/actions/workflows/ci.yml).
