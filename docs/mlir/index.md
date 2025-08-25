# MLIR in the MQT

This part of the MQT explores the capabilities of the Multi-Level Intermediate Representation (MLIR) in the context of compilation for quantum computing.

We define multiple dialects, each with its dedicated purpose:

- The {doc}`MQTOpt dialect <MQTOpt>` is designed for optimization.
  We also provide a set of passes for the dialect.

- The {doc}`MQTRef dialect <MQTRef>` uses reference semantics and is designed as a compatibility dialect that simplifies translations from and to existing languages such as QASM or QIR.
  Also the MQTRef dialect is accompanied by passes.

For intercompatibility, we provide {doc}`conversions <Conversions>` between dialects.
So far, this comprises a conversion from MQTOpt to MQTRef and one from MQTRef to MQTOpt.

:::{note}
This page is a work in progress.
The content is not yet complete and subject to change.
Contributions are welcome.
See the [contribution guidelines](contributing.md) for more information.
:::

```{toctree}
:maxdepth: 1
:caption: Table of Contents

MQTOpt Dialect <MQTOpt>
MQTRef Dialect <MQTRef>
Conversions
```
