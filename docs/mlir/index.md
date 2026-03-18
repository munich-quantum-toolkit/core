# MLIR in the MQT

This part of the MQT explores the capabilities of the Multi-Level Intermediate Representation (MLIR) in the context of compilation for quantum computing.

We define multiple dialects, each with its dedicated purpose:

- The {doc}`QC dialect <QC>` uses reference semantics and is designed as a compatibility dialect that simplifies translations from and to existing languages such as Qiskit, OpenQASM, or QIR.

- The {doc}`QCO dialect <QCO>` uses value semantics and is mainly designed for running optimizations.

- The {doc}`QTensor dialect <QTensor>` adds support for one-dimensional tensors of qubits with linear typing and is used in the QCO dialect to represent collections of qubits such as registers.

These dialects define various canonicalization and transformations that enable the compilation of quantum programs to native quantum hardware.

For intercompatibility, we provide {doc}`conversions <Conversions>` between dialects.
So far, this comprises conversions between QC and QCO as well as from QC to QIR.

```{toctree}
:maxdepth: 2

QC
QCO
QTensor
Conversions
```

:::{note}
This page is a work in progress.
The content is not yet complete and subject to change.
Contributions are welcome.
See the {doc}`contribution guide <../contributing>` for more information.
:::
