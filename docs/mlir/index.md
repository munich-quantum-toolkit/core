# MLIR in the MQT

This part of the MQT explores the capabilities of the Multi-Level Intermediate Representation (MLIR) in the context of compilation for quantum computing.

We define multiple dialects, each with its dedicated purpose:

- The {doc}`QC dialect <QC>` uses reference semantics and is designed as a compatibility dialect that simplifies translations from and to existing languages such as Qiskit, OpenQASM, or QIR.

- The {doc}`QCO dialect <QCO>` uses value semantics and is mainly designed for running optimizations.

Both dialects define various canonicalization and transformations that enable the compilation of quantum programs to native quantum hardware.

For intercompatibility, we provide {doc}`conversions <Conversions>` between dialects.
So far, this comprises conversions between QC and QCO as well as from QC to QIR.

New to MLIR here? Start with the {doc}`Getting Started with MLIR <getting_started>` tutorial.

```{toctree}
:maxdepth: 2

getting_started
QC
QCO
Conversions
```

:::{note}
This page is a work in progress.
The content is not yet complete and subject to change.
Contributions are welcome.
See the {doc}`contribution guide <../contributing>` for more information.
:::
