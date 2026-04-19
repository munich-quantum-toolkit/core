# The MQT Compiler Collection

The MQT Compiler Collection (`mqt-cc`) is a blueprint for a future-proof quantum-classical compilation framework built
on the Multi-Level Intermediate Representation (MLIR).
For an overview, see {cite:p}`MQTCompilerCollection2026`.

This page gives a technical introduction.
It links to the API documentation for the MLIR infrastructure.

We define multiple dialects, each with its dedicated purpose:

- The {doc}`QC dialect <QC>` uses reference semantics and is designed as a compatibility dialect that simplifies translations from and to existing languages such as Qiskit, OpenQASM, or QIR.
- The {doc}`QCO dialect <QCO>` uses value semantics and is mainly designed for running optimizations.
- The {doc}`QTensor dialect <QTensor>` adds support for one-dimensional tensors of qubits with linear typing and is used in the QCO dialect to represent collections of qubits such as registers.

These dialects define various canonicalization and transformation passes that enable the compilation of quantum programs
to native quantum hardware.
For interoperability, we provide {doc}`conversions <Conversions>` between dialects.

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
