# MLIR dialects and passes

This reference describes the MQT Compiler Collection's MLIR infrastructure. For
compilation, simulation, and device execution, start with
{doc}`compilation and execution <../compilation/index>`. Compiler contributions
follow the [development policy](../development.md#mlir).

We define multiple dialects, each with its dedicated purpose:

- The {doc}`MQT dialect <MQT>` stores frontend-neutral program metadata that
  remains meaningful across dialect conversions.
- The {doc}`QC dialect <QC>` uses reference semantics and is designed as a
  compatibility dialect that simplifies translations from and to existing
  representations such as Qiskit circuits, OpenQASM, or QIR.
- The {doc}`QCO dialect <QCO>` uses value semantics and is mainly designed for
  running optimizations.
- The {doc}`QTensor dialect <QTensor>` adds support for one-dimensional tensors
  of qubits with linear semantics and is used in the QCO dialect to represent
  collections of qubits such as registers.
- The {doc}`CBit dialect <CBit>` represents initialized classical-bit registers
  shared by QC and QCO.

These dialects define various canonicalization patterns and transformation
passes that enable the compilation of quantum programs to native quantum
hardware. Passes that are not tied to a single dialect are documented on the
{doc}`passes <Transforms>` page. For interoperability, we provide
{doc}`conversions <Conversions>` between dialects.

```{toctree}
:maxdepth: 2

MQT
QC
QCO
QTensor
CBit
Transforms
Conversions
```
