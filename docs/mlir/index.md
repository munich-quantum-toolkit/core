# MQT Compiler Collection

The MQT Compiler Collection (`mqt-cc`) is a blueprint for a future-proof
quantum-classical compilation framework built on the Multi-Level Intermediate
Representation (MLIR). For an overview, see {cite:p}`MQTCompilerCollection2026`.

The {doc}`Python compiler guide <python_compiler_collection>` describes how to
compile and inspect quantum programs from Python. The
{doc}`target-compilation guide <target_compilation>` shows how to compile for
QDMI devices from Python, C++, and `mqt-cc`. The remaining pages are the
technical reference for the underlying MLIR infrastructure.

We define multiple dialects, each with its dedicated purpose:

- The {doc}`MQT dialect <MQT>` stores frontend-neutral program metadata that
  remains meaningful across dialect conversions.
- The {doc}`QC dialect <QC>` uses reference semantics and is designed as a
  compatibility dialect that simplifies translations from and to existing
  languages such as Qiskit, OpenQASM, or QIR.
- The {doc}`QCO dialect <QCO>` uses value semantics and is mainly designed for
  running optimizations.
- The {doc}`QTensor dialect <QTensor>` adds support for one-dimensional tensors
  of qubits with linear typing and is used in the QCO dialect to represent
  collections of qubits such as registers.
- The {doc}`CBit dialect <CBit>` represents initialized classical-bit registers
  shared by QC and QCO.

These dialects define various canonicalization and transformation passes that
enable the compilation of quantum programs to native quantum hardware. Passes
that are not tied to a single dialect are documented on the
{doc}`passes <Transforms>` page. For interoperability, we provide
{doc}`conversions <Conversions>` between dialects.

The {doc}`OpenQASM interface <OpenQASM>` translates supported OpenQASM input
directly to QC and emits structured OpenQASM from QC.

```{toctree}
:maxdepth: 2

python_compiler_collection
target_compilation
development
MQT
QC
QCO
QTensor
CBit
Transforms
Conversions
OpenQASM
```

:::{note}
This page is a work in progress. The content is not yet complete and subject to
change. Contributions are welcome. See the
{doc}`contribution guide <../contributing>` for more information.
:::
