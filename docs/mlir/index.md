# MQT Compiler Collection

The MQT Compiler Collection (`mqt-cc`) compiles quantum-classical programs using
the Multi-Level Intermediate Representation (MLIR). For an overview, see
{cite:p}`MQTCompilerCollection2026`.

The {doc}`compiler guide <mqt_compiler_collection>` introduces the Python,
command-line, and C++ interfaces for compiling and inspecting quantum programs.
The {doc}`target-compilation guide <target_compilation>` shows how to compile
for QDMI devices from Python, C++, and `mqt-cc`. The interface guides describe
OpenQASM and Qiskit interoperability, followed by the MLIR technical reference
and compiler development guidance.

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

The {doc}`OpenQASM interface <OpenQASM>` translates supported OpenQASM input
directly to QC and emits structured OpenQASM from QC.

```{toctree}
:maxdepth: 2

GettingStarted
mqt_compiler_collection
target_compilation
OpenQASM
qiskit
MQT
QC
QCO
QTensor
CBit
Transforms
Conversions
development
```
