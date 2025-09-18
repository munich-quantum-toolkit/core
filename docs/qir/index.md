# QIR in the MQT

The [_Quantum Intermediate Representation_ (QIR)](https://www.qir-alliance.org) is a standardized intermediate representation for quantum programs based on the [_LLVM intermediate representation_ (LLVM IR)](http://llvm.org/).
MQT provides a QIR runtime that is based on the decision diagram-based quantum simulator that is bundled with MQT Core.
The runtime can be utilized in two ways:

1. As a standalone library that can be linked to any QIR program resulting in a binary executable.
2. Or, by executing the QIR runner incorporating the QIR runtime that interprets QIR programs directly.

For the latter option, MQT Core provides a command-line tool called `mqt-core-qir-runner` that can be used to execute QIR programs.
To build this tool, the CMake option `BUILD_MQT_QIR_RUNNER` has to be enabled (this depends on the CMake option `BUILD_MQT_MLIR`).
Then, after building the executable's target `mqt-core-qir-runner`, the tool can be found in the build directory under `bin/mqt-core-qir-runner`.
It can be used as follows:

```bash
mqt-core-qir-runner <qir-file> [<args>...]
```
