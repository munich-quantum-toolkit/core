# QIR Support in the MQT

The [_Quantum Intermediate Representation_ (QIR)](https://www.qir-alliance.org) is a standardized intermediate representation for quantum programs based on the [_LLVM intermediate representation_ (LLVM IR)](http://llvm.org/).

## The QIR Runtime in MQT Core

MQT Core provides a runtime for QIR that is based on its decision diagram-based quantum simulator.
This allows for the execution of QIR programs using MQT Core's high-performance simulation capabilities.

The runtime can be utilized in two ways:

1.  As a standalone library that can be linked to any QIR program, resulting in a binary executable.
2.  By using the `mqt-core-qir-runner` command-line tool, which interprets QIR programs directly.

### Building the Runner

To build this tool, the CMake option `BUILD_MQT_CORE_QIR_RUNNER` has to be enabled (which depends on `BUILD_MQT_CORE_MLIR` being set).
From the root of the repository, you can build the runner as follows:

```bash
cmake -S . -B build -DBUILD_MQT_CORE_QIR_RUNNER=ON -DBUILD_MQT_CORE_MLIR=ON
cmake --build build --target mqt-core-qir-runner
```

After building, the tool can be found in the build directory under `bin/mqt-core-qir-runner`.

### Executing a QIR Program

The `mqt-core-qir-runner` can be used to execute a QIR file (typically with a `.ll` extension).

```bash
./build/bin/mqt-core-qir-runner bell.ll
```

This will simulate the circuit and print the measurement results to the console.
The runner supports the QIR Base Profile.
