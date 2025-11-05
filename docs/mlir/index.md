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
See the {doc}`contribution guide <../contributing>` for more information.
:::

## Register Handling

In MQT's MLIR dialects, quantum and classical registers are represented by MLIR-native `memref` operations rather than custom types.
This design choice offers several advantages:

- **MLIR Integration**: Seamless compatibility with existing MLIR infrastructure, enabling reuse of memory handling patterns and optimization passes
- **Implementation Efficiency**: No need to define and maintain custom register operations, significantly reducing implementation complexity
- **Enhanced Interoperability**: Easier integration with other MLIR dialects and passes, allowing for more flexible compilation pipelines
- **Sustainable Evolution**: Standard memory operations can handle transformations while allowing new features without changing the fundamental register model

### Quantum Register Representation

A quantum register is represented by a `memref` of type `!mqtref.Qubit` or `!mqtopt.Qubit`, depending on the dialect:

```mlir
// A quantum register with 2 qubits
%qreg = memref.alloc() : memref<2x!mqtref.Qubit>
```

Individual qubits are accessed through standard memory operations:

```mlir
// Load qubits from the register
%q0 = memref.load %qreg[%i0] : memref<2x!mqtref.Qubit>
%q1 = memref.load %qreg[%i1] : memref<2x!mqtref.Qubit>
```

Here's a complete example of quantum register allocation, qubit access, and deallocation:

```mlir
module {
  func.func @main() attributes {passthrough = ["entry_point"]} {
    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtref.Qubit>
    %q0 = memref.load %qreg[%i0] : memref<2x!mqtref.Qubit>
    %q1 = memref.load %qreg[%i1] : memref<2x!mqtref.Qubit>
    memref.dealloc %qreg : memref<2x!mqtref.Qubit>
    return
  }
}
```

### Classical Register Representation

Classical registers follow the same pattern as quantum registers but use the `i1` type for boolean measurement results:

```mlir
// A classical register with 1 bit
%creg = memref.alloc() : memref<1xi1>
```

Measurement operations produce `i1` values that can be stored in classical registers:

```mlir
// Measure a qubit and store the result
%c = mqtref.measure %q
memref.store %c, %creg[%i0] : memref<1xi1>
```

### Example: Full Quantum Program with Measurement

Consider the following quantum computation represented in OpenQASM 3.0 code:

```qasm3
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
x q[0];
c[0] = measure q[0];
```

In the MQTRef dialect, this corresponds to:

```mlir
module {
  func.func @main() attributes {passthrough = ["entry_point"]} {
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtref.Qubit>
    %q = memref.load %qreg[%i0] : memref<1x!mqtref.Qubit>
    %creg = memref.alloca() : memref<1xi1>
    mqtref.x() %q
    %c = mqtref.measure %q
    memref.store %c, %creg[%i0] : memref<1xi1>
    memref.dealloc %qreg : memref<1x!mqtref.Qubit>
    return
  }
}
```

## Development

Building the MLIR library requires LLVM version 21.0 or later.
Our CI pipeline on GitHub continuously builds and tests the MLIR library on Linux, macOS, and Windows.
To access the latest build logs, visit the [GitHub Actions page](https://github.com/munich-quantum-toolkit/core/actions/workflows/ci.yml).
