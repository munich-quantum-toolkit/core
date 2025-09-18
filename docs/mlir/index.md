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
See the {doc}`contribution guide <contributing>` for more information.
:::

## Classical Result Semantics

The `measure` operations of the MQTRef and MQTOpt dialects return classical results as `i1` values.
If an input program defines a classical register, a `memref<?xi1>` operation of appropriate size is allocated and measurement results are stored into it.
Similarly, if the input program contains a classically controlled operation, the necessary `i1` values are loaded from the `memref<?xi1>` operation.

As an example, consider the following `QuantumComputation`:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
x q[0];
c[0] = measure q[0];
```

In the MQTRef dialect, this program corresponds to:

```mlir
module {
  func.func @main() attributes {passthrough = ["entry_point"]} {
    %i = arith.constant 0 : index
    %qreg = memref.alloca() : memref<1x!mqtref.Qubit>
    %q = memref.load %qreg[%i] : memref<1x!mqtref.Qubit>
    %creg = memref.alloca() : memref<1xi1>
    mqtref.x() %q
    %c = mqtref.measure %q
    memref.store %c, %creg[%i] : memref<1xi1>
    return
  }
}
```

### Rationale

The approach of using MLIR-native `memref<?xi1>` operations allows us to stay more flexible.
An alternative definition of an MQT-specific `ClassicalRegister` would have restricted us without adding benefits.
By using `memref<?xi1>` operations, the development of passes can be more agnostic to implementation specifics.

## Development

Building the MLIR library requires LLVM version 21.0 or later.
Our CI pipeline on GitHub continuously builds and tests the MLIR library on Linux, macOS, and Windows.
To access the latest build logs, visit the [GitHub Actions page](https://github.com/munich-quantum-toolkit/core/actions/workflows/ci.yml).
