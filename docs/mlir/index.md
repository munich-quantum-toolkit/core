# MLIR in the MQT

This part of the MQT explores the capabilities of the Multi-Level Intermediate Representation (MLIR) in the context of compilation for quantum computing.

We define multiple dialects, each with its dedicated purpose:

- The MQTOpt dialect is designed for optimization.
  See the [dialect documentation](mlir/Dialects/MLIRMQTOptDialect.md) and [interface definitions](mlir/Dialects/MLIRMQTOptInterfaces.md) for more information.
  We also provide a set of [passes](mlir/Passes/MLIRMQTOptPasses.md) for the MQTOpt dialect.

- The MQTRef dialect uses reference semantics and is designed as a compatibility dialect that simplifies
  translations from and to existing languages such as QASM or QIR.
  See the [dialect documentation](mlir/Dialects/MLIRMQTRefDialect.md) and [interface definitions](mlir/Dialects/MLIRMQTRefInterfaces.md) for more information.
  Also the MQTRef dialect is accompanied by [passes](mlir/Passes/MLIRMQTRefPasses.md).

For intercompatibility, we provide conversions between dialects.
So far, this comprises a conversion from [MQTOpt to MQTRef](mlir/Conversions/MLIRMQTOptToMQTRef.md) and one from [MQTRef to MQTOpt](mlir/Conversions/MLIRMQTRefToMQTOpt.md).

:::{note}
This page is a work in progress.
The content is not yet complete and subject to change.
Contributions are welcome.
See the [contribution guidelines](contributing.md) for more information.
:::
