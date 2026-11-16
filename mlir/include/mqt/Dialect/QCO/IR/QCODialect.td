// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QCO_IR_QCODIALECT_TD
#define MLIR_DIALECT_QCO_IR_QCODIALECT_TD

include "mlir/IR/DialectBase.td"

def QCODialect : Dialect {
  let name = "qco";

  let summary = "The QCO (value semantics) dialect for quantum computing.";

  let description = [{
    QCO (Quantum Circuit Optimization) represents each qubit at a particular
    point in the computation as an SSA value. A gate consumes its input values
    and produces the values used by subsequent operations. This model, called **value
    semantics**, makes dependencies explicit, so optimizations can follow the
    quantum data flow directly.

    ```mlir
    %q_out = qco.h %q_in
    %q0_out, %q1_out = qco.swap %q0_in, %q1_in
    ```

    Each quantum SSA value has exactly one use. It represents a qubit's state
    at that point, but does not contain a simulated statevector. Allocation
    operations for qubits and qubit tensors must appear in the entry block of the
    `mqt.entry_point` function. Helper functions receive their qubits as
    arguments; the entry-point verifier checks this allocation rule.

    The exactly-one-use rule does not determine which output continues which
    input wire. Core's `WireOpInterface` specifies that correspondence
    positionally: the first quantum result continues the first quantum operand,
    and so on. A SWAP exchanges their states, not their wire identities.
    `qco::verifyLinearity` checks use counts as well as correspondence through
    functions and structured control flow.

    Register slots retain their qubit identities, and a gate or call must
    receive distinct quantum resources. See the shared
    [quantum resource rules](index.md#quantum-resource-rules) for register
    access, helper functions, and conversion to QC.
  }];

  let cppNamespace = "::mlir::qco";
  let dependentDialects = ["::mlir::arith::ArithDialect"];

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_QCO_IR_QCODIALECT_TD
