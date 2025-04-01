// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTCoreRoundTrip : Pass<"mqt-core-round-trip", "mlir::ModuleOp"> {
  let summary = "This pass performs a round trip to MQT Core's QuantumComputation and back";
  let description = [{
    This pass starts by extracting all quantum-computation-relevant operations and then
    translates them into an MQT Core quantum computation, deleting all quantum operations
    in the process. Then, the quantum computation is read in a second pattern and used
    to generate new MLIR code from it.

    This pass makes the following assumptions:
    - Each module consists of a single function that uses `mqtopt` operations with just a single qubit register.
    - All qubits are measured exactly once and the boolean measurement results are returned from the function in order.
    - The first returned value is the `AllocOp` that constructs a qubit register.
    - Measurements/Operations may only target a single qubit. Unitary operations may also use any number of (positive) controls.
    - Supported operations are `x`, `y`, `z`, `h`, and `measure`.
  }];
}
