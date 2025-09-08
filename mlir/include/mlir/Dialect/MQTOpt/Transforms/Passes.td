// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTO_PASSES
#define MQTO_PASSES

include "mlir/Pass/PassBase.td"

def MQTCoreRoundTrip : Pass<"mqt-core-round-trip", "mlir::ModuleOp"> {
  let summary = "This pass performs a round trip to MQT Core's QuantumComputation and back";
  let description = [{
    This pass starts by extracting all quantum-computation-relevant operations and then translates them into an MQT Core `QuantumComputation`,
    deleting all quantum operations in the process.
    Then, the quantum computation is read in a second pattern and used to generate new MLIR code from it.

    This pass makes the following assumptions:
    - Each module consists of a single function that uses `mqtopt` operations with just a single qubit register.
    - All qubits are measured exactly once and the boolean measurement results are returned from the function in order.
    - The first returned value is the `AllocOp` that constructs a qubit register.
    - Unitary operations may also use any number of (positive) controls.
  }];
}

def GateElimination : Pass<"gate-elimination", "mlir::ModuleOp"> {
  let summary = "This pass searches for consecutive applications of gates and their inverses and cancels them.";
  let description = [{
    This pass searches for applications of gates that are their own inverses. Walking down their def-use chain,
    it then checks if the same gate is applied once again. In that case, the two gates are cancelled.
    Additionally, all occurrences of the identity gate are removed.
  }];
}

def MergeRotationGates : Pass<"merge-rotation-gates", "mlir::ModuleOp"> {
  let summary = "This pass searches for consecutive applications of rotation gates that can be merged.";
  let description = [{
    Consecutive applications of gphase, `p`, `rx`, `ry`, `rz`, `rxx`, `ryy`, `rzz`, and `rzx` are merged into one by adding their angles.
    The merged gate is currently not removed if the angles add up to zero.

    This pass currently does not affect `xx_minus_yy`, `xx_plus_yy`, `u`, and `u2`.
  }];
}

def ElidePermutations : Pass<"elide-permutations", "mlir::ModuleOp"> {
  let summary = "This pass removes any permutation (i.e., SWAP gates) it encounters and modifies the order of the qubits instead.";
  let description = [{
    Elides permutations (i.e., SWAP gates) by removing them from the circuit and, instead, permuting the respective qubit values.

    This optimization will not preserve the current qubit mapping and has no awareness of potential connectivity limitations.

    Example:
    ```
    // before
    %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // after
    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    ```
  }];
}

def QuantumSinkPass : Pass<"quantum-sink", "mlir::ModuleOp"> {
  let summary = "This pass attempts to push down operations into branches for possible optimizations.";
  let description = [{
    This pass searches for branch instructions and attempts to push instructions from previous branches into them.
  }];
}

def RoutingPass : Pass<"routing", "mlir::ModuleOp"> {
  let summary = "This pass ensures that the connectivity constraints of a target architecture are met.";
  let description = [{
    This pass inserts SWAP operations to ensure two-qubit gates are executable on a given target architecture.
  }];
}

def RoutingVerificationPass : Pass<"verify-routing", "mlir::ModuleOp"> {
  let summary = "This pass verifies that the connectivity constraints of a target architecture are met.";
  let description = [{
    This pass ensures that all two-qubit gates are executable on the target's architecture.
  }];
}

#endif // MQTO_PASSES
