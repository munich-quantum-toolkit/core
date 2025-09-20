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

def GateDecomposition : Pass<"gate-decomposition", "mlir::ModuleOp"> {
  let summary = "This pass will perform various gate decompositions to simplify the used gate set.";
  let description = [{
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

def SwapReconstructionAndElision : Pass<"swap-reconstruction-and-elision", "mlir::ModuleOp"> {
  let summary = "Reconstruct SWAPs from eligible CNOT pairs and eliminate SWAPs by applying the implied qubit permutation.";
  let description = [{
    Eliminates all uncontrolled `swap` operations by:
    1. Reconstructing explicit `swap` ops from eligible CNOT patterns (canonical 3–CNOT form or two alternating CNOT pairs).
    2. Replacing remaining `swap` ops by applying the induced qubit permutation to subsequent uses.

    Notes:
    - No connectivity or coupling constraints are preserved.
    - Controlled `swap` operations (if ever introduced) are not touched.

    Example (reconstruction):
    ```
         ┌───┐         ┌───┐
    ──■──┤ X ├    ──■──┤ X ├──■────■──    ──╳────■──
    ┌─┴─┐└─┬─┘ => ┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐ =>   |  ┌─┴─┐
    ┤ X ├──■──    ┤ X ├──■──┤ X ├┤ X ├    ──╳──┤ X ├
    └───┘         └───┘     └───┘└───┘         └───┘
    ```

    Example (elision):
    ```
                             ┌───┐                 ┌───┐
    q0 ──╳────■── q0    q0 ──┤ X ├──╳── q0    q0 ──┤ X ├─ q1
         |  ┌─┴─┐    =>      └─┬─┘  |      =>      └─┬─┘
    q1 ──╳──┤ X ├ q1    q1 ────■────╳── q1    q1 ────■─── q0
            └───┘
    ```
  }];
}

def QuantumSinkPass : Pass<"quantum-sink", "mlir::ModuleOp"> {
  let summary = "This pass attempts to push down operations into branches for possible optimizations.";
  let description = [{
    This pass searches for branch instructions and attempts to push instructions from previous branches into them.
  }];
}

#endif // MQTO_PASSES
