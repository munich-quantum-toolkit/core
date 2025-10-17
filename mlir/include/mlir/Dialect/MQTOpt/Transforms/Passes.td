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

//===----------------------------------------------------------------------===//
// Round-Trip Passes
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Optimization Passes
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Transpilation Passes For Superconducting Devices
//===----------------------------------------------------------------------===//

def PlacementPassSC : Pass<"placement-sc", "mlir::ModuleOp"> {
  let summary = "This pass maps dynamic qubits to static qubits on superconducting quantum devices using initial placement strategies.";
  let options = [
    Option<"strategy", "strategy", "PlacementStrategy", "PlacementStrategy::Random",
           "The initial placement strategy to use.", [{llvm::cl::values(
             clEnumValN(PlacementStrategy::Random, "random", "Random placement"),
             clEnumValN(PlacementStrategy::Identity, "identity", "Identity placement"))}]>
  ];
  let statistics = [
    Statistic<"tms", "runtime-milliseconds", "The runtime in milliseconds">
  ];
}

def RoutingPassSC : Pass<"route-sc", "mlir::ModuleOp"> {
  let summary = "This pass ensures that a program meets the connectivity constraints of a given architecture.";
  let description = [{
    This pass inserts SWAP operations to ensure two-qubit gates are executable on a given target architecture.
  }];
  let options = [
    Option<"method", "method", "RoutingMethod", "RoutingMethod::QMAP",
           "The routing method to use.", [{llvm::cl::values(
             clEnumValN(RoutingMethod::Naive, "naive", "Swap along shortest paths"),
             clEnumValN(RoutingMethod::QMAP, "astar", "A*-search-based routing algorithm"))}]>,
    Option<"nlookahead", "nlookahead", "std::size_t", "1",
           "qmap option: Number of lookahead steps (heuristic horizon)">,
    Option<"alpha", "alpha", "float", "1.0F",
           "qmap option: The alpha factor in the cost function">,
    Option<"beta", "beta", "float", "1.0F",
           "qmap option: The beta factor in the cost function">,
    Option<"lambda", "lambda", "float", "0.9F",
           "qmap option: The lambda factor in the cost function">
  ];
  let statistics = [
    Statistic<"nadd", "n-additional-gates", "The number of additional gates">,
    Statistic<"t_ms", "runtime-milliseconds", "The runtime in milliseconds">
  ];
}

def RoutingVerificationSCPass : Pass<"verify-routing-sc", "mlir::ModuleOp"> {
  let summary = "This pass verifies that a program meets the connectivity constraints of a given architecture.";
  let description = [{
    This pass ensures that all two-qubit gates are executable on the target's architecture.
  }];
}

#endif // MQTO_PASSES
