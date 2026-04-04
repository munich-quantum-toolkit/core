// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QCO_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_QCO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MergeSingleQubitRotationGates
    : Pass<"merge-single-qubit-rotation-gates", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect"];
  let summary = "Merge rotation gates using quaternion-based fusion";
  let description = [{
    Merges consecutive single-qubit rotation gates acting on the same qubit into
    a single equivalent `u` gate, reducing circuit depth and gate count.

    Supported gate types: `rx`, `ry`, `rz`, `p`, `r`, `u2`, `u`.

    The pass greedily collects the longest possible chain of consecutive
    mergeable gates. Each gate is converted to a unit quaternion:

    - `rx`, `ry`, `rz`, `p`: single-axis rotations via half-angle formulas.
    - `r(theta, phi)`: rotation by `theta` around axis `(cos(phi), sin(phi), 0)`.
    - `u2(phi, lambda) = u(pi/2, phi, lambda)`.
    - `u(theta, phi, lambda)`: ZYZ decomposition `rz(phi)*ry(theta)*rz(lambda)`,
      each factor converted to a quaternion and merged via the Hamilton product.

    The gates are then folded one by one via the Hamilton product into a single
    quaternion, which is decomposed back into ZYZ Euler angles and emitted as a
    single `u` gate.

    Gates that are not purely SU(2) carry a global phase (`p`, `u`, `u2`).
    This phase is accumulated across the chain and emitted as a correcting
    global phase gate.
  }];
}

//===----------------------------------------------------------------------===//
// Transpilation Passes
//===----------------------------------------------------------------------===//

def MappingPass : Pass<"place-and-route", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary = "This pass ensures that a program meets the connectivity "
                "constraints of a target architecture.";
  let description = [{
    This pass maps the dynamically allocated qubits in a quantum program to the static qubits of a target architecture.
    The pass performs both placement and routing of the qubits to ensure that all two-qubit operations in the program
    can be executed on the target architecture.

    First, the pass assigns static qubits to the dynamically allocated ones by creating an initial dynamic-to-static
    mapping, which is referred to as the initial layout. Then, it traverses the circuit and inserts `SWAP` operations to
    ensure that all two-qubit operations are executable on the target architecture, a process known as routing.

    For routing, the pass first divides the circuit into layers. A layer is a set of independently executable (sequences
    or blocks of) two-qubit operations. Subsequently, the pass performs an A* search for each layer to find and insert a
    sequence of `SWAP` operations that makes these two-qubit operations executable. The A* search also considers
    subsequent layers, which are determined by the `nlookahead` parameter.

    The cost function of the A* search is defined as follows:

    - `f(n) = g(n) + h(n)`
    - `g(n) = alpha * depth(n)`
    - `h(n) = sum(pow(lambda, i) * h(L, p) for [i, L] in enumerate(layers))`

    Where:

    - `p` is the dynamic-to-static mapping associated with search node `n`.
    - `layers` is an array of layers with size `1 + nlookahead`.
    - `depth(n)` returns the distance from the node `n` to the root node.
    - `dist(i, j)` returns the distance between the qubits `i` and `j` on the target's coupling graph.
    - `h(L, p) := sum(dist(p[gate.first], p[gate.second]) for gate in L)`

    To iteratively refine the mapping, the pass performs multiple forward and backward traversals of the circuit. In
    each traversal, the pass routes the circuit and updates the dynamic-to-static mapping based on the routing decisions
    made during that traversal. By performing multiple traversals, the pass can iteratively refine the mapping and
    potentially find a more optimal solution. This is behavior is controlled by the `niterations` parameter.

    The pass option `ntrials` determines how many random initial layouts the pass explores. If compiled with multi-threading on, these trials will be executed in parallel.
  }];
  let options = [Option<"nlookahead", "nlookahead", "std::size_t", "1",
                        "The number of lookahead steps.">,
                 Option<"alpha", "alpha", "float", "1.0F",
                        "The alpha factor in the cost function. Must be > 0.">,
                 Option<"lambda", "lambda", "float", "0.5F",
                        "The lambda factor in the cost function.">,
                 Option<"niterations", "niterations", "std::size_t", "2",
                        "The number of forwards and backwards traversal to "
                        "improve the initial layout.">,
                 Option<"ntrials", "ntrials", "std::size_t", "4",
                        "The number of (possibly parallel) random trials of "
                        "the forwards and backwards mechanism. Must be > 0.">,
                 Option<"seed", "seed", "std::size_t", "42",
                        "A seed used for randomization.">];
  let statistics = [Statistic<"numSwaps", "num-inserted-swaps",
                              "The number of inserted SWAPs">];
}

#endif // MLIR_DIALECT_QCO_TRANSFORMS_PASSES_TD
