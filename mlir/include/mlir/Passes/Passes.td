// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QCO_PASSES
#define QCO_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Transpilation Passes
//===----------------------------------------------------------------------===//

def MappingPass : Pass<"place-and-route", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary = "This pass ensures that a program meets the connectivity constraints of a target architecture.";
  let description = [{
    This pass initially assigns static qubits to the dynamically allocated ones ("placement")
    by creating an initial dynamic-to-static mapping (the initial "layout"). Consequently, it
    traverses the circuit and inserts `SWAP` operations such that all two-qubit operations are
    executable on the target architecture ("routing").

    For routing, the pass first divides the circuit into layers. A layer is a set of
    independently executable two-qubit operations. Subsequently, the pass performs an A*
    search for each layer to find and insert a sequence of `SWAP` operations that makes these
    two-qubit operations executable. The A* search also considers the subsequent `nlookahead`
    layers.

    The cost function of the A* search is defined as follows:

    * `f(n) = g(n) + h(n)`
    * `g(n) = alpha * depth(n)`
    * `h(n) = sum(pow(lambda, i) * h(L, pi) for [i, L] in enumerate(layers))`

    Where:
    * `pi` is the dynamic-to-static mapping associated with search node `n`.
    * `layers` is an array of layers with size `1 + nlookahead`.
    * `depth(n)` returns the distance from the node `n` to the root node.
    * `dist(i, j)` returns the distance between the qubits `i` and `j` on the target's coupling graph.
    * `h(L, pi) := sum(dist(pi[gate.first], pi[gate.second]) for gate in L)`

    To improve on the chosen initial layout, the pass iteratively performs `iterations` forward and
    backward routing traversals.
  }];
  let options = [
    Option<"nlookahead", "nlookahead", "std::size_t", "1",
           "The number of lookahead steps.">,
    Option<"alpha", "alpha", "float", "1.0F",
           "The alpha factor in the cost function. Must be > 0.">,
    Option<"lambda", "lambda", "float", "0.5F",
           "The lambda factor in the cost function.">,
    Option<"iterations", "iterations", "std::size_t", "2",
           "The number of forwards and backwards traversal to improve the initial layout.">
  ];
}

#endif // QCO_PASSES
