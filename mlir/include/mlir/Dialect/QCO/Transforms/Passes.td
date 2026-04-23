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
    Merges consecutive single-qubit rotation gates acting on the same qubit into a single equivalent U gate, reducing circuit depth and gate count.

    Supported gate types: `rx`, `ry`, `rz`, `p`, `r`, `u2`, `u`.

    The pass greedily collects the longest possible chain of consecutive mergeable gates.
    Each gate is converted to a unit quaternion:

    - `rx`, `ry`, `rz`, `p`: single-axis rotations via half-angle formulas.
    - `r(theta, phi)`: rotation by `theta` around axis `(cos(phi), sin(phi), 0)`.
    - `u2(phi, lambda) = u(pi / 2, phi, lambda)`.
    - `u(theta, phi, lambda)`: ZYZ decomposition `rz(phi) * ry(theta) * rz(lambda)`, each factor converted to a quaternion and merged via the Hamilton product.

    The gates are then folded one by one via the Hamilton product into a single quaternion, which is decomposed back into ZYZ Euler angles and emitted as a single `UOp`, representing the same rotation as the chain of single gates.
    The global phase of each gate is tracked alongside and combined together.

    The emitted `UOp` is defined by $U = \exp [i (\phi + \lambda) / 2] R_z (\phi) R_y (\theta) R_z (\lambda)$.
    Each merge emits a `GPhaseOp` carrying the accumulated input phase of the chain.
    Because the synthesized `UOp` introduces an additional intrinsic phase of $(\phi + \lambda) / 2$, the `GPhaseOp` must compensate for it.
    This applies even to chains composed entirely of $\mathrm{SU} (2)$ gates (`rx`, `ry`, `rz`, `r`) because the synthesis into a `UOp` still produces the intrinsic phase term.
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

def NativeGateSynthesisPass : Pass<"native-gate-synthesis", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary = "Lower QCO unitary gates to a user-specified native gate menu.";
  let description = [{
    This pass rewrites a module so that every remaining unitary operation is
    allowed by the `native-gates` menu. `qco.barrier` and `qco.gphase` are
    preserved; controlled gates (`qco.ctrl`) must have a single control and a
    single target.

    The menu is a comma-separated list of gate tokens (order not significant)
    from which the pass builds a profile: a single-qubit synthesis mode
    (generic `qco.u` when `u` is present; IBM-style surface gates when all of
    `x`, `sx`, and `rz`/`p` are present; IQM-style `qco.r` when `r` is present;
    or a supported rotation pair chosen from `rx`, `ry`, `rz`) plus optional
    two-qubit entanglers `cx`, `cz`, and optional `rzz`.

    Recognised tokens: `u`, `x`, `sx`, `rz` (or `p`), `rx`, `ry`, `r`, `cx`,
    `cz`, `rzz`. An empty or whitespace-only menu is a no-op, which is the
    intended pipeline default when synthesis is not needed. An unrecognised
    token or an invalid score weight (non-finite or negative) causes the pass
    to fail.

    Example menus (each line is one illustrative menu; pick either `cx` or
    `cz` as the entangler, or list both if both are native):
    - IBM basic (no fractional two-qubit): `x,sx,rz,cx` or `x,sx,rz,cz`
    - IBM fractional: `x,sx,rz,rx,rzz,cx` or `x,sx,rz,rx,rzz,cz`
    - Generic single-qubit U: `u,cx` or `u,cz`
    - IQM default: `r,cz` (or `r,cx` if CX is the native entangler)
    - Rotation pair + entangler: `rx,rz,cx`, `rx,ry,cz`, `ry,rz,cx`, etc.
      Supported pairs are exactly `rx`+`rz`, `rx`+`ry`, and `ry`+`rz`.

    Execution order (mirrors the implementation): fuse consecutive
    single-qubit runs; consolidate two-qubit windows (including absorbed
    single-qubit padding); run up to four synthesis sweeps over remaining
    non-native unitaries until every single-qubit op matches the menu (two-qubit
    lowering may temporarily emit off-menu 1q ops that later sweeps absorb—if
    any remain after that cap, the pass fails); fuse 1q seams between two-qubit
    blocks; merge `rz` through eligible `qco.ctrl` control wires; fuse 1q runs
    again; then up to four further synthesis + fusion rounds until the full menu
    holds (including native `qco.ctrl` shells and bare `rzz` when allowed). If
    anything is still off-menu, the pass fails.

    Candidate selection minimises the linear cost
    `score-weight-twoq * #2q + score-weight-oneq * #1q +
    score-weight-depth * local-depth`. Defaults (`1.0 / 0.1 / 0.01`) favour
    minimising two-qubit count first, then single-qubit count, then depth.

    `qco.ctrl` wrappers whose body is `qco.x` or `qco.z` are left untouched
    when `cx` or `cz` is on the menu; otherwise they are treated as a `4×4`
    unitary and go through the same two-qubit search as bare two-qubit gates.
    `qco.xx_plus_yy` / `qco.xx_minus_yy` are lowered via a dedicated
    `rzz`-centric rewrite when `rzz` is on the menu, and via the general
    two-qubit decomposition otherwise.
  }];
  let options =
      [Option<"nativeGates", "native-gates", "std::string", "\"\"",
              "Comma-separated native gate menu. Empty or whitespace-only is "
              "a no-op. Tokens: u, x, sx, rz (or p), rx, ry, r, cx, cz, rzz. "
              "Examples: x,sx,rz,cx; x,sx,rz,rx,rzz,cz; u,cx; r,cz; rx,rz,cx.">,
       Option<"scoreWeightTwoQ", "score-weight-twoq", "double", "1.0",
              "Weight for the number of two-qubit gates in candidate "
              "scoring. Must be finite and non-negative.">,
       Option<"scoreWeightOneQ", "score-weight-oneq", "double", "0.1",
              "Weight for the number of single-qubit gates in candidate "
              "scoring. Must be finite and non-negative.">,
       Option<"scoreWeightDepth", "score-weight-depth", "double", "0.01",
              "Weight for the local candidate depth in candidate scoring. "
              "Must be finite and non-negative.">];
}

#endif // MLIR_DIALECT_QCO_TRANSFORMS_PASSES_TD
