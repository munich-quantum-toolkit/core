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
  let summary = "Merge single-qubit gates using quaternion-based fusion";
  let description = [{
    Merges consecutive mergeable single-qubit gates acting on the same qubit
    into a single equivalent U gate, reducing circuit depth and gate count.

    Supported gate types: `rx`, `ry`, `rz`, `p`, `r`, `u2`, `u`, `x`, `y`,
    `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, and `id`.

    The pass greedily collects the longest possible chain of consecutive mergeable gates.
    Each gate is converted to a unit quaternion:

    - `rx`, `ry`, `rz`, `p`: single-axis rotations via half-angle formulas.
    - `x`, `y`, `z`: fixed-axis `pi` rotations (`rx`/`ry`/`rz`).
    - `s` / `sdg`: `rz(+/- pi / 2)`; `t` / `tdg`: `rz(+/- pi / 4)`.
    - `sx` / `sxdg`: `rx(+/- pi / 2)`.
    - `h`: a `pi` rotation around the `(x + z) / sqrt(2)` axis.
    - `id`: the identity quaternion.
    - `r(theta, phi)`: rotation by `theta` around axis `(cos(phi), sin(phi), 0)`.
    - `u2(phi, lambda) = u(pi / 2, phi, lambda)`.
    - `u(theta, phi, lambda)`: ZYZ decomposition `rz(phi) * ry(theta) * rz(lambda)`, each factor converted to a quaternion and merged via the Hamilton product.

    The gates are then folded one by one via the Hamilton product into a single quaternion, which is decomposed back into ZYZ Euler angles and emitted as a single `UOp`, representing the same unitary as the input chain.
    The global phase of each gate is tracked alongside and combined together
    (`x`/`y`/`z`/`h`: `pi / 2`; `s`/`sx`: `pi / 4`; `sdg`/`sxdg`: `-pi / 4`;
    `t`/`tdg`: `+/- pi / 8`; `id` and pure `SU(2)` rotations: `0`;
    `p(theta)`: `theta / 2`; `u`/`u2`: `(phi + lambda) / 2`).

    When every angle in the chain is a compile-time constant, the same algorithm
    runs on host scalars and emits constant `U` / `gphase` results directly.
    Otherwise it emits `arith` / `math` operations that compute the merged
    parameters at run time.

    The emitted `UOp` is defined by $U = \exp [i (\phi + \lambda) / 2] R_z (\phi) R_y (\theta) R_z (\lambda)$.
    Normalizing either extracted Z angle into $[-\pi, \pi)$ by $\pm 2\pi$ flips
    the $\mathrm{SU}(2)$ representative; half of each removed angle is returned
    as an Euler-wrap phase correction. Each merge therefore computes
    $\mathrm{inputPhase} - (\phi + \lambda) / 2 + \mathrm{eulerPhase}$, which
    restores exact matrix equality (including global phase). This applies even
    to chains of purely $\mathrm{SU}(2)$ gates (`rx`, `ry`, `rz`, `r`).
    On the host (static) path, a near-zero correction may be omitted; on the
    SSA (dynamic) path a `GPhaseOp` is always emitted. After greedy merging,
    the implementation directly invokes the shared global-phase normalization
    utility to combine, normalize, and remove trivial corrections in their
    respective scopes.
  }];
}

def FuseSingleQubitUnitaryRuns
    : Pass<"fuse-single-qubit-unitary-runs", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::qtensor::QTensorDialect"];
  let summary = "Fuse single-qubit unitary runs using Euler resynthesis";
  let description = [{
    Matches maximal runs of consecutive single-qubit unitary operations on the
    same qubit wire (anchored at each run head), composes their constant unitary
    matrices, and replaces a run with an equivalent sequence of basis gates when
    beneficial: when the run contains a gate outside the target `basis`, or when
    Euler resynthesis would shorten it (`synthesizeUnitary1QEuler`). Runs that are
    already in the target `basis` and no shorter than the canonical synthesis
    length are left unchanged.

    The emitted basis is controlled via the `basis` option (e.g. `zyz`, `zsxx`).
    A `gphase` correction is inserted when needed so the rewritten sequence
    matches the composed matrix exactly (not only up to global phase).

    The pass also composes supported named gates with dynamic `f64` parameters
    and emits conservative runtime sequences in the requested basis. Dynamic
    `pow` and arbitrary dynamic unitaries remain unchanged.
  }];
  let options = [Option<
      "basis", "basis", "std::string", "\"zyz\"",
      "Target Euler basis (zyz, zxz, xzx, xyx, u, zsxx, r).">];
}

//===----------------------------------------------------------------------===//

def QuantumLoopUnroll
    : InterfacePass<"quantum-loop-unroll", "FunctionOpInterface"> {
  let dependentDialects = ["mlir::qco::QCODialect", "mlir::scf::SCFDialect"];
  let summary = "Unroll bounded loops containing quantum operations";
  let description = [{
    This pass unrolls `scf.for` operations which contain quantum operations. Classical loops are not unrolled.

    Because the pass uses the `mlir::loopUnrollByFactor` utility internally, the same assumptions and restrictions hold for both.
    See the [MLIR documentation](https://mlir.llvm.org/doxygen/include_2mlir_2Dialect_2SCF_2Utils_2Utils_8h.html) for more information.
  }];
  let options = [Option<"unrollFactor", "unroll-factor", "int64_t",
                        /*default=*/"-1",
                        "Use this unroll factor for all loops being unrolled, "
                        "set it to -1, and it will fully unroll.">];
}

//===----------------------------------------------------------------------===//
// Transpilation Passes
//===----------------------------------------------------------------------===//

def MappingPass : Pass<"place-and-route", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary = "This pass ensures that a program meets the connectivity "
                "constraints of a target architecture.";
  let description = [{
    This pass maps top-level scalar and tensor-backed dynamically allocated
    qubits in a quantum program to the static sites of a `CompilerTarget`. It
    uses only the target's undirected coupling topology: arbitrary one-qubit
    operations pass through, and every two-qubit operation is routed to adjacent
    target vertices. Higher-arity unitaries must be decomposed before mapping.
    Target-defined site identifiers are retained in the resulting
    `qco.static` operations.

    First, the pass assigns static qubits to the dynamically allocated ones by creating an initial dynamic-to-static
    mapping, which is referred to as the initial layout. The layout covers the
    complete target virtually, while only program qubits and vacant sites used
    as routing workspace are materialized. Then, the pass traverses the circuit
    and inserts `qco.swap` operations to ensure that all two-qubit operations
    are executable on the target architecture, a process known as routing.

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
                 Option<"niterations", "niterations", "std::size_t", "1",
                        "The number of forwards and backwards traversal to "
                        "improve the initial layout. Must be > 0.">,
                 Option<"ntrials", "ntrials", "std::size_t", "4",
                        "The number of (possibly parallel) random trials of "
                        "the forwards and backwards mechanism. Must be > 0.">,
                 Option<"seed", "seed", "std::size_t", "42",
                        "A seed used for randomization.">];
  let statistics = [Statistic<"numSwaps", "num-inserted-swaps",
                              "The number of inserted SWAPs">];
}

//===----------------------------------------------------------------------===//
// Optimization Passes
//===----------------------------------------------------------------------===//

def PauliTwirl2QGates : Pass<"pauli-twirl-2q-gates", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect"];
  let summary = "Apply Pauli twirling to supported two-qubit gates";
  let description = [{
    Surrounds each supported two-qubit gate with two single-qubit Pauli gates
    before and after the gate. The pass selects uniformly from the 16 twirls
    that preserve the exact gate unitary and inserts a global-phase correction
    when required. Identity gates remain explicit.

    The pass supports controlled X and Z gates with one control and one target,
    ECR gates, and iSWAP gates. Gates nested in modifier regions remain
    unchanged.
  }];
  let options = [Option<"seed", "seed", "uint64_t", "42",
                        "Seed used to select Pauli twirls.">];
}

def HadamardLifting : Pass<"hadamard-lifting", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary = "This pass attempts to move Hadamard gates as far away from "
                "measurements as possible by commuting them "
                "with Pauli gates. This is done in order to apply measurement "
                "lifting more efficiently, which is a sub-routine of "
                "qubit reuse. Additionally, it can change target and control "
                "qubits from Pauli-Z gates to make Hadamard lifting "
                "applicable. It also lifts Hadamard gates over CNOT gates if "
                "that moves a measurement directly after a control.";
  let description = [{
    This pass lifts Hadamard gates away from the measurements in order to apply measurement lifting more effectively.
    Measurement lifting is a subroutine of the qubit reuse routine. The goal is to measure qubits earlier in the
    circuit to reuse them and to potentially remove some quantum gates.

    Measurement lifting commutes measurements with Pauli and phase gates. However, the routine stops if a Hadamard gate
    is applied before the measurement takes place. Hadamard lifting attempts to move Hadamards further in front of the
    circuit, in order to improve the results provided by measurement lifting.

    Hadamard lifting uses the following commutation rules:
    ```text
         ┌───┐ ┌───┐      ┌───┐ ┌───┐
        ─┤ X ├─┤ H ├─ => ─┤ H ├─┤ Z ├─
         └───┘ └───┘      └───┘ └───┘
         ┌───┐ ┌───┐      ┌───┐ ┌───┐
        ─┤ Z ├─┤ H ├─ => ─┤ H ├─┤ X ├─
         └───┘ └───┘      └───┘ └───┘
         ┌───┐ ┌───┐      ┌───┐ ┌───┐                                ┌───────┐
        ─┤ Y ├─┤ H ├─ => ─┤ H ├─┤ Y ├─, while adding a global phase: │ G(pi) │
         └───┘ └───┘      └───┘ └───┘                                └───────┘
    ```

    Hadamard lifting is only applied to commute Hadamard gates further in front in the circuit, not the other way
    around. A global phase is added if the Hadamard gate is commuted with a Pauli-Y gate, as YH = -HY.

    The routine is only applied to non-controlled Pauli and Hadamard gates.

    In order to reduce multi-qubit gates, a second transformation routine is applied. Using the commutation rule
    ```text
                 ┌───┐┌───┐┌───┐
        ──■──    ┤ H ├┤ X ├┤ H ├
        ┌─┴─┐ => │───│└─┬─┘│───│
        ┤ X ├    ┤ H ├──■──┤ H ├
        └───┘    └───┘     └───┘
    ```

    the following transformation is applied to circuits where a Hadamard gate follows a target gate of a controlled
    Pauli-X gate, which is followed by a measurement:
    ```text
                              ┌───┐┌───┐┌───┐
        ──■─────────────      ┤ H ├┤ X ├┤ H ├──
        ┌─┴─┐┌───┐┌──────┐ => │───│└─┬─┘│───┘──┐
        ┤ X ├┤ H ├┤ Meas │    ┤ H ├──■──┤ Meas │
        └───┘└───┘└──────┘    └───┘     └──────┘
    ```

    Afterward the measurement lifting routine could transform the CNOT into a classically controlled Pauli-X.

  }];
}

def MeasurementLifting : Pass<"measurement-lifting", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect"];
  let summary = "Move measurements before compatible unitary operations";
  let description = [{
    This pass moves computational-basis measurements toward the start of a
    circuit. A measurement of a control can move before the controlled
    operation. A single-qubit diagonal operation immediately before a
    measurement is removed because it cannot affect either the outcome or the
    post-measurement qubit. Multi-qubit diagonal operations are only commuted:
    removing them could discard phase kickback on another qubit. Measurements
    also move before `x` and `y`, with the classical result inverted.

    Earlier measurements enable qubit reuse and subsequent optimizations.
  }];
}

def ReplaceClassicalControls
    : Pass<"replace-classical-controls", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect",
  ];
  let summary = "Replace measured controls and diagonal targets with "
                "classical data flow";
  let description = [{
    This pass rewrites a `qco.ctrl` that immediately consumes one or more
    measured controls into one `qco.if`. Multiple measurement outcomes are
    combined into the condition, while unmeasured controls remain quantum
    controls inside it. The controlled body is not duplicated.

    The pass also simplifies supported diagonal operations with measured
    targets. A measured `rz` target becomes an outcome-dependent phase on the
    controls. One measured `rzz` target turns the operation into an
    outcome-dependent `rz` on the other target; two measured targets leave only
    phase kickback on coherent controls. If every participating qubit has been
    measured, the diagonal operation is removed. Other single-target phase
    gates can exchange a measured target with an unmeasured control before the
    general rewrite.

    Replacing quantum controls with classical conditions reduces quantum
    interactions and can enable qubit reuse and subsequent optimizations.
  }];
}

def ReuseQubits : Pass<"reuse-qubits", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary =
      "This pass attempts to utilize qubit reuse to reduce the number of "
      "resources required for a quantum program. "
      "It does so by looking for qubits that do not interact with each other "
      "directly or indirectly and attempting to reset and reuse one of them "
      "for the other. "
      "By combining this pass with the `measurement-lifting` and "
      "`replace-classical-controls` passes, it is possible to reduce the "
      "number of qubits required for a quantum program significantly.";
  let description = [{
    This pass searches for qubits that do not interact with each other directly or indirectly and attempts to reset
    and reuse one of them for the other. Improved results can be achieved by combining this pass with the
    `measurement-lifting` and `replace-classical-controls` passes, which can reduce the number of qubits required for a quantum program significantly.
    Only single-qubit allocations are considered for reuse, no `qtensor` allocations.

    Use the `mqt-qubit-reuse` pipeline to run the preparation passes followed by this pass.
  }];
}

def RemoveDeadGates : Pass<"remove-dead-gates", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary = "Remove quantum gates whose results cannot be observed";
  let description = [{
    This pass removes dead quantum operations by walking backwards from
    `qco.sink` and `qco.reset` operations. It removes memory-effect-free
    unitary operations and other removable quantum operations until it reaches
    an operation whose result is observable or whose effects must be preserved.

    Dead-gate removal is explicit rather than part of canonicalization because
    measurement-free programs may intentionally represent state preparation or
    larger quantum building blocks. The `mqt-qubit-reuse` pipeline runs this
    pass before attempting to reuse qubits.
  }];
}

//===----------------------------------------------------------------------===//
// Decomposition Passes
//===----------------------------------------------------------------------===//

def DecomposeMultiControlled
    : Pass<"decompose-multi-controlled", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect"];
  let summary = "Decompose controlled X/Z/phase/SWAP gates and qco.rccx that "
                "act on at least min-qubits qubits";
  let description = [{
    Decomposes multi-qubit controlled operations that act on at least
    `min-qubits` qubits (default 3: everything wider than a two-qubit gate).
    Supported shapes: `qco.ctrl` with a `qco.x`, `qco.z`, `qco.swap`, or
    constant-angle `qco.p` body, and `qco.rccx`.

    | Family | Width (qubits) | Decomposition |
    | ------ | -------------- | ------------- |
    | X/Z | 3 | Elementary CCX/CCZ; `rccx` when `min-qubits` is at most 3 |
    | X/Z | 4 | Elementary CCCX/CCCZ |
    | X/Z | 5 | Specialized ancilla-free relative-phase `C^4(Z)` |
    | X/Z | 6–33 | da Silva-Park SP22 MCP(π) core (`H · MCP(π) · H` for X) |
    | X/Z | ≥34 | Huang-Palsberg (HP24) borrowed-helper synthesis with a compile-time CX policy table |
    | Phase | 3 | Optimized `C^2(P)` |
    | Phase | 4–5 | Vale (Barenco-relative residual) |
    | Phase | ≥6 | da Silva-Park SP22 linear-depth |
    | Phase | any (`±π`) | Routed through the Z path |
    | SWAP | ≥ `min-qubits` | `MCSWAP(C, a, b) = CX(a,b) · MCX(C ∪ {b}, a) · CX(a,b)` |

    Width is the total number of qubits the gate acts on (controls plus
    targets).

    For controlled SWAP, `C ∪ {b}` is the original control set together with
    SWAP target `b`, and the MCX target is the other SWAP qubit `a`. The
    emitted MCX is then lowered by the X path under the same `min-qubits`
    threshold.

    Intermediate building blocks may be left as `qco.ctrl` / `qco.rccx` when
    `min-qubits` keeps them; the greedy rewriter lowers further when the
    threshold allows.
  }];
  let options = [Option<
      "minQubits", "min-qubits", "uint64_t", "3",
      "Decompose controlled X/Z/phase/SWAP gates and qco.rccx that act on at "
      "least this many qubits (must be at least 3; default 3 means wider than "
      "two-qubit).">];
}

#endif // MLIR_DIALECT_QCO_TRANSFORMS_PASSES_TD
