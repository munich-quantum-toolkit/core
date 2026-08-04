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

def FuseSingleQubitUnitaryRuns
    : Pass<"fuse-single-qubit-unitary-runs", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect",
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

    Currently, only operations whose unitary matrix can be obtained at compile
    time are fused.
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
  let summary = "This pass attempts to move measurements as far up as "
                "possible, shifting them above gates that commute with them. "
                "This is done to enable qubit reuse and other optimizations.";
  let description = [{
    This pass applies measurement lifting, moving measurements up the code as far as possible.
    Measurement lifting is a subroutine of the qubit reuse routine. The goal is to measure qubits earlier in the
    circuit to reuse them and to potentially remove some quantum gates.

    Measurement lifting uses the following commutation rules:
    ```text
         ┌──────┐             ┌──────┐
    ──■──┤ Meas │──────      ─┤ Meas ├──■───
      │  └──────┘             └──────┘  │
    ┌─┴─┐                 =>          ┌─┴─┐
    ┤ U ├────────────────    ─────────┤ U ├─
    └───┘                             └───┘
    ```
    Here, U is any (controlled) unitary gate.

    ```text
    ┌───┐┌──────┐    ┌──────┐┌───┐
    ┤ P ├┤ Meas ├ => ┤ Meas ├┤ P ├
    └───┘└──────┘    └──────┘└───┘
    ```
    Here, P is any diagonal gate, for example `z` or `s`.

    ```text
    ┌───┐┌──────┐    ┌───────┐┌───┐
    ┤ X ├┤ Meas ├ => ┤ Meas* ├┤ X ├
    └───┘└──────┘    └───────┘└───┘
    ```
    Here, Meas* is a measurement after which the outcome is classically negated. X can also be replaced by Y, which
    is likewise anti-diagonal.
  }];
}

def ReplaceClassicalControls
    : Pass<"replace-classical-controls", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect",
  ];
  let summary =
      "This pass attempts to replace controlled gates that are controlled by "
      "values that are in a computational basis state and available as "
      "measurement outcome by a classical "
      "`qco.if` instruction. This is done to enable qubit reuse and other "
      "optimizations.";
  let description = [{
    This pass searches for control operations that immediately follow measurements and replaces them
    with classically controlled operations, represented by `qco.if` operations.
    This reduces quantum interactions and allows for more optimizations, e.g., by enabling qubit reuse.
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

    - Three qubits (X/Z): elementary CCX/CCZ; `rccx` when `min-qubits` is at
      most 3.
    - Four qubits (X/Z): elementary CCCX/CCCZ.
    - Five qubits (X/Z): specialized ancilla-free relative-phase `C^4(Z)`.
    - Six through 33 qubits (X/Z): da Silva-Park SP22 MCP(π) core
      (`H · MCP(π) · H` for X).
    - 34 or more qubits (X/Z): Huang-Palsberg (HP24) borrowed-helper synthesis
      with a compile-time CX policy table.
    - Phase: optimized `C^2(P)` at three qubits; Vale (Barenco-relative
      residual) for four and five qubits; da Silva-Park SP22 linear-depth
      decomposition at six and above. A compile-time angle of `±π` is routed
      through the Z path for every width.
    - Controlled SWAP: `MCSWAP(C, a, b) = CX(a,b) · MCX(C ∪ {b}, a) · CX(a,b)`,
      where `C ∪ {b}` is the original control set together with SWAP target
      `b` and the MCX target is the other SWAP qubit `a`. The emitted MCX is
      then lowered by the X path under the same `min-qubits` threshold.

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
