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
         ┌───┐ ┌───┐      ┌───┐ ┌───┐
        ─┤ X ├─┤ H ├─ => ─┤ H ├─┤ Z ├─
         └───┘ └───┘      └───┘ └───┘
         ┌───┐ ┌───┐      ┌───┐ ┌───┐
        ─┤ Z ├─┤ H ├─ => ─┤ H ├─┤ X ├─
         └───┘ └───┘      └───┘ └───┘
         ┌───┐ ┌───┐      ┌───┐ ┌───┐                                ┌───────┐
        ─┤ Y ├─┤ H ├─ => ─┤ H ├─┤ Y ├─, while adding a global phase: │ G(pi) │
         └───┘ └───┘      └───┘ └───┘                                └───────┘
    Hadamard lifting is only applied to commute Hadamard gates further in front in the circuit, not the other way
    around. A global phase is added if the Hadamard gate is commuted with a Pauli-Y gate, as YH = -HY.

    The routine is only applied to non-controlled Pauli and Hadamard gates.

    In order to reduce multi-qubit gates, a second transformation routine is applied. Using the commutation rule
                 ┌───┐┌───┐┌───┐
        ──■──    ┤ H ├┤ X ├┤ H ├
        ┌─┴─┐ => │───│└─┬─┘│───│
        ┤ X ├    ┤ H ├──■──┤ H ├
        └───┘    └───┘     └───┘
    the following transformation is applied to circuits where a Hadamard gate follows a target gate of a controlled
    Pauli-X gate, which is followed by a measurement:
                              ┌───┐┌───┐┌───┐
        ──■─────────────      ┤ H ├┤ X ├┤ H ├──
        ┌─┴─┐┌───┐┌──────┐ => │───│└─┬─┘│───┘──┐
        ┤ X ├┤ H ├┤ Meas │    ┤ H ├──■──┤ Meas │
        └───┘└───┘└──────┘    └───┘     └──────┘

     Afterward the measurement lifting routine could transform the CNOT into a classically controlled Pauli-X.

  }];
}

def ConstantPropagation : Pass<"constant-propagation", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect"];
  let summary =
      "This pass applies constant propagation to a circuit. It "
      "assumes that all input qubits are |0>. It propagates the "
      "state of the qubits up to a given complexity threshold and "
      "removes gates which are superfluous considering the current "
      "state. It also replaces quantum control with classical control "
      "if possible and moves measurements as far to the front as possible.";
  let description = [{
     This pass applies quantum constant propagation. This optimization routines assumes that the input qubits of the
     circuits are |0>. It propagates the qubit states and the state of additional classical values through the circuit.
     All quantum instructions are removed which are superfluous considering the current state. Additionally, quantum
     controls can be replaced by equivalent classical control. To do this most efficiently, the measurements are moved
     as far to the front of a circuit as possible.

     The qubit states and classical values are stored in hybrid states. Hybrid states are stored in a union table to
     reduce the amount of complex amplitudes and classical values to track. There is a maximum number of non zero
     amplitudes that is saved per union table entry. Additionally, there is also a maximum of hybrid states that can be
     propagated. If the maximum number of amplitudes or the maximum number of hybrid states is exceeded, the propagated
     state reaches top and no optimization routines are further applied.

     The applied optimization routines are:

     **General Control Reduction**
     If a controlling qubit or classical value is always true, the control is removed. Classical vales are considered
     true if they are not zero.
     If a controlling qubit or classical value is alaways false, the complete gate is removed.

     **Unsatisfiable Controls**
     If a combination of controls (both quantum and classical) cannot be satisfied, the complete controlled gate is
     removed.

     **Quantum Control Reduction**
     If a controlling qubit is true if and only if a classical value is true (false), the quantum control is replaced by a
     classical one. If the classical value is false if and only if the qubit is true, the classical value becomes a
     negative control.

     **Implied Qubits**
     If a controlling qubit is implied by other controlling qubits/classical values, the implied qubit is removed from
     the controls.

     **Phase Gate Reduction**
     If a phase gate would only apply a global phase (e.g. if the quantum state is |11> and a Z-gate is applied), the
     phase gate is removed and replaced by a global phase gate instead.

  }];
  let options = [Option<"maximumNonzeroAmplitudes", "maximumNonzeroAmplitudes",
                        "std::size_t", "4",
                        "The maximum number of non-zero amplitudes in the "
                        "tracted quantum states before reaching top.">,
                 Option<"maximumHybridStates", "maximumHybridStates",
                        "std::size_t", "4",
                        "The maximum number of hybrid states which have a "
                        "non-zero probability.">];
}

#endif // MLIR_DIALECT_QCO_TRANSFORMS_PASSES_TD
