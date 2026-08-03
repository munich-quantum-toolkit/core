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

def FuseTwoQubitUnitaryRuns
    : Pass<"fuse-two-qubit-unitary-runs", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect"];
  let summary = "Lower supported one- and two-qubit QCO unitaries to a "
                "user-specified native gate menu.";
  let description = [{
    Rewrites supported one- and two-qubit unitaries to match the comma-separated
    `native-gates` gateset. Fusion and synthesis require unitary matrices that
    are available at compile time. Already-native runtime-parameterized
    operations remain valid, but an off-gateset operation without a compile-time
    matrix cannot be synthesized and causes the pass to fail.

    `qco.barrier` and `qco.gphase` are preserved. A `qco.ctrl` is within the
    two-qubit scope only when it has a single control and a single target. Every
    entangler token listed in the menu is accepted as already native: bare
    `qco.rxx`, `qco.ryy`, `qco.rzx`, `qco.rzz`, `qco.iswap`, or `qco.ecr`, or a
    `qco.ctrl` with exactly one `qco.x` or `qco.z` body when `cx` or `cz` is
    listed. Runtime-parameterized `rxx`/`ryy`/`rzx`/`rzz` operations remain
    valid when their token is listed, although only operations with compile-time
    matrices can participate in fusion or off-gateset synthesis. Entangler
    preference (below) selects the Weyl synthesis basis only and does not
    restrict which menu tokens count as native. Weyl synthesis emits
    `rxx`/`ryy`/`rzx`/`rzz` at a fixed angle of π/2. Gates acting on more than
    two qubits (e.g. multi-controlled `qco.ctrl`) are out of scope and left
    untouched for a dedicated multi-controlled synthesis pass.

    The gateset is a comma-separated list of gate tokens (order not
    significant) from which the pass resolves a single-qubit Euler basis
    (generic `qco.u` when `u` is present; IBM-style surface gates when all of
    `x`, `sx`, and `rz` are present; IQM-style `qco.r` when `r` is present;
    or a supported rotation pair chosen from `rx`, `ry`, `rz`) plus one
    two-qubit entangler from `rxx`, `ryy`, `rzx`, `rzz`, `iswap`, `cz`, `cx`, or
    `ecr`.

    Recognised tokens: `u`, `x`, `sx`, `rz`, `rx`, `ry`, `r`, `rxx`, `ryy`,
    `rzx`, `rzz`, `iswap`, `cz`, `cx`, `ecr`. An empty or whitespace-only
    gateset is a no-op, which is the intended pipeline default when synthesis is
    not needed. An unrecognised token causes the pass to fail.

    Example gatesets (each line is one illustrative gateset; pick one entangler
    or list several if multiple are native):
    - IBM: `x,sx,rz,cx`, `x,sx,rz,cz`, or `x,sx,rz,ecr`
    - Generic single-qubit U: `u,rxx`, `u,ryy`, `u,rzx`, `u,rzz`, `u,iswap`,
      `u,cz`, `u,cx`, or `u,ecr`
    - IQM default: `r,cz` (or `r,cx` if CX is the native entangler)
    - Rotation pair + entangler: `rx,rz,cx`, `rx,ry,cz`, `ry,rz,cx`, etc.
      Supported pairs are exactly `rx`+`rz`, `rx`+`ry`, and `ry`+`rz`.

    When several entanglers are listed, synthesis preference is `rxx` > `ryy` >
    `rzx` > `rzz` > `iswap` > `cz` > `cx` > `ecr` (alphabetic among two-qubit
    rotations; then discrete named gates; ECR last).

    Stages: fuse single-qubit runs (reusing `fuse-single-qubit-unitary-runs`,
    which also lowers lone off-gateset single-qubit gates); fuse two-qubit runs
    and lower any remaining off-gateset two-qubit ops via Weyl synthesis; fuse
    the single-qubit seams introduced by that synthesis. The pass fails if any
    single- or two-qubit op remains off the native gateset afterwards; gates on
    more than two qubits are left untouched and do not cause a failure.

    Lowering is deterministic: entangler preference is `rxx` > `ryy` > `rzx` >
    `rzz` > `iswap` > `cz` > `cx` > `ecr`; single-qubit factors use the resolved
    Euler basis; and two-qubit run replacement uses the minimal entangler count
    from the synthesizer.
  }];
  let options = [Option<
      "nativeGates", "native-gates", "std::string", "\"\"",
      "Comma-separated native gateset. Empty or whitespace-only is "
      "a no-op. Tokens: u, x, sx, rz, rx, ry, r, rxx, ryy, rzx, rzz, iswap, "
      "cz, cx, ecr. "
      "Examples: x,sx,rz,cx; x,sx,rz,ecr; u,rxx; u,ryy; u,rzx; u,rzz; u,iswap; "
      "u,cx; r,cz; rx,rz,cx; x,sx,rz,rzz.">];
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

def QuantumIPO : Pass<"quantum-ipo", "mlir::ModuleOp"> {
  let summary = "Quantum Interprocedural Optimization Pass";
  let description = [{
        Performs interprocedural optimizations on quantum functions.
    }];

  let dependentDialects = ["::mlir::func::FuncDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::qtensor::QTensorDialect",
                           "mlir::qco::QCODialect"];
}];
}

//===----------------------------------------------------------------------===//
// Decomposition Passes
//===----------------------------------------------------------------------===//

def DecomposeMultiControlled
    : Pass<"decompose-multi-controlled", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect"];
  let summary = "Decompose controlled X/Z/phase gates and qco.rccx with at "
                "least min-controls controls";
  let description = [{
    Decomposes controlled X, Z, and constant-angle phase gates (`qco.ctrl` with
    a `qco.x`, `qco.z`, or `qco.p` body, and `qco.rccx`) that have at least
    `min-controls` controls.

    - Two or three controls (X/Z): elementary CCX/CCZ and CCCX/CCCZ; `rccx`
      when `min-controls` is at most 2.
    - Four or five controls (X/Z): specialized ancilla-free cores (relative-
      phase `C^4(Z)`; Vale MCP(π) hybrid residual).
    - Six or more controls (X/Z): Huang-Palsberg (HP24) borrowed-helper
      synthesis with a compile-time CX policy table.
    - Phase: optimized `C^2(P)` at two controls; Vale (Barenco-relative
      residual at four) for three and four controls; da Silva-Park SP22
      linear-depth decomposition at five and above. A compile-time angle of
      `±π` is routed through the Z path for every control count.

    Intermediate two- and three-control building blocks may be left as
    `qco.ctrl` / `qco.rccx` when `min-controls` keeps them; the greedy rewriter
    lowers further when the threshold allows.
  }];
  let options = [Option<
      "minControls", "min-controls", "uint64_t", "2",
      "Decompose controlled X/Z/phase gates and qco.rccx with at least this "
      "many controls (must be at least 2).">];
}

#endif // MLIR_DIALECT_QCO_TRANSFORMS_PASSES_TD
