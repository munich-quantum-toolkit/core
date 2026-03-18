// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MergeRotationGates : Pass<"qco-merge-rotation-gates",
"mlir::ModuleOp"> {
  let summary = "Merge rotation gates using quaternion-based fusion";
  let description = [{
    Merges consecutive rotation gates of different types (rx, ry, rz, u)
    using quaternion representation and Hamilton product.
  }];

  let dependentDialects = ["::mlir::math::MathDialect"];
}

def ReuseQubitsPass : Pass<"reuse-qubits", "mlir::ModuleOp"> {
  let dependentDialects = [ "mlir::scf::SCFDialect" ];
  let summary = "Reduce the number of required qubits by reusing existing ones that are no longer used.";
  let description = [{
    This pass performs qubit reuse, searching for qubits that are first referenced after the final reference to a different qubit.
    In these cases, rather than using a new qubit, the old qubit can be reset and reused.
  }];
}


def QuantumIPO : Pass<"quantum-ipo", "mlir::ModuleOp"> {
    let summary = "Quantum Interprocedural Optimization Pass";
    let description = [{
        Performs interprocedural optimizations on quantum functions.
    }];

    let dependentDialects = ["::mlir::func::FuncDialect", "::mlir::arith::ArithDialect", "::mlir::tensor::TensorDialect"];
}
