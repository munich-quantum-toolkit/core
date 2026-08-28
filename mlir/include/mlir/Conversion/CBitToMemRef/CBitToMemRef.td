// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def ConvertCBitToMemRef : Pass<"convert-cbit-to-memref", "mlir::ModuleOp"> {
  let summary = "Lower CBit registers to memrefs";

  let description = [{
    This pass explicitly lowers CBit register types and operations to memrefs.
    A zero-initialized register becomes an allocation followed by a loop that
    stores false to each element. An undefined register becomes an allocation
    only.
    The pass also converts function signatures, calls, returns, branches, and
    structural SCF operation types. It does not infer CBit registers from
    existing memrefs.

    Input may contain at most 64 region-owning operations along any IR path.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}
