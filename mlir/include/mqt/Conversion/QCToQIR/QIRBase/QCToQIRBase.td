// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCToQIRBase : Pass<"qc-to-qir-base", "mlir::ModuleOp"> {
  let summary = "Lower the QC dialect to the LLVM dialect compliant with the "
                "QIR Base Profile 2.1";

  let description = [{
    Lower supported QC operations to QIR Base Profile runtime calls and
    LLVM dialect operations. Run `set-qir-attributes-and-metadata` separately
    to attach QIR attributes and module flags.

    Requirements:

    - Stores to returned CBit registers must share a block with their measurement,
      use an index available at measurement (or a constant), and have no
      intervening classical memory effects except stores to provably distinct
      constant indices of the same register.
    - Input is a valid module in the QC dialect.
    - The entry function must be marked with `mqt.entry_point`.
    - The input entry function must consist of a single block.
      Multi-block input functions are currently not supported.
    - The program must have straight-line control flow (i.e., Base Profile QIR).
    - A measured qubit must not be used by another quantum instruction,
      including another measurement. Gates on independent qubits may follow
      measurements in the input.
    - Explicit static qubit IDs cannot be mixed with qubit allocations.
    - Qubit-register loads require constant, in-bounds indices into statically
      sized allocations. Register aliases must be resolved before conversion.

    Behavior:

    - Constant classical tensors and element reads are lowered to constant LLVM arrays and loads.
      Other classical tensor operations must be lowered before this pass.

    - Each QC quantum operation is replaced in place by its QIR call. After
      validating qubit usage in instruction order, the pass moves terminal
      measurements to the irreversible operations block.
    - The pass transforms the single-block entry function into four blocks to satisfy QIR Base Profile constraints:
      0. Initialization block: Sets up the execution environment and performs required runtime initialization.
      1. Reversible operations block: Contains only void-returning calls to reversible quantum operations.
      2. Irreversible operations block: Contains terminal measurement calls (`__quantum__qis__mz__body`). Reset operations are not supported.
      3. Epilogue block: Records measurement results and returns from the entry function.
    - Blocks are connected via unconditional branches in the order listed above.
    - Non-quantum dialects are lowered via MLIR's built-in conversions.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}
