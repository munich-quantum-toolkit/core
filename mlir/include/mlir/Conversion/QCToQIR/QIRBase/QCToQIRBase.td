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
    This pass lowers all operations from the QC dialect to their equivalent operations in the LLVM dialect, ensuring compliance with the QIR Base Profile 2.1 standard.
    It translates quantum operations and types from QC to their corresponding representations in QIR, facilitating interoperability with quantum computing frameworks that utilize the QIR standard.

    Requirements:

    - Input is a valid module in the QC dialect.
    - The entry function must be marked with `mqt.entry_point`.
    - The entry function must be defined and have no arguments.
    - The input entry function must consist of a single block.
      Multi-block input functions are currently not supported.
    - QC, CBit, and MemRef operations must occur only in the entry function.
    - The total width of all `cbit.alloc` operations must not exceed 1,048,576
      (`1 << 20`) classical result slots per module.
    - Existing `__quantum__` runtime symbols must be Func or LLVM function
      declarations.
    - Qubit memrefs must be direct, statically sized, one-dimensional
      allocations accessed with constant in-bounds indices.
    - The program must have straight-line control flow (i.e., Base Profile QIR).
    - Operations with observable effects or non-speculatable behavior must
      precede every `qc.measure` and `qc.reset` unless they are independent
      quantum operations on qubits that have not yet been measured or reset.
      This lets the block split commute independent quantum operations while
      preserving each qubit's order. Semantically inert no-op/resource
      plumbing and pure, speculatable operations may also follow irreversible
      operations.

    Behavior:

    - Each QC quantum operation is replaced by a call to the corresponding QIR function in the LLVM dialect.
    - QIR module flags and entry-point passthrough metadata are attached by the
      separate QIR metadata pass, normally through the QIR cleanup pipeline.
    - The pass transforms the single-block entry function into four blocks to satisfy QIR Base Profile constraints:
      0. Initialization block: Sets up the execution environment and performs required runtime initialization.
      1. Reversible operations block: Contains only void-returning calls to reversible quantum operations.
      2. Irreversible operations block: Contains only void-returning calls to operations marked irreversible (e.g., `__quantum__qis__mz__body` and `__quantum__qis__reset__body`).
      3. Epilogue block: Records measurement results and returns from the entry function.
    - Blocks are connected via unconditional branches in the order listed above.
    - Supported Func, Arith, CF, and Math operations are lowered via MLIR's
      built-in conversions; other dialects are preserved.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}
