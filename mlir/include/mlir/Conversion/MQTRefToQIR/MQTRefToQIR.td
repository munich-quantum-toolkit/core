// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTRefToQIR : Pass<"mqtref-to-qir"> {
   let summary = "Convert MQT's `Ref` dialect into the llvm dialect that adheres to the qir specification";

   let description = [{
      This pass converts all operations to the llvm dialect. The quantum operations of the mqt ref dialect are converted
      to an equivalent operation in the qir instruction set expressed with a function call in the llvm dialect. This requires a valid
      input in the mqtref dialect where the main function is marked with the entry_point attribute. At the moment the pass only supports
      simple quantum programs with a straight controlflow. In order to adhere to the base profile of QIR, the initial module is divided
      into multiple blocks and an initialize operation is added. During the conversion both QubitType and RegisterType from mqtref are
      replaced with llvm.ptr type to represent an opaque pointer in the llvm dialect.
      The remaining operations from the standard dialects are lowered to the llvm dialect via built-in conversion passes from mlir.
      This includes the following dialects at the moment: func, arith, cf

      Once the module is converted fully in the llvm dialect, one can convert the resulting file into LLVM-IR with the following command:
      mlir-translate --mlir-to-llvmir input.mlir > output.ll
   }];

  // Define dependent dialects
  let dependentDialects = [
     "::mqt::ir::ref::MQTRefDialect",
     "mlir::LLVM::LLVMDialect"
  ];
}
