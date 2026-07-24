/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <string>
#include <variant>

namespace mlir::qir {

LLVM::LLVMFuncOp getMainFunction(Operation* op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<ModuleOp>();
  }
  if (!module) {
    return nullptr;
  }

  // Search for function with entry_point attribute
  for (const auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
    auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough");
    if (!passthrough) {
      continue;
    }
    if (llvm::any_of(passthrough, [](Attribute attr) {
          const auto strAttr = dyn_cast<StringAttr>(attr);
          return strAttr && strAttr.getValue() == "entry_point";
        })) {
      return funcOp;
    }
  }
  return nullptr;
}

LLVM::LLVMFuncOp getOrCreateFunctionDeclaration(OpBuilder& builder,
                                                Operation* op, StringRef fnName,
                                                Type fnType) {
  // Check if the function already exists
  auto* fnDecl =
      SymbolTable::lookupNearestSymbolFrom(op, builder.getStringAttr(fnName));

  if (fnDecl == nullptr) {
    // Save current insertion point
    const OpBuilder::InsertionGuard guard(builder);

    // Create the declaration at the end of the module
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) {
      module = op->getParentOfType<ModuleOp>();
    }
    if (!module) {
      llvm::reportFatalInternalError("Module not found");
    }
    builder.setInsertionPointToEnd(module.getBody());

    fnDecl = LLVM::LLVMFuncOp::create(builder, op->getLoc(), fnName, fnType);

    // Add irreversible attribute to irreversible quantum operations
    if (fnName == QIR_MEASURE || fnName == QIR_RESET) {
      fnDecl->setAttr("passthrough", builder.getStrArrayAttr({"irreversible"}));
    }
  }

  return cast<LLVM::LLVMFuncOp>(fnDecl);
}

LLVM::AddressOfOp createResultLabel(OpBuilder& builder, Operation* op,
                                    const StringRef label,
                                    const StringRef symbolPrefix) {
  // Save current insertion point
  const OpBuilder::InsertionGuard guard(builder);

  auto module = dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<ModuleOp>();
  }
  if (!module) {
    llvm::reportFatalInternalError("Module not found");
  }

  const auto symbolName =
      builder.getStringAttr((symbolPrefix + "_" + label).str());

  if (!module.lookupSymbol<LLVM::GlobalOp>(symbolName)) {
    const auto llvmArrayType = LLVM::LLVMArrayType::get(
        builder.getIntegerType(8), static_cast<unsigned>(label.size() + 1));
    const auto stringInitializer = builder.getStringAttr(label.str() + '\0');

    // Create the declaration at the start of the module
    builder.setInsertionPointToStart(module.getBody());

    const auto globalOp = LLVM::GlobalOp::create(
        builder, op->getLoc(), llvmArrayType, /*isConstant=*/true,
        LLVM::Linkage::Internal, symbolName, stringInitializer);
    globalOp->setAttr("addr_space", builder.getI32IntegerAttr(0));
    globalOp->setAttr("dso_local", builder.getUnitAttr());
  }

  // Create AddressOfOp
  // Shall be added to the first block of the `main` function in the module
  auto main = getMainFunction(op);
  if (!main) {
    llvm::reportFatalInternalError("Main function not found");
  }
  auto& firstBlock = *(main.getBlocks().begin());
  builder.setInsertionPointToStart(&firstBlock);

  const auto addressOfOp = LLVM::AddressOfOp::create(
      builder, op->getLoc(), LLVM::LLVMPointerType::get(builder.getContext()),
      symbolName);

  return addressOfOp;
}

Value createPointerFromIndex(OpBuilder& builder, const Location loc,
                             const int64_t index) {
  auto constantOp =
      LLVM::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(index));
  auto intToPtrOp = LLVM::IntToPtrOp::create(
      builder, loc, LLVM::LLVMPointerType::get(builder.getContext()),
      constantOp.getResult());
  return intToPtrOp.getResult();
}

void emitOutputRecording(OpBuilder& builder, Operation* anchor,
                         ArrayRef<ClassicalRegister> classicalRegisters,
                         const DenseMap<int64_t, StaticResult>& staticResults) {
  if (classicalRegisters.empty() && staticResults.empty()) {
    return;
  }

  auto* ctx = builder.getContext();
  auto i64Type = builder.getI64Type();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);
  auto loc = anchor->getLoc();

  auto resultSig = LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto resultDec = getOrCreateFunctionDeclaration(builder, anchor,
                                                  QIR_RECORD_OUTPUT, resultSig);

  // Classical registers
  for (const auto& reg : classicalRegisters) {
    if (!reg.record) {
      continue;
    }

    auto size = resolveIntVariant(builder, loc, reg.size);
    auto label = createResultLabel(builder, anchor, reg.label).getResult();

    // Adaptive Profile: emit `__quantum__rt__result_array_record_output`
    if (reg.array) {
      auto arraySig =
          LLVM::LLVMFunctionType::get(voidType, {i64Type, ptrType, ptrType});
      auto arrayDec = getOrCreateFunctionDeclaration(
          builder, anchor, QIR_RESULT_ARRAY_RECORD_OUTPUT, arraySig);
      LLVM::CallOp::create(builder, loc, arrayDec,
                           ValueRange{size, reg.array, label});
      continue;
    }

    // Base Profile: emit `__quantum__rt__array_record_output` followed by
    // `__quantum__rt__result_record_output` for each bit
    auto arraySig =
        LLVM::LLVMFunctionType::get(voidType, {builder.getI64Type(), ptrType});
    auto arrayDec = getOrCreateFunctionDeclaration(
        builder, anchor, QIR_ARRAY_RECORD_OUTPUT, arraySig);
    LLVM::CallOp::create(builder, loc, arrayDec, ValueRange{size, label});
    for (const auto& [index, ptr] : llvm::enumerate(reg.results)) {
      auto bitLabel = createResultLabel(builder, anchor,
                                        reg.label + "_" + std::to_string(index))
                          .getResult();
      LLVM::CallOp::create(builder, loc, resultDec, ValueRange{ptr, bitLabel});
    }
  }

  // Static results
  for (const auto& [index, result] : staticResults) {
    if (!result.record) {
      continue;
    }
    auto label = createResultLabel(builder, anchor,
                                   "__unnamed__" + std::to_string(index))
                     .getResult();
    LLVM::CallOp::create(builder, loc, resultDec,
                         ValueRange{result.pointer, label});
  }
}

} // namespace mlir::qir
