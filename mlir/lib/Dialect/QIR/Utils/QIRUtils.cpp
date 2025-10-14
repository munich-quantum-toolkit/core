/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <string>

namespace mlir::qir {

LLVM::LLVMFuncOp getMainFunction(Operation* op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<ModuleOp>();
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

void setQIRAttributes(LLVM::LLVMFuncOp& main, const QIRMetadata& metadata) {
  OpBuilder builder(main.getBody());
  SmallVector<Attribute> attributes;

  // Core QIR attributes
  attributes.emplace_back(builder.getStringAttr("entry_point"));
  attributes.emplace_back(
      builder.getStrArrayAttr({"output_labeling_schema", "labeled"}));
  attributes.emplace_back(
      builder.getStrArrayAttr({"qir_profiles", "base_profile"}));

  // Resource requirements
  attributes.emplace_back(builder.getStrArrayAttr(
      {"required_num_qubits", std::to_string(metadata.numQubits)}));
  attributes.emplace_back(builder.getStrArrayAttr(
      {"required_num_results", std::to_string(metadata.numResults)}));

  // QIR version (Base Profile spec requires version 2.0)
  attributes.emplace_back(builder.getStrArrayAttr({"qir_major_version", "2"}));
  attributes.emplace_back(builder.getStrArrayAttr({"qir_minor_version", "0"}));

  // Management model
  attributes.emplace_back(
      builder.getStrArrayAttr({"dynamic_qubit_management",
                               metadata.useDynamicQubit ? "true" : "false"}));
  attributes.emplace_back(
      builder.getStrArrayAttr({"dynamic_result_management",
                               metadata.useDynamicResult ? "true" : "false"}));

  main->setAttr("passthrough", builder.getArrayAttr(attributes));
}

LLVM::LLVMFuncOp getOrCreateFunctionDeclaration(OpBuilder& builder,
                                                Operation* op, StringRef fnName,
                                                Type fnType) {
  // Check if the function already exists
  auto* fnDecl =
      SymbolTable::lookupNearestSymbolFrom(op, builder.getStringAttr(fnName));

  if (fnDecl == nullptr) {
    // Save current insertion point
    const OpBuilder::InsertionGuard insertGuard(builder);

    // Create the declaration at the end of the module
    if (auto module = dyn_cast<ModuleOp>(op)) {
      builder.setInsertionPointToEnd(module.getBody());
    } else {
      module = op->getParentOfType<ModuleOp>();
      builder.setInsertionPointToEnd(module.getBody());
    }

    fnDecl = builder.create<LLVM::LLVMFuncOp>(op->getLoc(), fnName, fnType);

    // Add irreversible attribute to irreversible quantum operations
    if (fnName == QIR_MEASURE || fnName == QIR_QUBIT_RELEASE ||
        fnName == QIR_RESET) {
      fnDecl->setAttr("passthrough", builder.getStrArrayAttr({"irreversible"}));
    }
  }

  return cast<LLVM::LLVMFuncOp>(fnDecl);
}

LLVM::AddressOfOp createResultLabel(OpBuilder& builder, Operation* op,
                                    const StringRef label,
                                    const StringRef symbolPrefix) {
  // Save current insertion point
  const OpBuilder::InsertionGuard insertGuard(builder);

  // Create the declaration at the start of the module
  if (auto module = dyn_cast<ModuleOp>(op)) {
    builder.setInsertionPointToStart(module.getBody());
  } else {
    module = op->getParentOfType<ModuleOp>();
    builder.setInsertionPointToStart(module.getBody());
  }

  const auto symbolName =
      builder.getStringAttr((symbolPrefix + "_" + label).str());
  const auto llvmArrayType = LLVM::LLVMArrayType::get(
      builder.getIntegerType(8), static_cast<unsigned>(label.size() + 1));
  const auto stringInitializer = builder.getStringAttr(label.str() + '\0');

  const auto globalOp = builder.create<LLVM::GlobalOp>(
      op->getLoc(), llvmArrayType, /*isConstant=*/true, LLVM::Linkage::Internal,
      symbolName, stringInitializer);
  globalOp->setAttr("addr_space", builder.getI32IntegerAttr(0));
  globalOp->setAttr("dso_local", builder.getUnitAttr());

  // Create addressOf operation
  // Shall be added to the first block of the `main` function in the module
  auto main = getMainFunction(op);
  auto& firstBlock = *(main.getBlocks().begin());
  builder.setInsertionPointToStart(&firstBlock);

  const auto addressOfOp = builder.create<LLVM::AddressOfOp>(
      op->getLoc(), LLVM::LLVMPointerType::get(builder.getContext()),
      symbolName);

  return addressOfOp;
}

Value createPointerFromIndex(OpBuilder& builder, const Location loc,
                             const int64_t index) {
  auto constantOp =
      builder.create<LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(index));
  auto intToPtrOp = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()),
      constantOp.getResult());
  return intToPtrOp.getResult();
}

} // namespace mlir::qir
