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
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>

namespace mlir::qir {

void emitQISCall(OpBuilder& builder, Operation* anchor, const Location loc,
                 const ValueRange parameters, const ValueRange controls,
                 const ValueRange targets, const StringRef fnName) {
  const auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  const auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  const auto isGenericControlled =
      fnName.ends_with("__ctl") || fnName.ends_with("__ctladj");

  if (!isGenericControlled) {
    SmallVector<Value> operands;
    operands.reserve(parameters.size() + controls.size() + targets.size());
    operands.append(parameters.begin(), parameters.end());
    operands.append(controls.begin(), controls.end());
    operands.append(targets.begin(), targets.end());

    SmallVector<Type> argumentTypes;
    argumentTypes.reserve(operands.size());
    llvm::transform(operands, std::back_inserter(argumentTypes),
                    [](const Value value) { return value.getType(); });
    const auto signature = LLVM::LLVMFunctionType::get(voidType, argumentTypes);
    const auto declaration =
        getOrCreateFunctionDeclaration(builder, anchor, fnName, signature);
    LLVM::CallOp::create(builder, loc, declaration, operands);
    return;
  }
  assert(!controls.empty() &&
         "generic controlled specialization requires controls");

  const auto i32Type = builder.getI32Type();
  const auto i64Type = builder.getI64Type();
  const auto layout = DataLayout::closest(anchor);
  const auto pointerSize = layout.getTypeSize(ptrType);
  assert(!pointerSize.isScalable() && "pointer size must be fixed");
  const auto pointerBytes = pointerSize.getFixedValue();
  assert(pointerBytes <= std::numeric_limits<std::uint32_t>::max());

  const auto arrayCreateType =
      LLVM::LLVMFunctionType::get(ptrType, {i32Type, i64Type});
  const auto arrayCreate = getOrCreateFunctionDeclaration(
      builder, anchor, QIR_ARRAY_CREATE, arrayCreateType);
  const auto elementSize =
      LLVM::ConstantOp::create(
          builder, loc,
          builder.getI32IntegerAttr(static_cast<std::int32_t>(pointerBytes)))
          .getResult();
  const auto controlCount =
      LLVM::ConstantOp::create(
          builder, loc,
          builder.getI64IntegerAttr(static_cast<std::int64_t>(controls.size())))
          .getResult();
  const auto controlArray =
      LLVM::CallOp::create(builder, loc, arrayCreate,
                           ValueRange{elementSize, controlCount})
          .getResult();

  const auto arrayElementType =
      LLVM::LLVMFunctionType::get(ptrType, {ptrType, i64Type});
  const auto arrayElement = getOrCreateFunctionDeclaration(
      builder, anchor, QIR_ARRAY_ELEMENT, arrayElementType);
  for (const auto& [index, control] : llvm::enumerate(controls)) {
    const auto indexValue =
        LLVM::ConstantOp::create(
            builder, loc,
            builder.getI64IntegerAttr(static_cast<std::int64_t>(index)))
            .getResult();
    const auto element =
        LLVM::CallOp::create(builder, loc, arrayElement,
                             ValueRange{controlArray, indexValue})
            .getResult();
    LLVM::StoreOp::create(builder, loc, control, element);
  }

  const bool usesTuple = !parameters.empty() || targets.size() != 1;
  Value gateArgs;
  if (!usesTuple) {
    gateArgs = targets.front();
  } else {
    SmallVector<Value> payload;
    payload.reserve(parameters.size() + targets.size());
    payload.append(parameters.begin(), parameters.end());
    payload.append(targets.begin(), targets.end());

    SmallVector<Type> payloadTypes;
    payloadTypes.reserve(payload.size());
    llvm::transform(payload, std::back_inserter(payloadTypes),
                    [](const Value value) { return value.getType(); });
    const auto tupleType =
        LLVM::LLVMStructType::getLiteral(builder.getContext(), payloadTypes);
    const auto tupleSize = layout.getTypeSize(tupleType);
    assert(!tupleSize.isScalable() && "QIR tuple size must be fixed");

    const auto tupleCreateType = LLVM::LLVMFunctionType::get(ptrType, i64Type);
    const auto tupleCreate = getOrCreateFunctionDeclaration(
        builder, anchor, QIR_TUPLE_CREATE, tupleCreateType);
    const auto sizeValue =
        LLVM::ConstantOp::create(
            builder, loc,
            builder.getI64IntegerAttr(
                static_cast<std::int64_t>(tupleSize.getFixedValue())))
            .getResult();
    gateArgs =
        LLVM::CallOp::create(builder, loc, tupleCreate, sizeValue).getResult();

    for (const auto& [index, value] : llvm::enumerate(payload)) {
      const SmallVector<LLVM::GEPArg> indices{0,
                                              static_cast<std::int32_t>(index)};
      const auto element = LLVM::GEPOp::create(builder, loc, ptrType, tupleType,
                                               gateArgs, indices)
                               .getResult();
      LLVM::StoreOp::create(builder, loc, value, element);
    }
  }

  const auto controlledType =
      LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  const auto controlled =
      getOrCreateFunctionDeclaration(builder, anchor, fnName, controlledType);
  LLVM::CallOp::create(builder, loc, controlled,
                       ValueRange{controlArray, gateArgs});

  const auto releaseType =
      LLVM::LLVMFunctionType::get(voidType, {ptrType, i32Type});
  const auto decrement =
      LLVM::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(-1))
          .getResult();
  if (usesTuple) {
    const auto tupleRelease = getOrCreateFunctionDeclaration(
        builder, anchor, QIR_TUPLE_RELEASE, releaseType);
    LLVM::CallOp::create(builder, loc, tupleRelease,
                         ValueRange{gateArgs, decrement});
  }
  const auto arrayRelease = getOrCreateFunctionDeclaration(
      builder, anchor, QIR_ARRAY_RELEASE, releaseType);
  LLVM::CallOp::create(builder, loc, arrayRelease,
                       ValueRange{controlArray, decrement});
}

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
