/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mqt/Dialect/QIR/QIRDefinitions.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

namespace mlir::qir {

void normalizeQIRModuleFlags(llvm::Module& moduleOp) {
  for (const StringRef key : {
           "dynamic_qubit_management",
           "dynamic_result_management",
           "arrays",
           "ir_functions",
           "multiple_target_branching",
           "multiple_return_points",
           "backwards_branching",
       }) {
    if (auto* flag = moduleOp.getModuleFlag(key)) {
      const auto* integer = llvm::mdconst::dyn_extract<llvm::ConstantInt>(flag);
      const uint32_t bitWidth = key == "backwards_branching" ? 2 : 1;
      moduleOp.setModuleFlag(
          llvm::Module::Error, key,
          llvm::ConstantInt::get(
              llvm::IntegerType::get(moduleOp.getContext(), bitWidth),
              integer != nullptr ? integer->getZExtValue() : 0));
    }
  }
}

LLVM::CallOp emitQISCall(OpBuilder& builder, Operation* anchor,
                         const Location loc, ValueRange parameters,
                         ValueRange controls, ValueRange targets,
                         const StringRef fnName) {
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

    const auto signature = LLVM::LLVMFunctionType::get(
        voidType, llvm::to_vector(ValueRange(operands).getTypes()));
    auto declaration =
        getOrCreateFunctionDeclaration(builder, anchor, fnName, signature);
    return LLVM::CallOp::create(builder, loc, declaration, operands);
  }
  assert(!controls.empty() &&
         "generic controlled specialization requires controls");

  const auto i32Type = builder.getI32Type();
  const auto i64Type = builder.getI64Type();
  const auto layout = DataLayout::closest(anchor);
  const auto pointerSize = layout.getTypeSize(ptrType);
  assert(!pointerSize.isScalable() && "pointer size must be fixed");
  const auto pointerBytes = pointerSize.getFixedValue();
  assert(pointerBytes <= std::numeric_limits<uint32_t>::max());

  const auto arrayCreateType =
      LLVM::LLVMFunctionType::get(ptrType, {i32Type, i64Type});
  auto arrayCreate = getOrCreateFunctionDeclaration(
      builder, anchor, QIR_ARRAY_CREATE, arrayCreateType);
  auto elementSize =
      LLVM::ConstantOp::create(
          builder, loc,
          builder.getI32IntegerAttr(static_cast<int32_t>(pointerBytes)))
          .getResult();
  auto controlCount =
      LLVM::ConstantOp::create(
          builder, loc,
          builder.getI64IntegerAttr(static_cast<int64_t>(controls.size())))
          .getResult();
  auto controlArray =
      LLVM::CallOp::create(builder, loc, arrayCreate,
                           ValueRange{elementSize, controlCount})
          .getResult();

  const auto arrayElementType =
      LLVM::LLVMFunctionType::get(ptrType, {ptrType, i64Type});
  auto arrayElement = getOrCreateFunctionDeclaration(
      builder, anchor, QIR_ARRAY_ELEMENT, arrayElementType);
  for (const auto& [index, control] : llvm::enumerate(controls)) {
    auto indexValue = LLVM::ConstantOp::create(builder, loc,
                                               builder.getI64IntegerAttr(
                                                   static_cast<int64_t>(index)))
                          .getResult();
    auto element = LLVM::CallOp::create(builder, loc, arrayElement,
                                        ValueRange{controlArray, indexValue})
                       .getResult();
    auto store = LLVM::StoreOp::create(builder, loc, control, element);
    store->setAttr(QIR_QUBIT_STORE_ATTR, builder.getUnitAttr());
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

    const auto tupleType = LLVM::LLVMStructType::getLiteral(
        builder.getContext(), llvm::to_vector(ValueRange(payload).getTypes()));
    const auto tupleSize = layout.getTypeSize(tupleType);
    assert(!tupleSize.isScalable() && "QIR tuple size must be fixed");

    const auto tupleCreateType = LLVM::LLVMFunctionType::get(ptrType, i64Type);
    auto tupleCreate = getOrCreateFunctionDeclaration(
        builder, anchor, QIR_TUPLE_CREATE, tupleCreateType);
    auto sizeValue =
        LLVM::ConstantOp::create(builder, loc,
                                 builder.getI64IntegerAttr(static_cast<int64_t>(
                                     tupleSize.getFixedValue())))
            .getResult();
    gateArgs =
        LLVM::CallOp::create(builder, loc, tupleCreate, sizeValue).getResult();

    for (const auto& [index, value] : llvm::enumerate(payload)) {
      const SmallVector<LLVM::GEPArg> indices{
          0,
          static_cast<int32_t>(index),
      };
      auto element = LLVM::GEPOp::create(builder, loc, ptrType, tupleType,
                                         gateArgs, indices)
                         .getResult();
      auto store = LLVM::StoreOp::create(builder, loc, value, element);
      if (index >= parameters.size()) {
        store->setAttr(QIR_QUBIT_STORE_ATTR, builder.getUnitAttr());
      }
    }
  }

  const auto controlledType =
      LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType});
  auto controlled =
      getOrCreateFunctionDeclaration(builder, anchor, fnName, controlledType);
  auto call = LLVM::CallOp::create(builder, loc, controlled,
                                   ValueRange{controlArray, gateArgs});

  const auto releaseType =
      LLVM::LLVMFunctionType::get(voidType, {ptrType, i32Type});
  auto decrement =
      LLVM::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(-1))
          .getResult();
  if (usesTuple) {
    auto tupleRelease = getOrCreateFunctionDeclaration(
        builder, anchor, QIR_TUPLE_RELEASE, releaseType);
    LLVM::CallOp::create(builder, loc, tupleRelease,
                         ValueRange{gateArgs, decrement});
  }
  auto arrayRelease = getOrCreateFunctionDeclaration(
      builder, anchor, QIR_ARRAY_RELEASE, releaseType);
  LLVM::CallOp::create(builder, loc, arrayRelease,
                       ValueRange{controlArray, decrement});
  return call;
}

LLVM::LLVMFuncOp getMainFunction(Operation* op) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }
  if (!moduleOp) {
    return nullptr;
  }

  LLVM::LLVMFuncOp main;
  const auto entryPoint =
      StringAttr::get(op->getContext(), ::qir::ENTRY_POINT_ATTR);
  for (auto function : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    const auto passthrough = function.getPassthroughAttr();
    if (!passthrough || !llvm::is_contained(passthrough, entryPoint)) {
      continue;
    }
    if (main) {
      return nullptr;
    }
    main = function;
  }
  return main;
}

LLVM::LLVMFuncOp getOrCreateFunctionDeclaration(OpBuilder& builder,
                                                Operation* op, StringRef fnName,
                                                Type fnType) {
  // Check if the function already exists
  auto* fnDecl =
      SymbolTable::lookupNearestSymbolFrom(op, builder.getStringAttr(fnName));

  if (fnDecl == nullptr) {

    const OpBuilder::InsertionGuard guard(builder);

    // Create the declaration at the end of the module
    auto moduleOp = dyn_cast<ModuleOp>(op);
    if (!moduleOp) {
      moduleOp = op->getParentOfType<ModuleOp>();
    }
    if (!moduleOp) {
      llvm::reportFatalInternalError("Module not found");
    }
    builder.setInsertionPointToEnd(moduleOp.getBody());

    fnDecl = LLVM::LLVMFuncOp::create(builder, op->getLoc(), fnName, fnType);
  }

  auto function = cast<LLVM::LLVMFuncOp>(fnDecl);
  if (fnName == QIR_MEASURE) {
    function.setArgAttr(1, LLVM::LLVMDialect::getWriteOnlyAttrName(),
                        builder.getUnitAttr());
  } else if (fnName == QIR_READ_RESULT) {
    function.setArgAttr(0, LLVM::LLVMDialect::getReadonlyAttrName(),
                        builder.getUnitAttr());
  }
  if (fnName != QIR_MEASURE && fnName != QIR_RESET) {
    return function;
  }

  const auto irreversible = builder.getStringAttr(::qir::IRREVERSIBLE_ATTR);
  const auto passthrough = function->getAttrOfType<ArrayAttr>("passthrough");
  if (passthrough && llvm::is_contained(passthrough, irreversible)) {
    return function;
  }

  SmallVector<Attribute> entries;
  if (passthrough) {
    entries.append(passthrough.begin(), passthrough.end());
  }
  entries.push_back(irreversible);
  function->setAttr("passthrough", builder.getArrayAttr(entries));
  return function;
}

static LLVM::AddressOfOp createResultLabel(OpBuilder& builder, Operation* op,
                                           StringRef label,
                                           StringRef symbolPrefix,
                                           SymbolTable& symbols,
                                           LLVM::LLVMFuncOp main) {
  const OpBuilder::InsertionGuard guard(builder);
  auto moduleOp = cast<ModuleOp>(symbols.getOp());

  auto symbolName = builder.getStringAttr((symbolPrefix + "_" + label).str());

  auto global = symbols.lookup<LLVM::GlobalOp>(symbolName);
  auto initializer = builder.getStringAttr(label.str() + '\0');
  if (!global || !global.getConstant() || global.getAddrSpace() != 0 ||
      global.getValueAttr() != initializer) {
    const auto llvmArrayType =
        LLVM::LLVMArrayType::get(builder.getIntegerType(8), label.size() + 1);

    // Create the declaration at the start of the module
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto globalOp = LLVM::GlobalOp::create(
        builder, op->getLoc(), llvmArrayType, /*isConstant=*/true,
        LLVM::Linkage::Internal, symbolName, initializer);
    globalOp->setAttr("addr_space", builder.getI32IntegerAttr(0));
    globalOp->setAttr("dso_local", builder.getUnitAttr());
    symbolName = symbols.insert(globalOp);
  }

  // Create AddressOfOp
  // Shall be added to the first block of the `main` function in the module
  if (!main) {
    llvm::reportFatalInternalError("Main function not found");
  }
  auto& firstBlock = *main.getBlocks().begin();
  builder.setInsertionPointToStart(&firstBlock);

  auto addressOfOp = LLVM::AddressOfOp::create(
      builder, op->getLoc(), LLVM::LLVMPointerType::get(builder.getContext()),
      symbolName);

  return addressOfOp;
}

LLVM::AddressOfOp createResultLabel(OpBuilder& builder, Operation* op,
                                    StringRef label, StringRef symbolPrefix) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }
  if (!moduleOp) {
    llvm::reportFatalInternalError("Module not found");
  }
  SymbolTable symbols(moduleOp);
  return createResultLabel(builder, op, label, symbolPrefix, symbols,
                           getMainFunction(op));
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
                         ArrayRef<OutputRecord> outputs) {
  if (outputs.empty()) {
    return;
  }

  auto* ctx = builder.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);
  auto loc = anchor->getLoc();

  auto main = getMainFunction(anchor);
  if (!main) {
    llvm::reportFatalInternalError("Main function not found");
  }
  SymbolTable symbols(main->getParentOfType<ModuleOp>());
  llvm::StringSet<> outputLabels;
  for (const auto& output : outputs) {
    outputLabels.insert(output.label);
  }
  llvm::StringSet<> labels;
  size_t suffix = 0;
  const auto createLabel = [&](StringRef base, bool topLevel = false) {
    auto label = base.str();
    // Generated child labels must not consume original register names.
    while ((!topLevel && outputLabels.contains(label)) ||
           !labels.insert(label).second) {
      label = base.str() + "_" + std::to_string(++suffix);
      topLevel = false;
    }
    return createResultLabel(builder, anchor, label, "qir.result_label",
                             symbols, main)
        .getResult();
  };

  Operation* firstRecord = nullptr;
  const auto record = [&](LLVM::LLVMFuncOp& declaration, StringRef name,
                          ValueRange operands) {
    if (!declaration) {
      const auto signature = LLVM::LLVMFunctionType::get(
          voidType, llvm::to_vector(operands.getTypes()));
      declaration =
          getOrCreateFunctionDeclaration(builder, anchor, name, signature);
    }
    auto call = LLVM::CallOp::create(builder, loc, declaration, operands);
    if (!firstRecord) {
      firstRecord = call;
    }
  };
  LLVM::LLVMFuncOp resultDec;
  LLVM::LLVMFuncOp baseArrayDec;
  LLVM::LLVMFuncOp adaptiveArrayDec;
  LLVM::LLVMFuncOp booleanDec;
  for (const auto& output : outputs) {
    auto label = createLabel(output.label, true);
    if (const auto* result = std::get_if<Value>(&output.value)) {
      if (result->getType().isInteger(1)) {
        record(booleanDec, QIR_BOOL_RECORD_OUTPUT, {*result, label});
      } else {
        record(resultDec, QIR_RECORD_OUTPUT, {*result, label});
      }
      continue;
    }
    const auto& reg = *std::get<const ClassicalRegister*>(output.value);
    auto size = resolveIntVariant(builder, loc, reg.size);

    // Adaptive Profile: emit `__quantum__rt__result_array_record_output`
    if (reg.array && !reg.booleanStorage) {
      record(adaptiveArrayDec, QIR_RESULT_ARRAY_RECORD_OUTPUT,
             {size, reg.array, label});
      continue;
    }

    // Record the array header, then each Result or Boolean element.
    record(baseArrayDec, QIR_ARRAY_RECORD_OUTPUT, {size, label});
    if (reg.booleanStorage) {
      for (int64_t index = 0; index < std::get<int64_t>(reg.size); ++index) {
        Value bit;
        {
          // QIR requires output recording to follow all computation.
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(firstRecord);
          auto position = LLVM::ConstantOp::create(
              builder, loc, builder.getI64IntegerAttr(index));
          auto address =
              LLVM::GEPOp::create(builder, loc, ptrType, builder.getI1Type(),
                                  reg.array, ValueRange{position});
          bit =
              LLVM::LoadOp::create(builder, loc, builder.getI1Type(), address);
        }
        auto bitLabel = createLabel(output.label + "_" + std::to_string(index));
        record(booleanDec, QIR_BOOL_RECORD_OUTPUT, {bit, bitLabel});
      }
      continue;
    }
    for (auto [index, ptr] : llvm::enumerate(reg.results)) {
      auto bitLabel = createLabel(output.label + "_" + std::to_string(index));
      record(resultDec, QIR_RECORD_OUTPUT, {ptr, bitLabel});
    }
  }
}

} // namespace mlir::qir
