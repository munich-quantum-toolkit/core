/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"

#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <string>

using namespace mlir;
using namespace mlir::mqt;

#include "mlir/Dialect/MQT/IR/MQTDialect.cpp.inc"

void MQTDialect::initialize() {}

[[nodiscard]] static LogicalResult
verifyEntryPoint(Operation* operation, const NamedAttribute attribute) {
  if (!isa<UnitAttr>(attribute.getValue())) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' must be a unit attribute";
  }

  auto function = dyn_cast<FunctionOpInterface>(operation);
  auto moduleOp = operation->getParentOfType<ModuleOp>();
  if (!function || !moduleOp ||
      operation->getParentOp() != moduleOp.getOperation() ||
      function.getFunctionBody().empty()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' requires a defined module-level function";
  }

  for (Operation& candidate : moduleOp.getBody()->getOperations()) {
    if (&candidate != operation && isEntryPoint(&candidate)) {
      return operation->emitError()
             << "module must contain at most one program entry point";
    }
  }
  return success();
}

[[nodiscard]] static LogicalResult verifyName(Operation* operation,
                                              const NamedAttribute attribute) {
  const auto name = dyn_cast<StringAttr>(attribute.getValue());
  if (!name) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' must be a string";
  }
  if (name.getValue().empty()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' must not be empty";
  }
  if (name.getValue().contains('\0')) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' must not contain a null character";
  }
  return success();
}

[[nodiscard]] static LogicalResult
verifyParameterGroup(Operation* operation, const Attribute attribute) {
  const auto group = dyn_cast<DictionaryAttr>(attribute);
  const auto identity = group ? group.getAs<StringAttr>("identity") : nullptr;
  const auto groupName = group ? group.getAs<StringAttr>("name") : nullptr;
  const auto groupIndex = group ? group.getAs<IntegerAttr>("index") : nullptr;
  const auto groupSize = group ? group.getAs<IntegerAttr>("size") : nullptr;
  if (!group || group.size() != 4U || !identity || !groupName || !groupIndex ||
      !groupSize) {
    return operation->emitError()
           << "parameter-group metadata must contain exactly identity, "
              "name, index, and size";
  }
  if (identity.getValue().empty() || identity.getValue().contains('\0') ||
      groupName.getValue().contains('\0')) {
    return operation->emitError()
           << "parameter-group string metadata is invalid";
  }
  if (!groupIndex.getType().isInteger(64) ||
      groupIndex.getValue().isNegative() ||
      !groupSize.getType().isInteger(64) || groupSize.getValue().isNegative()) {
    return operation->emitError()
           << "parameter-group index and size must be nonnegative i64 "
              "integers";
  }
  return success();
}

[[nodiscard]] static LogicalResult
verifyInputGroup(FunctionOpInterface function, Operation* operation,
                 const unsigned argIndex, const Attribute attribute) {
  const auto inputName = function.getArgAttrOfType<StringAttr>(
      argIndex, MQTDialect::InputNameAttrHelper::getNameStr());
  if (!inputName) {
    return operation->emitError()
           << "parameter-group metadata on a function argument requires an "
              "input name";
  }
  if (failed(verifyParameterGroup(operation, attribute))) {
    return failure();
  }
  const auto group = cast<DictionaryAttr>(attribute);
  const auto groupName = group.getAs<StringAttr>("name");
  const auto groupIndex = group.getAs<IntegerAttr>("index");
  const auto expectedName =
      groupName.str() + "[" + std::to_string(groupIndex.getInt()) + "]";
  if (inputName.getValue() != expectedName) {
    return operation->emitError()
           << "parameter input name must match its group name and index";
  }
  return success();
}

[[nodiscard]] static bool isRegisterAllocation(Operation* operation) {
  if (isa<cbit::AllocOp>(operation)) {
    return true;
  }
  if (auto alloc = dyn_cast<memref::AllocOp>(operation)) {
    const auto type = alloc.getType();
    return type.getRank() == 1 && (isa<qc::QubitType>(type.getElementType()) ||
                                   type.getElementType().isInteger(1));
  }
  if (auto alloc = dyn_cast<qtensor::AllocOp>(operation)) {
    const auto type = cast<RankedTensorType>(alloc.getType());
    return type.getRank() == 1 && isa<qco::QubitType>(type.getElementType());
  }
  return false;
}

[[nodiscard]] static LogicalResult
verifyRegisterName(Operation* operation, const NamedAttribute attribute) {
  if (failed(verifyName(operation, attribute))) {
    return failure();
  }
  if (!isRegisterAllocation(operation)) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' requires a rank-one quantum or classical register allocation";
  }

  auto function = operation->getParentOfType<FunctionOpInterface>();
  if (!function || function.getFunctionBody().empty() ||
      operation->getBlock() != &function.getFunctionBody().front()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' requires an allocation in a function entry block";
  }

  const auto name = cast<StringAttr>(attribute.getValue());
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (function.getArgAttrOfType<StringAttr>(
            index, MQTDialect::InputNameAttrHelper::getNameStr()) == name) {
      return operation->emitError()
             << "duplicate program name '" << name.getValue() << "'";
    }
  }
  for (Operation& candidate : function.getFunctionBody().front()) {
    if (&candidate == operation) {
      continue;
    }
    if (candidate.getAttrOfType<StringAttr>(
            MQTDialect::RegisterNameAttrHelper::getNameStr()) == name) {
      return operation->emitError()
             << "duplicate program name '" << name.getValue() << "'";
    }
  }
  return success();
}

LogicalResult
MQTDialect::verifyOperationAttribute(Operation* operation,
                                     const NamedAttribute attribute) {
  if (attribute.getName() == EntryPointAttrHelper::getNameStr()) {
    return verifyEntryPoint(operation, attribute);
  }
  if (attribute.getName() == RegisterNameAttrHelper::getNameStr()) {
    return verifyRegisterName(operation, attribute);
  }
  if (attribute.getName() == ParameterGroupAttrHelper::getNameStr()) {
    if (!isa<scf::ForOp>(operation)) {
      return operation->emitError()
             << "attribute '" << attribute.getName().getValue()
             << "' is only valid on scf.for";
    }
    return verifyParameterGroup(operation, attribute.getValue());
  }
  if (attribute.getName() == InputNameAttrHelper::getNameStr()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' is only valid on a function argument";
  }
  return operation->emitError()
         << "unknown MQT attribute '" << attribute.getName().getValue() << "'";
}

LogicalResult MQTDialect::verifyRegionArgAttribute(
    Operation* operation, const unsigned regionIndex, const unsigned argIndex,
    const NamedAttribute attribute) {
  const auto attributeName = attribute.getName();
  if (attributeName != InputNameAttrHelper::getNameStr() &&
      attributeName != ParameterGroupAttrHelper::getNameStr()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' is not valid on a region argument";
  }

  auto function = dyn_cast<FunctionOpInterface>(operation);
  if (!function || regionIndex != 0) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' requires a function entry-block argument";
  }

  if (attributeName == ParameterGroupAttrHelper::getNameStr()) {
    return verifyInputGroup(function, operation, argIndex,
                            attribute.getValue());
  }
  if (failed(verifyName(operation, attribute))) {
    return failure();
  }

  const auto name = cast<StringAttr>(attribute.getValue());
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (index == argIndex) {
      continue;
    }
    if (function.getArgAttrOfType<StringAttr>(index, attribute.getName()) ==
        name) {
      return operation->emitError()
             << "duplicate program name '" << name.getValue() << "'";
    }
  }
  if (!function.getFunctionBody().empty()) {
    for (Operation& candidate : function.getFunctionBody().front()) {
      if (candidate.getAttrOfType<StringAttr>(
              RegisterNameAttrHelper::getNameStr()) == name) {
        return operation->emitError()
               << "duplicate program name '" << name.getValue() << "'";
      }
    }
  }
  return success();
}

LogicalResult MQTDialect::verifyRegionResultAttribute(
    Operation* operation, unsigned /*regionIndex*/, unsigned /*resultIndex*/,
    const NamedAttribute attribute) {
  return operation->emitError()
         << "attribute '" << attribute.getName().getValue()
         << "' is not valid on a region result";
}

bool mlir::mqt::isEntryPoint(Operation* operation) {
  return operation != nullptr &&
         operation->hasAttr(MQTDialect::EntryPointAttrHelper::getNameStr());
}

void mlir::mqt::setEntryPoint(Operation* operation) {
  operation->setAttr(MQTDialect::EntryPointAttrHelper::getNameStr(),
                     UnitAttr::get(operation->getContext()));
}

void mlir::mqt::removeEntryPoint(Operation* operation) {
  operation->removeAttr(MQTDialect::EntryPointAttrHelper::getNameStr());
}

func::FuncOp mlir::mqt::getEntryPoint(ModuleOp moduleOp) {
  for (auto function : moduleOp.getOps<func::FuncOp>()) {
    if (isEntryPoint(function)) {
      return function;
    }
  }
  return nullptr;
}
