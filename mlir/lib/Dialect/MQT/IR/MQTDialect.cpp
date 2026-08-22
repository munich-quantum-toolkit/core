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
#include "mlir/Dialect/MQT/IR/MQTAttributes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::mqt;

#include "mlir/Dialect/MQT/IR/MQTDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQT/IR/MQTAttributes.cpp.inc"

void MQTDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/MQT/IR/MQTAttributes.cpp.inc"
      >();
}

TargetEnvAttr TargetEnvAttr::get(
    MLIRContext* const context, const StringRef id, const StringRef version,
    const StringRef profile, const StringRef encoding,
    const ArrayRef<std::pair<StringRef, uint64_t>> capabilities,
    const bool optionalFeaturesKnown, const DictionaryAttr target) {
  SmallVector<Attribute> capabilityAttrs;
  capabilityAttrs.reserve(capabilities.size());
  for (const auto& [capability, value] : capabilities) {
    capabilityAttrs.emplace_back(DictionaryAttr::get(
        context, {NamedAttribute(StringAttr::get(context, "id"),
                                 StringAttr::get(context, capability)),
                  NamedAttribute(StringAttr::get(context, "value"),
                                 IntegerAttr::get(IntegerType::get(context, 64),
                                                  value))}));
  }
  return Base::get(
      context, StringAttr::get(context, id), StringAttr::get(context, version),
      StringAttr::get(context, profile), StringAttr::get(context, encoding),
      ArrayAttr::get(context, capabilityAttrs), optionalFeaturesKnown, target);
}

LogicalResult TargetEnvAttr::verify(
    const function_ref<InFlightDiagnostic()> emitError, const StringAttr id,
    const StringAttr version, const StringAttr /*profile*/,
    const StringAttr encoding, const ArrayAttr capabilities,
    const bool /*optionalFeaturesKnown*/, const DictionaryAttr target) {
  if (id.getValue().empty() || version.getValue().empty()) {
    return emitError()
           << "target environment requires a payload ID and version";
  }
  if (encoding.getValue() != "text" && encoding.getValue() != "binary") {
    return emitError() << "target environment encoding must be text or binary";
  }
  if (target.empty()) {
    return emitError() << "target environment requires a target snapshot";
  }
  llvm::SmallDenseSet<std::pair<StringRef, uint64_t>> seen;
  seen.reserve(capabilities.size());
  for (const Attribute attribute : capabilities) {
    const auto capability = dyn_cast<DictionaryAttr>(attribute);
    const auto capabilityId =
        capability ? capability.getAs<StringAttr>("id") : StringAttr{};
    const auto value =
        capability ? capability.getAs<IntegerAttr>("value") : IntegerAttr{};
    if (!capabilityId || !value || !value.getType().isSignlessInteger(64)) {
      return emitError() << "target environment capabilities require string id "
                            "and i64 value";
    }
    if (capabilityId.getValue().empty()) {
      return emitError()
             << "target environment capability ID must not be empty";
    }
    const auto key =
        std::pair(capabilityId.getValue(), value.getValue().getZExtValue());
    if (!seen.insert(key).second) {
      return emitError() << "target environment contains duplicate capability '"
                         << capabilityId.getValue() << "'";
    }
  }
  return success();
}

bool TargetEnvAttr::supports(const StringRef feature,
                             const uint64_t value) const {
  return llvm::any_of(getCapabilities(), [&](const Attribute candidate) {
    const auto capability = dyn_cast<DictionaryAttr>(candidate);
    const auto id =
        capability ? capability.getAs<StringAttr>("id") : StringAttr{};
    const auto candidateValue =
        capability ? capability.getAs<IntegerAttr>("value") : IntegerAttr{};
    return id && candidateValue && id.getValue() == feature &&
           candidateValue.getValue().getZExtValue() == value;
  });
}

FailureOr<Attribute> TargetEnvAttr::query(const DataLayoutEntryKey key) {
  const auto stringKey = key.dyn_cast<StringAttr>();
  if (!stringKey) {
    return failure();
  }
  if (stringKey.getValue() == "mqt.target_env.id") {
    return getId();
  }
  if (stringKey.getValue() == "mqt.target_env.version") {
    return getVersion();
  }
  if (stringKey.getValue() == "mqt.target_env.profile") {
    return getProfile();
  }
  if (stringKey.getValue() == "mqt.target_env.encoding") {
    return getEncoding();
  }
  if (stringKey.getValue() == "mqt.target_env.capabilities") {
    return getCapabilities();
  }
  if (stringKey.getValue() == "mqt.target_env.optional_features_known") {
    return BoolAttr::get(getContext(), getOptionalFeaturesKnown());
  }
  if (stringKey.getValue() == "mqt.target_env.target") {
    return getTarget();
  }
  return failure();
}

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
  if (attribute.getName() == TargetEnvAttr::getOperationAttributeName()) {
    if (!isa<ModuleOp>(operation)) {
      return operation->emitError()
             << "attribute '" << attribute.getName().getValue()
             << "' is only valid on a module";
    }
    if (!isa<TargetEnvAttr>(attribute.getValue())) {
      return operation->emitError()
             << "attribute '" << attribute.getName().getValue()
             << "' must be an mqt target environment";
    }
    return success();
  }
  if (attribute.getName() == EntryPointAttrHelper::getNameStr()) {
    return verifyEntryPoint(operation, attribute);
  }
  if (attribute.getName() == RegisterNameAttrHelper::getNameStr()) {
    return verifyRegisterName(operation, attribute);
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
  if (attribute.getName() != InputNameAttrHelper::getNameStr()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' is not valid on a region argument";
  }
  if (failed(verifyName(operation, attribute))) {
    return failure();
  }

  auto function = dyn_cast<FunctionOpInterface>(operation);
  if (!function || regionIndex != 0) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' requires a function entry-block argument";
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
