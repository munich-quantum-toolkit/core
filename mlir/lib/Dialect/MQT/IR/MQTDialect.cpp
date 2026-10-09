/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTDialect.h"

#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Support/RandomSeed.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::mqt;

#include "mqt/Dialect/MQT/IR/MQTDialect.cpp.inc"
#include "mqt/Dialect/MQT/IR/MQTEnums.cpp.inc"

void MQTDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mqt/Dialect/MQT/IR/MQTAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "mqt/Dialect/MQT/IR/MQTAttributes.cpp.inc"

LogicalResult mlir::mqt::verifyQuantumAllocations(ModuleOp moduleOp) {
  auto entryPoint = getEntryPoint(moduleOp);
  Block* entryBlock = entryPoint && !entryPoint.isExternal()
                          ? &entryPoint.getBody().front()
                          : nullptr;
  bool hasStatic = false;
  bool hasDynamic = false;
  const auto result =
      moduleOp.walk<WalkOrder::PreOrder>([&](Operation* operation) {
        if (isa<ModuleOp>(operation) && operation != moduleOp.getOperation()) {
          return WalkResult::skip();
        }
        bool allocatesQubits =
            isa<qc::AllocOp, qco::AllocOp, qtensor::AllocOp>(operation);
        if (isa<memref::AllocOp>(operation) &&
            operation->getNumResults() == 1) {
          auto type = dyn_cast<MemRefType>(operation->getResult(0).getType());
          allocatesQubits = type && isa<qc::QubitType>(type.getElementType());
        }
        hasDynamic |= allocatesQubits;
        hasStatic |= isa<qc::StaticOp, qco::StaticOp>(operation);
        if (hasDynamic && hasStatic) {
          operation->emitOpError(
              "cannot mix static and dynamic qubit allocation modes");
          return WalkResult::interrupt();
        }
        if (allocatesQubits &&
            (!entryBlock || operation->getBlock() != entryBlock)) {
          operation->emitOpError(
              "dynamic quantum allocations must be in the entry "
              "block of the 'mqt.entry_point' function");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  return success(!result.wasInterrupted());
}

[[nodiscard]] static LogicalResult
verifyEntryPoint(Operation* operation, const NamedAttribute attribute) {
  if (!isa<UnitAttr>(attribute.getValue())) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' must be a unit attribute";
  }

  auto function = dyn_cast<func::FuncOp>(operation);
  auto moduleOp = operation->getParentOfType<ModuleOp>();
  if (!function || !function.isPublic() || function.isExternal() || !moduleOp ||
      operation->getParentOp() != moduleOp.getOperation()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' requires a public, defined module-level function";
  }

  for (Operation& candidate : moduleOp.getBody()->getOperations()) {
    if (&candidate != operation && isEntryPoint(&candidate)) {
      return operation->emitError()
             << "module must contain at most one program entry point";
    }
  }
  return verifyQuantumAllocations(moduleOp);
}

template <typename CallOp>
[[nodiscard]] static LogicalResult
verifyNoUnitaryRecursion(func::FuncOp function) {
  // A cycle of unitary calls passes local body checks without any control flow.
  DenseSet<Operation*> visited;
  SmallVector<func::FuncOp> worklist{function};
  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }
    WalkResult result = current.walk([&](CallOp call) {
      if (failed(verify(call))) {
        return WalkResult::interrupt();
      }
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee) {
        return WalkResult::advance();
      }
      if (callee == function) {
        function.emitError("unitary function must not be recursive");
        return WalkResult::interrupt();
      }
      worklist.emplace_back(callee);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

[[nodiscard]] static LogicalResult verifyQCUnitaryBody(func::FuncOp function) {
  bool valid = true;
  function.walk([&](Operation* nested) {
    if (!valid || nested == function.getOperation()) {
      return;
    }
    if (isa<func::ReturnOp>(nested)) {
      return;
    }
    if (isa<qc::UnitaryOpInterface, qc::YieldOp>(nested) ||
        (isa<scf::YieldOp>(nested) && isa<scf::ForOp>(nested->getParentOp()))) {
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(nested)) {
      valid = loop.getStaticTripCount().has_value();
      return;
    }
    valid =
        nested->getNumRegions() == 0 && isMemoryEffectFree(nested) &&
        llvm::none_of(nested->getOperandTypes(),
                      llvm::IsaPred<qc::QubitType>) &&
        llvm::none_of(nested->getResultTypes(), llvm::IsaPred<qc::QubitType>);
  });
  if (!valid) {
    return function.emitError()
           << "unitary QC function body contains an unsupported operation";
  }

  return verifyNoUnitaryRecursion<qc::CallOp>(function);
}

[[nodiscard]] static LogicalResult
verifyQCOUnitaryQubitFlow(Operation* owner, ValueRange inputs,
                          ValueRange outputs) {
  for (auto [resultIndex, returned] : llvm::enumerate(outputs)) {
    if (!isa<qco::QubitType>(returned.getType())) {
      continue;
    }
    Value current = returned;
    llvm::SmallDenseSet<Value> visited;
    while (auto result = dyn_cast<OpResult>(current)) {
      if (!visited.insert(current).second) {
        return owner->emitError("unitary QCO result has cyclic qubit flow");
      }
      if (auto loop = dyn_cast<scf::ForOp>(result.getOwner())) {
        current = loop.getInitArgs()[result.getResultNumber()];
        continue;
      }
      auto unitary = dyn_cast<qco::UnitaryOpInterface>(result.getOwner());
      if (!unitary) {
        return owner->emitError()
               << "unitary QCO result does not originate from a qubit "
                  "argument";
      }
      current = unitary.getInputForOutput(current);
      if (!current) {
        return owner->emitError()
               << "unitary QCO operation has no input corresponding to its "
                  "returned qubit";
      }
    }
    if (current != inputs[resultIndex]) {
      return owner->emitError()
             << "unitary QCO results must continue qubit arguments "
                "positionally";
    }
  }
  return success();
}

[[nodiscard]] static LogicalResult verifyQCOUnitaryBody(func::FuncOp function,
                                                        unsigned firstQubit) {
  bool valid = true;
  function.walk([&](Operation* nested) {
    if (!valid || nested == function.getOperation()) {
      return;
    }
    if (isa<func::ReturnOp, qco::UnitaryOpInterface, qco::YieldOp>(nested) ||
        (isa<scf::YieldOp>(nested) && isa<scf::ForOp>(nested->getParentOp()))) {
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(nested)) {
      valid = loop.getStaticTripCount().has_value() &&
              succeeded(verifyQCOUnitaryQubitFlow(
                  loop, loop.getRegionIterArgs(),
                  cast<scf::YieldOp>(loop.getBody()->getTerminator())
                      .getOperands()));
      return;
    }
    valid =
        nested->getNumRegions() == 0 && isMemoryEffectFree(nested) &&
        llvm::none_of(nested->getOperandTypes(),
                      llvm::IsaPred<qco::QubitType>) &&
        llvm::none_of(nested->getResultTypes(), llvm::IsaPred<qco::QubitType>);
  });
  if (!valid) {
    return function.emitError()
           << "unitary QCO function body contains an unsupported operation";
  }

  auto returnOp = cast<func::ReturnOp>(function.getBody().front().back());
  if (failed(verifyQCOUnitaryQubitFlow(
          function, function.getArguments().drop_front(firstQubit),
          returnOp.getOperands()))) {
    return failure();
  }
  return verifyNoUnitaryRecursion<qco::CallOp>(function);
}

[[nodiscard]] static LogicalResult
verifyUnitaryFunction(Operation* operation, const NamedAttribute attribute) {
  if (!isa<UnitAttr>(attribute.getValue())) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' must be a unit attribute";
  }

  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function || function.isExternal() || !function.isPrivate() ||
      isEntryPoint(operation) || !function.getBody().hasOneBlock()) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' requires a private, defined, single-block non-entry function";
  }
  if (function.getBody().front().empty()) {
    return operation->emitError("unitary function body must not be empty");
  }

  unsigned firstQubit = function.getNumArguments();
  bool usesQC = false;
  bool usesQCO = false;
  for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
    if (isa<qc::QubitType, qco::QubitType>(type)) {
      if (firstQubit == function.getNumArguments()) {
        firstQubit = index;
      }
      usesQC |= isa<qc::QubitType>(type);
      usesQCO |= isa<qco::QubitType>(type);
      continue;
    }
    if (firstQubit != function.getNumArguments() || !type.isF64()) {
      return operation->emitError()
             << "unitary function arguments must be f64 parameters "
                "followed by scalar qubits";
    }
  }
  if (firstQubit == function.getNumArguments() || usesQC == usesQCO) {
    return operation->emitError()
           << "unitary function requires at least one QC or QCO qubit "
              "argument";
  }

  const auto numQubits = function.getNumArguments() - firstQubit;
  if (usesQC && function.getNumResults() != 0) {
    return operation->emitError()
           << "unitary QC function must not return values";
  }
  if (usesQCO && (function.getNumResults() != numQubits ||
                  llvm::any_of(function.getResultTypes(), [](Type type) {
                    return !isa<qco::QubitType>(type);
                  }))) {
    return operation->emitError()
           << "unitary QCO function must return one qubit per qubit argument";
  }
  auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
  if (!returnOp || (usesQC && returnOp.getNumOperands() != 0)) {
    return operation->emitError(
        usesQC ? "unitary QC function must end in an empty func.return"
               : "unitary QCO function must end in func.return");
  }

  // Attribute verification precedes nested operation verification. Check the
  // body before querying memory effects or qubit correspondence.
  for (Operation& nested : function.getBody().front()) {
    if (failed(verify(&nested))) {
      return failure();
    }
  }
  return usesQC ? verifyQCUnitaryBody(function)
                : verifyQCOUnitaryBody(function, firstQubit);
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
  if (auto storage = dyn_cast<memref::AllocaOp>(operation)) {
    auto type = storage.getType();
    return type.getRank() == 1 && isa<qc::QubitType>(type.getElementType());
  }
  if (isa<qtensor::FromElementsOp>(operation)) {
    return true;
  }
  if (auto alloc = dyn_cast<qtensor::AllocOp>(operation)) {
    const auto type = alloc.getType();
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
  if (attribute.getName() == COMPILATION_SEED_ATTR) {
    auto seed = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!isa<ModuleOp>(operation) || !seed ||
        !seed.getType().isSignlessInteger(64)) {
      return operation->emitError(
          "mqt.compilation_seed requires a signless i64 on a module");
    }
    return success();
  }
  if (attribute.getName() == kSourceQubitCountAttr) {
    auto count = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!isa<ModuleOp>(operation) || !count ||
        !count.getType().isSignlessInteger(64) || count.getInt() < 0) {
      return operation->emitError(
          "source qubit count requires a nonnegative i64 on a module");
    }
    llvm::SmallDenseSet<int64_t> seen;
    const auto result = operation->walk([&](Operation* nested) {
      auto indices =
          nested->getAttrOfType<DenseI64ArrayAttr>(kSourceQubitIndicesAttr);
      if (indices) {
        for (auto index : indices.asArrayRef()) {
          if (index < 0 || index >= count.getInt() ||
              !seen.insert(index).second) {
            nested->emitError("source qubit indices must be distinct within "
                              "the module and smaller than its source count");
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }
  if (attribute.getName() == kSourceQubitIndicesAttr) {
    int64_t width = -1;
    if (isa<qco::AllocOp>(operation)) {
      width = 1;
    } else if (auto tensor = dyn_cast<qtensor::AllocOp>(operation)) {
      width = getConstantIntValue(tensor.getSize()).value_or(-1);
    }
    auto indices = dyn_cast<DenseI64ArrayAttr>(attribute.getValue());
    if (width < 0 || !indices || indices.size() != width) {
      return operation->emitError("source qubit indices require one i64 entry "
                                  "per fixed allocation slot");
    }
    llvm::SmallDenseSet<int64_t> seen;
    for (auto index : indices.asArrayRef()) {
      if (index < 0 || !seen.insert(index).second) {
        return operation->emitError(
            "source qubit indices must be distinct and nonnegative");
      }
    }
    return success();
  }
  if (attribute.getName() == kSourceOutputPermutationAttr) {
    auto permutation = dyn_cast<DenseI64ArrayAttr>(attribute.getValue());
    auto count = operation->getAttrOfType<IntegerAttr>(kSourceQubitCountAttr);
    if (!isa<ModuleOp>(operation) || !permutation || !count ||
        !count.getType().isSignlessInteger(64) || count.getInt() < 0 ||
        permutation.size() != count.getInt()) {
      return operation->emitError(
          "source output permutation requires one i64 entry per source qubit "
          "on a prepared module");
    }
    llvm::SmallDenseSet<int64_t> seen;
    for (auto index : permutation.asArrayRef()) {
      if (index < 0 || index >= permutation.size() ||
          !seen.insert(index).second) {
        return operation->emitError(
            "source output permutation must be a complete permutation");
      }
    }
    return success();
  }
  if (attribute.getName() == "mqt.layout") {
    if (!isa<ModuleOp>(operation)) {
      return operation->emitError("qubit layout belongs on a program module");
    }
    return success(succeeded(QubitLayout::fromAttr(
        attribute.getValue(), [&] { return operation->emitError(); })));
  }
  if (attribute.getName() == TargetEnvAttr::name) {
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
  if (attribute.getName() == UnitaryAttrHelper::getNameStr()) {
    return verifyUnitaryFunction(operation, attribute);
  }
  if (attribute.getName() == RegisterNameAttrHelper::getNameStr()) {
    return verifyRegisterName(operation, attribute);
  }
  if (attribute.getName() == SourceNameAttrHelper::getNameStr()) {
    if (!isa<FunctionOpInterface>(operation)) {
      return operation->emitError()
             << "attribute '" << attribute.getName().getValue()
             << "' is only valid on a function";
    }
    return verifyName(operation, attribute);
  }
  if (attribute.getName() == ParameterGroupAttrHelper::getNameStr()) {
    if (!isa<scf::ForOp>(operation)) {
      return operation->emitError()
             << "attribute '" << attribute.getName().getValue()
             << "' is only valid on scf.for";
    }
    return verifyParameterGroup(operation, attribute.getValue());
  }
  if (attribute.getName() == InputNameAttrHelper::getNameStr() ||
      attribute.getName() == InputIdAttrHelper::getNameStr()) {
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
      attributeName != InputIdAttrHelper::getNameStr() &&
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

  if (attributeName == InputIdAttrHelper::getNameStr()) {
    const auto id = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!id || !id.getType().isSignlessInteger(128)) {
      return operation->emitError("input identity must be an i128 attribute");
    }
    if (!function.getArgAttrOfType<StringAttr>(
            argIndex, InputNameAttrHelper::getNameStr())) {
      return operation->emitError("input identity requires an input name");
    }
    return success();
  }
  if (attributeName == ParameterGroupAttrHelper::getNameStr()) {
    return verifyInputGroup(function, operation, argIndex,
                            attribute.getValue());
  }
  if (failed(verifyName(operation, attribute))) {
    return failure();
  }

  /// The first named input owns cross-argument checks. Register-name
  /// verification owns collisions between inputs and registers.
  for (unsigned index = 0; index < argIndex; ++index) {
    if (function.getArgAttr(index, attributeName)) {
      return success();
    }
  }
  DenseSet<Attribute> names;
  DenseSet<Attribute> identities;
  for (unsigned index = argIndex; index < function.getNumArguments(); ++index) {
    if (auto name = function.getArgAttrOfType<StringAttr>(index, attributeName);
        name && !names.insert(name).second) {
      return operation->emitError()
             << "duplicate program name '" << name.getValue() << "'";
    }
    if (auto id = function.getArgAttr(index, InputIdAttrHelper::getNameStr());
        id && !identities.insert(id).second) {
      return operation->emitError("duplicate input identity");
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

void mlir::mqt::setEntryPoint(Operation* operation) {
  operation->setAttr(MQTDialect::EntryPointAttrHelper::getNameStr(),
                     UnitAttr::get(operation->getContext()));
}

void mlir::mqt::removeEntryPoint(Operation* operation) {
  operation->removeAttr(MQTDialect::EntryPointAttrHelper::getNameStr());
}

void mlir::mqt::setUnitaryFunction(Operation* operation) {
  operation->setAttr(MQTDialect::UnitaryAttrHelper::getNameStr(),
                     UnitAttr::get(operation->getContext()));
}
