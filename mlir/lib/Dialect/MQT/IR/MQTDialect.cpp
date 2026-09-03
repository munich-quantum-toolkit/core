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
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h> // IWYU pragma: keep
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h> // IWYU pragma: keep
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::mqt;

#include "mlir/Dialect/MQT/IR/MQTDialect.cpp.inc"
#include "mlir/Dialect/MQT/IR/MQTEnums.cpp.inc"

void MQTDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/MQT/IR/MQTAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQT/IR/MQTAttributes.cpp.inc"

LogicalResult
DurationUnitAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                         const StringAttr unit, const FloatAttr scaleFactor) {
  if (unit.getValue().trim().empty()) {
    return emitError() << "duration unit must not be empty";
  }
  if (!scaleFactor.getType().isF64()) {
    return emitError() << "duration scale factor must be an f64 value";
  }
  const auto value = scaleFactor.getValueAsDouble();
  if (!std::isfinite(value) || value <= 0.) {
    return emitError() << "duration scale factor must be positive and finite";
  }
  return success();
}

LogicalResult
SiteAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                 const int64_t id, const StringAttr name,
                 const std::optional<uint64_t> t1,
                 const std::optional<uint64_t> t2) {
  if (id < 0) {
    return emitError() << "compiler target site ID must be nonnegative";
  }
  if (name && name.getValue().empty()) {
    return emitError()
           << "compiler target site name must not be empty when present";
  }
  if (t1 == 0 || t2 == 0) {
    return emitError()
           << "compiler target site coherence times must be positive";
  }
  return success();
}

LogicalResult
CouplingAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                     const int64_t source, const int64_t target) {
  if (source < 0 || target < 0) {
    return emitError() << "compiler target coupling sites must be nonnegative";
  }
  if (source == target) {
    return emitError() << "compiler target coupling must join distinct sites";
  }
  return success();
}

[[nodiscard]] static LogicalResult
verifyFidelity(const function_ref<InFlightDiagnostic()>& emitError,
               const FloatAttr fidelity, const StringRef description) {
  if (!fidelity) {
    return success();
  }
  if (!fidelity.getType().isF64()) {
    return emitError() << description << " must be an f64 value";
  }
  const auto value = fidelity.getValueAsDouble();
  if (!std::isfinite(value) || value < 0. || value > 1.) {
    return emitError() << description << " must be finite and in [0, 1]";
  }
  return success();
}

LogicalResult
SiteTupleAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                      const ArrayRef<int64_t> sites,
                      const std::optional<uint64_t> /*duration*/,
                      const FloatAttr fidelity) {
  llvm::SmallDenseSet<int64_t> seen;
  seen.reserve(sites.size());
  for (const int64_t site : sites) {
    if (site < 0) {
      return emitError()
             << "compiler target site tuple contains a negative site ID";
    }
    if (!seen.insert(site).second) {
      return emitError()
             << "compiler target site tuple contains a duplicate site";
    }
  }
  return verifyFidelity(emitError, fidelity,
                        "compiler target site-tuple fidelity");
}

LogicalResult
OperationArityAttr::verify(const function_ref<InFlightDiagnostic()> emitError,
                           const OperationArityKind kind,
                           const uint64_t value) {
  if (kind == OperationArityKind::Variadic && value == 0) {
    return emitError()
           << "compiler target operation variadic minimum must be positive";
  }
  return success();
}

LogicalResult NativeOperationAttr::verify(
    const function_ref<InFlightDiagnostic()> emitError, const StringAttr name,
    const OperationArityAttr arity, const uint64_t /*numParameters*/,
    const ArrayRef<SiteTupleAttr> siteTuples,
    const std::optional<uint64_t> /*duration*/, const FloatAttr fidelity) {
  if (name.getValue().trim().empty()) {
    return emitError() << "compiler target operation name must not be empty";
  }
  if (failed(verifyFidelity(emitError, fidelity,
                            "compiler target operation fidelity"))) {
    return failure();
  }

  if (!siteTuples.empty() && arity.getKind() == OperationArityKind::Variadic) {
    return emitError()
           << "compiler target variadic operation cannot contain site tuples";
  }
  if (!siteTuples.empty() && arity.getValue() == 0) {
    return emitError()
           << "compiler target zero-arity operation cannot contain site tuples";
  }

  SmallVector<ArrayRef<int64_t>> seen;
  seen.reserve(siteTuples.size());
  for (const SiteTupleAttr siteTuple : siteTuples) {
    if (siteTuple.getSites().size() != arity.getValue()) {
      return emitError()
             << "compiler target operation site tuple does not match its arity";
    }
    if (llvm::is_contained(seen, siteTuple.getSites())) {
      return emitError()
             << "compiler target operation contains a duplicate site tuple";
    }
    seen.emplace_back(siteTuple.getSites());
  }
  return success();
}

LogicalResult CompilationTargetAttr::verify(
    const function_ref<InFlightDiagnostic()> emitError, const StringAttr name,
    const ArrayRef<SiteAttr> sites, const DurationUnitAttr durationUnit,
    const ConnectivityKind connectivity, const ArrayRef<CouplingAttr> couplings,
    const NativeOperationsKind nativeOperations,
    const ArrayRef<NativeOperationAttr> operations) {
  if (name && name.getValue().empty()) {
    return emitError() << "compiler target name must not be empty when present";
  }
  if (sites.empty()) {
    return emitError() << "compiler target must contain at least one site";
  }

  llvm::SmallDenseSet<int64_t> siteIds;
  siteIds.reserve(sites.size());
  for (const SiteAttr site : sites) {
    if (!siteIds.insert(site.getId()).second) {
      return emitError() << "compiler target contains duplicate site IDs";
    }
  }

  if (connectivity != ConnectivityKind::Explicit && !couplings.empty()) {
    return emitError()
           << "compiler target couplings require explicit connectivity";
  }
  if (connectivity == ConnectivityKind::Explicit) {
    llvm::SmallDenseSet<std::pair<int64_t, int64_t>> seen;
    for (const CouplingAttr coupling : couplings) {
      auto source = coupling.getSource();
      auto target = coupling.getTarget();
      if (!siteIds.contains(source) || !siteIds.contains(target)) {
        return emitError()
               << "compiler target coupling references an unknown site";
      }
      if (target < source) {
        std::swap(source, target);
      }
      if (!seen.insert({source, target}).second) {
        return emitError() << "compiler target contains a duplicate coupling";
      }
    }
  }

  if (nativeOperations != NativeOperationsKind::Explicit &&
      !operations.empty()) {
    return emitError()
           << "compiler target operations require explicit native operations";
  }
  for (const NativeOperationAttr operation : operations) {
    if (operation.getArity().getValue() > sites.size()) {
      if (operation.getArity().getKind() == OperationArityKind::Variadic) {
        return emitError() << "compiler target operation variadic minimum "
                              "exceeds its site count";
      }
      return emitError() << "compiler target operation arity exceeds its site "
                            "count";
    }
    for (const SiteTupleAttr siteTuple : operation.getSiteTuples()) {
      if (llvm::any_of(siteTuple.getSites(), [&](const int64_t site) {
            return !siteIds.contains(site);
          })) {
        return emitError() << "compiler target operation site tuple references "
                              "an unknown site";
      }
    }
  }

  const bool hasTiming =
      llvm::any_of(sites,
                   [](const SiteAttr site) {
                     return site.getT1().has_value() ||
                            site.getT2().has_value();
                   }) ||
      llvm::any_of(operations, [](const NativeOperationAttr operation) {
        return operation.getDuration().has_value() ||
               llvm::any_of(operation.getSiteTuples(),
                            [](const SiteTupleAttr siteTuple) {
                              return siteTuple.getDuration().has_value();
                            });
      });
  if (hasTiming && !durationUnit) {
    return emitError()
           << "compiler target timing metadata requires a duration unit";
  }
  return success();
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

[[nodiscard]] static bool hasQCQubit(Type type) {
  return isa<qc::QubitType>(type);
}

[[nodiscard]] static bool hasQCOQubit(Type type) {
  return isa<qco::QubitType>(type);
}

template <typename CallOp>
[[nodiscard]] static LogicalResult
verifyNoUnitaryRecursion(func::FuncOp function) {
  DenseSet<Operation*> visited;
  SmallVector<func::FuncOp> worklist{function};
  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }
    WalkResult result = current.walk([&](CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee) {
        return WalkResult::advance();
      }
      if (callee == function) {
        return WalkResult::interrupt();
      }
      worklist.emplace_back(callee);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return function.emitError() << "unitary function must not be recursive";
    }
  }
  return success();
}

[[nodiscard]] static LogicalResult verifyQCUnitaryBody(func::FuncOp function) {
  auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
  if (!returnOp || returnOp.getNumOperands() != 0) {
    return function.emitError(
        "unitary QC function must end in an empty func.return");
  }

  bool valid = true;
  function.walk([&](Operation* nested) {
    if (!valid || nested == function.getOperation()) {
      return;
    }
    if (isa<func::ReturnOp>(nested)) {
      return;
    }
    if (isa<qc::UnitaryOpInterface, qc::YieldOp>(nested)) {
      return;
    }
    valid = nested->getNumRegions() == 0 && isMemoryEffectFree(nested) &&
            llvm::none_of(nested->getOperandTypes(), hasQCQubit) &&
            llvm::none_of(nested->getResultTypes(), hasQCQubit);
  });
  if (!valid) {
    return function.emitError()
           << "unitary QC function body contains a non-unitary operation";
  }

  return verifyNoUnitaryRecursion<qc::CallOp>(function);
}

[[nodiscard]] static LogicalResult
verifyQCOUnitaryBody(func::FuncOp function, const unsigned firstQubit) {
  bool valid = true;
  function.walk([&](Operation* nested) {
    if (!valid || nested == function.getOperation()) {
      return;
    }
    if (isa<func::ReturnOp, qco::UnitaryOpInterface, qco::YieldOp>(nested)) {
      return;
    }
    valid = nested->getNumRegions() == 0 && isMemoryEffectFree(nested) &&
            llvm::none_of(nested->getOperandTypes(), hasQCOQubit) &&
            llvm::none_of(nested->getResultTypes(), hasQCOQubit);
  });
  if (!valid) {
    return function.emitError()
           << "unitary QCO function body contains a non-unitary operation";
  }

  auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
  if (!returnOp) {
    return function.emitError("unitary QCO function must end in func.return");
  }
  for (auto [resultIndex, returned] : llvm::enumerate(returnOp.getOperands())) {
    Value current = returned;
    while (auto result = dyn_cast<OpResult>(current)) {
      auto unitary = dyn_cast<qco::UnitaryOpInterface>(result.getOwner());
      if (!unitary) {
        return function.emitError()
               << "unitary QCO result does not originate from a qubit "
                  "argument";
      }
      current = unitary.getInputForOutput(current);
      if (!current) {
        return function.emitError()
               << "unitary QCO operation has no input corresponding to its "
                  "returned qubit";
      }
    }
    auto argument = dyn_cast<BlockArgument>(current);
    if (!argument || argument.getOwner() != &function.getBody().front() ||
        argument.getArgNumber() != firstQubit + resultIndex) {
      return function.emitError()
             << "unitary QCO results must continue qubit arguments "
                "positionally";
    }
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
  if (usesQC) {
    if (function.getNumResults() != 0) {
      return operation->emitError()
             << "unitary QC function must not return values";
    }
    return verifyQCUnitaryBody(function);
  }
  if (function.getNumResults() != numQubits ||
      llvm::any_of(function.getResultTypes(),
                   [](Type type) { return !isa<qco::QubitType>(type); })) {
    return operation->emitError()
           << "unitary QCO function must return one qubit per qubit argument";
  }
  return verifyQCOUnitaryBody(function, firstQubit);
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
