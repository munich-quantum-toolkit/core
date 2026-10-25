/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/QCOUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlir {
static StringAttr parameterName(func::FuncOp entry, unsigned index) {
  const auto type = entry.getArgument(index).getType();
  const auto tensor = dyn_cast<RankedTensorType>(type);
  if (!type.isF64() &&
      !(tensor && tensor.hasStaticShape() && tensor.getRank() == 1 &&
        tensor.getElementType().isF64())) {
    return {};
  }
  return entry.getArgAttrOfType<StringAttr>(
      index, mqt::MQTDialect::InputNameAttrHelper::getNameStr());
}

static std::vector<std::string> parameterNames(ModuleOp moduleOp) {
  std::vector<std::string> names;
  if (auto entry = mqt::getEntryPoint(moduleOp)) {
    for (unsigned index = 0; index < entry.getNumArguments(); ++index) {
      if (auto name = parameterName(entry, index)) {
        names.push_back(name.str());
      }
    }
  }
  return names;
}

static LogicalResult bind(ModuleOp moduleOp,
                          const std::map<std::string, ParameterValue>& values) {
  if (values.empty()) {
    return success();
  }
  auto entry = mqt::getEntryPoint(moduleOp);
  if (!entry) {
    moduleOp.emitError("parameter binding requires an entry point");
    return failure();
  }
  if (!SymbolTable::symbolKnownUseEmpty(entry, moduleOp)) {
    entry.emitError("cannot bind parameters of a referenced entry point");
    return failure();
  }
  llvm::StringMap<BlockArgument> arguments;
  for (unsigned index = 0; index < entry.getNumArguments(); ++index) {
    if (auto name = parameterName(entry, index)) {
      arguments.try_emplace(name.getValue(), entry.getArgument(index));
    }
  }
  OpBuilder builder(moduleOp.getContext());
  SmallVector<std::pair<BlockArgument, TypedAttr>> replacements;
  replacements.reserve(values.size());
  for (const auto& [name, value] : values) {
    const auto argument = arguments.find(name);
    if (argument == arguments.end()) {
      entry.emitError() << "unknown parameter '" << name << "'";
      return failure();
    }
    const auto type = argument->second.getType();
    TypedAttr attribute;
    if (type.isF64()) {
      const auto* const scalar = std::get_if<double>(&value);
      if (scalar == nullptr) {
        entry.emitError() << "parameter '" << name << "' requires a scalar";
        return failure();
      }
      if (!std::isfinite(*scalar)) {
        entry.emitError() << "parameter '" << name << "' must be finite";
        return failure();
      }
      attribute = builder.getF64FloatAttr(*scalar);
    } else {
      const auto* const array = std::get_if<std::vector<double>>(&value);
      if (array == nullptr) {
        entry.emitError() << "parameter '" << name << "' requires an array";
        return failure();
      }
      const auto size = cast<RankedTensorType>(type).getDimSize(0);
      if (array->size() != static_cast<size_t>(size)) {
        entry.emitError() << "parameter '" << name << "' requires " << size
                          << " elements, got " << array->size();
        return failure();
      }
      if (!std::ranges::all_of(
              *array, [](double element) { return std::isfinite(element); })) {
        entry.emitError() << "parameter '" << name
                          << "' must contain finite values";
        return failure();
      }
      attribute =
          DenseFPElementsAttr::get(cast<RankedTensorType>(type), *array);
    }
    replacements.emplace_back(argument->second, attribute);
  }
  builder.setInsertionPointToStart(&entry.getBody().front());
  llvm::BitVector erased(entry.getNumArguments());
  for (auto [argument, attribute] : replacements) {
    auto constant =
        arith::ConstantOp::create(builder, entry.getLoc(), attribute);
    argument.replaceAllUsesWith(constant);
    erased.set(argument.getArgNumber());
  }
  if (failed(entry.eraseArguments(erased))) {
    return failure();
  }
  auto pm = PassManager::on<func::FuncOp>(moduleOp.getContext());
  pm.addPass(createCanonicalizerPass());
  return pm.run(entry);
}
std::vector<std::string> QCProgram::parameters() const {
  return parameterNames(mod());
}
std::vector<std::string> QCOProgram::parameters() const {
  return parameterNames(mod());
}
LogicalResult
QCProgram::bindParameters(const std::map<std::string, ParameterValue>& values) {
  return bind(mod(), values);
}
LogicalResult QCOProgram::bindParameters(
    const std::map<std::string, ParameterValue>& values) {
  if (failed(bind(mod(), values))) {
    return failure();
  }
  return qco::verifyLinearity(mod());
}
} // namespace mlir
