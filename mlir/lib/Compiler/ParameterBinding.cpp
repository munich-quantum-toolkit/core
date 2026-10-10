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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/LogicalResult.h"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace mlir {
static StringAttr parameterName(func::FuncOp entry, const unsigned index) {
  if (!entry.getArgument(index).getType().isF64()) {
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
                          const std::map<std::string, double>& values) {
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
  llvm::StringMap<unsigned> indices;
  for (unsigned index = 0; index < entry.getNumArguments(); ++index) {
    if (auto name = parameterName(entry, index)) {
      indices.try_emplace(name.getValue(), index);
    }
  }
  for (const auto& [name, value] : values) {
    if (!indices.contains(name)) {
      entry.emitError() << "unknown f64 parameter '" << name << "'";
      return failure();
    }
    if (!std::isfinite(value)) {
      entry.emitError() << "parameter '" << name << "' must be finite";
      return failure();
    }
  }
  OpBuilder builder(&entry.getBody().front(), entry.getBody().front().begin());
  llvm::BitVector erased(entry.getNumArguments());
  for (const auto& [name, value] : values) {
    const auto index = indices.lookup(name);
    auto constant = arith::ConstantOp::create(builder, entry.getLoc(),
                                              builder.getF64FloatAttr(value));
    entry.getArgument(index).replaceAllUsesWith(constant);
    erased.set(index);
  }
  return entry.eraseArguments(erased);
}
std::vector<std::string> QCProgram::parameters() const {
  return parameterNames(mod());
}
std::vector<std::string> QCOProgram::parameters() const {
  return parameterNames(mod());
}
LogicalResult
QCProgram::bindParameters(const std::map<std::string, double>& values) {
  return bind(mod(), values);
}
LogicalResult
QCOProgram::bindParameters(const std::map<std::string, double>& values) {
  return bind(mod(), values);
}
} // namespace mlir
