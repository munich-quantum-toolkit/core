/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TerminalReadout.h"

#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace mlir {

bool ResultLayout::isIdentity() const noexcept {
  if (bits.size() != rawWidth) {
    return false;
  }
  for (auto [index, bit] : llvm::enumerate(bits)) {
    const auto* slot = std::get_if<size_t>(&bit);
    if (slot == nullptr || *slot != index) {
      return false;
    }
  }
  return true;
}

llvm::Expected<std::string>
ResultLayout::decodeBits(llvm::StringRef raw) const {
  if (raw.size() != rawWidth ||
      raw.find_first_not_of("01") != StringRef::npos) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "raw sample has an invalid width or digit");
  }
  std::string result(bits.size(), '0');
  for (auto [index, bit] : llvm::enumerate(bits)) {
    if (const auto* slot = std::get_if<size_t>(&bit)) {
      if (*slot >= rawWidth) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "result layout selects an invalid raw bit");
      }
      result[bits.size() - index - 1] = raw[rawWidth - *slot - 1];
    } else {
      result[bits.size() - index - 1] = std::get<bool>(bit) ? '1' : '0';
    }
  }
  return result;
}

FailureOr<ResultLayout> normalizeTerminalReadout(ModuleOp moduleOp,
                                                 bool physicalOrder,
                                                 bool measureAll) {
  auto function = mqt::getEntryPoint(moduleOp);
  if (!function || !llvm::hasSingleElement(function.getBody())) {
    return moduleOp.emitError(
        "terminal readout requires a straight-line entry function");
  }
  auto& block = function.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
  if (!returnOp || returnOp.getNumOperands() == 0) {
    return function.emitError("terminal readout requires binary outputs");
  }

  using Bit = std::variant<size_t, bool>;
  DenseMap<Value, SmallVector<std::optional<Bit>>> registers;
  DenseMap<Value, size_t> measurements;
  DenseMap<size_t, Value> sites;
  DenseSet<size_t> measured;
  SmallVector<size_t> measurementOrder;
  SmallVector<Operation*> oldReadout;
  const auto bitValue = [&](Value value) -> std::optional<Bit> {
    if (auto found = measurements.find(value); found != measurements.end()) {
      return Bit(found->second);
    }
    APInt constant;
    if (value.getType().isInteger(1) &&
        matchPattern(value, m_ConstantInt(&constant))) {
      return Bit(!constant.isZero());
    }
    return std::nullopt;
  };

  for (auto& operation : block.without_terminator()) {
    if (operation.getNumRegions() != 0 &&
        !isa<qc::UnitaryOpInterface>(operation)) {
      return operation.emitError(
          "terminal readout does not support runtime control flow");
    }
    if (auto qubit = dyn_cast<qc::StaticOp>(operation)) {
      sites.try_emplace(qubit.getIndex(), qubit.getQubit());
      continue;
    }
    if (auto measurement = dyn_cast<qc::MeasureOp>(operation)) {
      auto qubit = measurement.getQubit().getDefiningOp<qc::StaticOp>();
      if (!qubit) {
        return measurement.emitError("terminal readout requires static qubits");
      }
      const auto site = static_cast<size_t>(qubit.getIndex());
      measurements.try_emplace(measurement.getResult(), site);
      if (measured.insert(site).second) {
        measurementOrder.push_back(site);
      }
      oldReadout.push_back(&operation);
      continue;
    }
    if (auto allocation = dyn_cast<cbit::AllocOp>(operation)) {
      std::optional<Bit> initial;
      if (allocation.getInitialization() == cbit::Initialization::Zero) {
        initial = false;
      }
      registers.try_emplace(allocation.getResult(),
                            allocation.getResult().getType().getWidth(),
                            initial);
      oldReadout.push_back(&operation);
      continue;
    }
    if (auto store = dyn_cast<cbit::StoreOp>(operation)) {
      const auto index = getConstantIntValue(store.getIndex());
      auto found = registers.find(store.getReg());
      auto value = bitValue(store.getValue());
      if (!index || found == registers.end() || *index < 0 ||
          static_cast<size_t>(*index) >= found->second.size() || !value) {
        return store.emitError(
            "terminal readout requires constant destinations and measurement "
            "or constant bit values");
      }
      found->second[static_cast<size_t>(*index)] = *value;
      oldReadout.push_back(&operation);
      continue;
    }
    if (auto write = dyn_cast<cbit::WriteOp>(operation)) {
      auto found = registers.find(write.getReg());
      APInt constant;
      if (found == registers.end() ||
          !matchPattern(write.getValue(), m_ConstantInt(&constant))) {
        return write.emitError(
            "terminal readout requires constant register assignments");
      }
      for (auto [index, bit] : llvm::enumerate(found->second)) {
        bit = constant[index];
      }
      oldReadout.push_back(&operation);
      continue;
    }
    for (auto operand : operation.getOperands()) {
      if (registers.contains(operand) || measurements.contains(operand)) {
        return operation.emitError(
            "terminal readout does not support measurement feedback or "
            "classical computation");
      }
      if (auto site = operand.getDefiningOp<qc::StaticOp>();
          site && measured.contains(site.getIndex())) {
        return operation.emitError(
            "terminal readout does not support qubit use after measurement");
      }
    }
  }

  ResultLayout layout;
  for (auto value : returnOp.getOperands()) {
    if (auto found = registers.find(value); found != registers.end()) {
      for (const auto& bit : found->second) {
        if (!bit) {
          return returnOp.emitError(
              "terminal readout has an undefined output bit");
        }
        layout.bits.push_back(*bit);
      }
    } else if (auto bit = bitValue(value)) {
      layout.bits.push_back(*bit);
    } else {
      return returnOp.emitError(
          "terminal readout requires bit registers or binary scalar outputs");
    }
  }
  if (measurementOrder.empty() && !measureAll) {
    return returnOp.emitError(
        "terminal readout requires at least one measured qubit");
  }

  SmallVector<size_t> ordered;
  DenseMap<size_t, size_t> slots;
  const auto addSite = [&](size_t site) {
    if (slots.try_emplace(site, ordered.size()).second) {
      ordered.push_back(site);
    }
  };
  if (physicalOrder) {
    llvm::sort(measurementOrder);
  } else {
    for (const auto& bit : layout.bits) {
      if (const auto* site = std::get_if<size_t>(&bit)) {
        addSite(*site);
      }
    }
  }
  for (const auto site : measurementOrder) {
    addSite(site);
  }
  if (measureAll) {
    size_t width = 0;
    for (auto [site, qubit] : sites) {
      if (!qubit.use_empty()) {
        width = std::max(width, site + 1);
      }
    }
    if (width == 0) {
      return returnOp.emitError("terminal readout requires at least one qubit");
    }
    ordered.clear();
    slots.clear();
    for (size_t site = 0; site < width; ++site) {
      addSite(site);
    }
  }
  layout.rawWidth = ordered.size();
  for (auto& bit : layout.bits) {
    if (auto* site = std::get_if<size_t>(&bit)) {
      *site = slots.at(*site);
    }
  }

  OpBuilder builder(returnOp);
  auto raw = cbit::AllocOp::create(
      builder, returnOp.getLoc(),
      cbit::RegisterType::get(moduleOp.getContext(),
                              static_cast<int64_t>(ordered.size())),
      cbit::Initialization::Undefined);
  raw->setAttr(mqt::MQTDialect::RegisterNameAttrHelper::getNameStr(),
               builder.getStringAttr("c"));
  for (auto [index, site] : llvm::enumerate(ordered)) {
    if (!sites.contains(site)) {
      sites.try_emplace(site,
                        qc::StaticOp::create(builder, returnOp.getLoc(), site));
    }
    auto offset = arith::ConstantIndexOp::create(builder, returnOp.getLoc(),
                                                 static_cast<int64_t>(index));
    auto bit =
        qc::MeasureOp::create(builder, returnOp.getLoc(), sites.at(site));
    cbit::StoreOp::create(builder, returnOp.getLoc(), bit, raw, offset);
  }
  returnOp.getOperandsMutable().assign(raw.getResult());
  function.removeResAttrsAttr();
  function.setFunctionType(
      builder.getFunctionType(function.getArgumentTypes(), {raw.getType()}));
  for (auto* operation : llvm::reverse(oldReadout)) {
    operation->erase();
  }
  return layout;
}

} // namespace mlir
