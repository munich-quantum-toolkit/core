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
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace mlir {

bool ResultLayout::isIdentity() const noexcept {
  return bits.size() == rawWidth &&
         llvm::all_of(llvm::enumerate(bits), [](auto entry) {
           return entry.value() == entry.index();
         });
}

llvm::Expected<std::string>
ResultLayout::decodeBits(llvm::StringRef raw) const {
  if (raw.size() != rawWidth ||
      raw.find_first_not_of("01") != StringRef::npos) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "raw sample has an invalid width or digit");
  }
  std::string result(bits.size(), '0');
  for (auto [index, slot] : llvm::enumerate(bits)) {
    if (slot) {
      if (*slot >= rawWidth) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "result layout selects an invalid raw bit");
      }
      result[bits.size() - index - 1] = raw[rawWidth - *slot - 1];
    }
  }
  return result;
}

FailureOr<ResultLayout> normalizeTerminalReadout(ModuleOp moduleOp,
                                                 TerminalReadout readout) {
  auto function = mqt::getEntryPoint(moduleOp);
  if (!function || !llvm::hasSingleElement(function.getBody())) {
    return moduleOp.emitError(
        "terminal readout requires a straight-line entry function");
  }
  auto& block = function.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
  if (!returnOp || returnOp.getNumOperands() == 0) {
    return function.emitError("terminal readout requires bit-register outputs");
  }

  struct Register {
    SmallVector<std::optional<size_t>> sites;
    bool zeroInitialized;
  };
  DenseMap<Value, Register> registers;
  std::map<size_t, Value> measuredSites;
  SmallVector<Operation*> oldReadout;
  size_t addressWidth = 0;
  for (auto& operation : block.without_terminator()) {
    if (auto qubit = dyn_cast<qc::StaticOp>(operation)) {
      if (!qubit.getQubit().use_empty()) {
        addressWidth =
            std::max(addressWidth, static_cast<size_t>(qubit.getIndex()) + 1);
      }
    } else if (auto measurement = dyn_cast<qc::MeasureOp>(operation)) {
      auto qubit = measurement.getQubit().getDefiningOp<qc::StaticOp>();
      if (!qubit) {
        return measurement.emitError("terminal readout requires static qubits");
      }
      if (!llvm::all_of(
              measurement.getResult().getUsers(), [&](Operation* user) {
                return isa<cbit::StoreOp>(user) && user->getBlock() == &block;
              })) {
        return measurement.emitError(
            "terminal readout requires measurement results stored directly "
            "in top-level bit registers");
      }
      const auto site = static_cast<size_t>(qubit.getIndex());
      measuredSites.try_emplace(site, qubit.getQubit());
      oldReadout.push_back(&operation);
    } else if (auto allocation = dyn_cast<cbit::AllocOp>(operation)) {
      registers.try_emplace(
          allocation.getResult(),
          Register{
              SmallVector<std::optional<size_t>>(
                  allocation.getResult().getType().getWidth()),
              allocation.getInitialization() == cbit::Initialization::Zero,
          });
      oldReadout.push_back(&operation);
    } else if (auto store = dyn_cast<cbit::StoreOp>(operation)) {
      const auto index = getConstantIntValue(store.getIndex());
      auto reg = registers.find(store.getReg());
      auto measurement = store.getValue().getDefiningOp<qc::MeasureOp>();
      if (!index || *index < 0 || reg == registers.end() ||
          static_cast<size_t>(*index) >= reg->second.sites.size() ||
          !measurement) {
        return store.emitError("terminal readout requires direct measurements "
                               "at static bit destinations");
      }
      reg->second.sites[static_cast<size_t>(*index)] = static_cast<size_t>(
          measurement.getQubit().getDefiningOp<qc::StaticOp>().getIndex());
      oldReadout.push_back(&operation);
    } else if (auto call = dyn_cast<func::CallOp>(operation)) {
      if (!measuredSites.empty() || call.getNumResults() != 0 ||
          !llvm::all_of(call.getOperandTypes(), [](Type type) {
            return type.isF64() || isa<qc::QubitType>(type);
          })) {
        return call.emitError(
            "terminal readout only supports gate calls before measurements");
      }
    } else if (isa<arith::ConstantOp, qc::BarrierOp>(operation)) {
      continue;
    } else if (isa<qc::UnitaryOpInterface, qc::ResetOp>(operation)) {
      for (auto operand : operation.getOperands()) {
        if (!isa<qc::QubitType>(operand.getType())) {
          continue;
        }
        auto qubit = operand.getDefiningOp<qc::StaticOp>();
        if (!qubit || measuredSites.contains(qubit.getIndex())) {
          return operation.emitError(
              "terminal readout requires gates on unmeasured static qubits");
        }
      }
    } else {
      return operation.emitError(
          "terminal readout requires gates and direct bit stores");
    }
  }
  ResultLayout layout;
  for (auto output : returnOp.getOperands()) {
    auto reg = registers.find(output);
    if (reg == registers.end()) {
      return returnOp.emitError(
          "terminal readout requires bit-register outputs");
    }
    for (auto site : reg->second.sites) {
      if (!site && !reg->second.zeroInitialized) {
        return returnOp.emitError(
            "terminal readout has an undefined output bit");
      }
      layout.bits.push_back(site);
    }
  }
  if (readout == TerminalReadout::All) {
    for (size_t site = 0; site < addressWidth; ++site) {
      measuredSites.try_emplace(site, Value{});
    }
  }
  if (measuredSites.empty()) {
    return returnOp.emitError(
        "terminal readout requires at least one measured qubit");
  }

  OpBuilder builder(returnOp);
  auto raw = cbit::AllocOp::create(
      builder, returnOp.getLoc(),
      cbit::RegisterType::get(moduleOp.getContext(),
                              static_cast<int64_t>(measuredSites.size())),
      cbit::Initialization::Undefined);
  raw->setAttr(mqt::MQTDialect::RegisterNameAttrHelper::getNameStr(),
               builder.getStringAttr("c"));
  DenseMap<size_t, size_t> slots;
  for (auto [site, qubit] : measuredSites) {
    const auto index = slots.size();
    slots.try_emplace(site, index);
    if (!qubit) {
      qubit = qc::StaticOp::create(builder, returnOp.getLoc(), site);
    }
    auto offset = arith::ConstantIndexOp::create(builder, returnOp.getLoc(),
                                                 static_cast<int64_t>(index));
    auto bit = qc::MeasureOp::create(builder, returnOp.getLoc(), qubit);
    cbit::StoreOp::create(builder, returnOp.getLoc(), bit, raw, offset);
  }
  layout.rawWidth = slots.size();
  for (auto& site : layout.bits) {
    if (site) {
      *site = slots.at(*site);
    }
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
