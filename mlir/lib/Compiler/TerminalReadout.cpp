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
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
#include <utility>
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

static llvm::Expected<std::string>
decodeSample(std::string raw, std::optional<size_t>& width,
             const std::optional<ResultLayout>& layout) {
  std::erase(raw, ' ');
  if (layout) {
    auto decoded = layout->decodeBits(raw);
    if (!decoded) {
      return decoded.takeError();
    }
    raw = std::move(*decoded);
  }
  if (!width) {
    width = raw.size();
  }
  if (raw.size() != *width ||
      raw.find_first_not_of("01") != std::string::npos) {
    return llvm::createStringError("Invalid QDMI sample width or binary digit");
  }
  return raw;
}

llvm::Expected<std::map<std::string, size_t>>
decodeCounts(std::map<std::string, size_t> counts, std::optional<size_t> width,
             size_t shots, const std::optional<ResultLayout>& layout) {
  std::map<std::string, size_t> result;
  size_t total = 0;
  while (!counts.empty()) {
    auto entry = counts.extract(counts.begin());
    auto decoded = decodeSample(std::move(entry.key()), width, layout);
    if (!decoded) {
      return decoded.takeError();
    }
    const auto count = entry.mapped();
    if (count > shots - total) {
      return llvm::createStringError("Invalid QDMI histogram: too many shots");
    }
    total += count;
    if (count != 0) {
      entry.key() = std::move(*decoded);
      auto inserted = result.insert(result.end(), std::move(entry));
      // NOLINTNEXTLINE(bugprone-use-after-move): Duplicates retain the node.
      if (!entry.empty()) {
        inserted->second += count;
      }
    }
  }
  if (total != shots) {
    return llvm::createStringError(
        "Invalid QDMI histogram: incorrect shot total");
  }
  return result;
}

llvm::Expected<std::vector<std::string>>
decodeShots(std::vector<std::string> samples, std::optional<size_t> width,
            size_t shots, const std::optional<ResultLayout>& layout) {
  if (samples.size() != shots) {
    return llvm::createStringError(
        "Invalid QDMI SHOTS result: incorrect shot total");
  }
  std::map<std::string, std::string> cache;
  for (auto& bits : samples) {
    auto found = cache.find(bits);
    if (found == cache.end()) {
      auto decoded = decodeSample(bits, width, layout);
      if (!decoded) {
        return decoded.takeError();
      }
      found = cache.emplace(bits, std::move(*decoded)).first;
    }
    bits = found->second;
  }
  return samples;
}

namespace {

struct TerminalReadoutPlan {
  func::FuncOp function;
  func::ReturnOp returnOp;
  std::map<size_t, Value> measuredSites;
  SmallVector<Operation*> oldReadout;
  ResultLayout layout;
};

} // namespace

static FailureOr<TerminalReadoutPlan>
analyzeTerminalReadout(ModuleOp moduleOp) {
  auto function = mqt::getEntryPoint(moduleOp);
  if (!function || !llvm::hasSingleElement(function.getBody())) {
    return failure();
  }
  auto& block = function.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
  if (!returnOp || returnOp.getNumOperands() == 0) {
    return failure();
  }

  struct Register {
    SmallVector<std::optional<size_t>> sites;
    bool zeroInitialized;
  };
  DenseMap<Value, Register> registers;
  std::map<size_t, Value> measuredSites;
  SmallVector<Operation*> oldReadout;
  for (auto& operation : block.without_terminator()) {
    if (isa<qc::StaticOp>(operation)) {
      continue;
    }
    if (auto measurement = dyn_cast<qc::MeasureOp>(operation)) {
      auto qubit = measurement.getQubit().getDefiningOp<qc::StaticOp>();
      if (!qubit) {
        return failure();
      }
      if (!llvm::all_of(
              measurement.getResult().getUsers(), [&](Operation* user) {
                return isa<cbit::StoreOp>(user) && user->getBlock() == &block;
              })) {
        return failure();
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
        return failure();
      }
      reg->second.sites[static_cast<size_t>(*index)] = static_cast<size_t>(
          measurement.getQubit().getDefiningOp<qc::StaticOp>().getIndex());
      oldReadout.push_back(&operation);
    } else if (isa<arith::ConstantOp, qc::BarrierOp>(operation)) {
      continue;
    } else if (auto call = dyn_cast<func::CallOp>(operation)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      // Defined functions can hide sites or classical references. Native
      // declarations act only on their explicit qubit operands.
      if (!callee || !callee.isExternal() || call.getNumResults() != 0 ||
          !llvm::all_of(call.getOperandTypes(), [](Type type) {
            return type.isF64() || isa<qc::QubitType>(type);
          })) {
        return failure();
      }
      for (auto operand : call.getOperands()) {
        if (!isa<qc::QubitType>(operand.getType())) {
          continue;
        }
        auto qubit = operand.getDefiningOp<qc::StaticOp>();
        if (!qubit || measuredSites.contains(qubit.getIndex())) {
          return failure();
        }
      }
    } else if (isa<qc::UnitaryOpInterface, qc::ResetOp>(operation)) {
      for (auto operand : operation.getOperands()) {
        if (!isa<qc::QubitType>(operand.getType())) {
          continue;
        }
        auto qubit = operand.getDefiningOp<qc::StaticOp>();
        if (!qubit || measuredSites.contains(qubit.getIndex())) {
          return failure();
        }
      }
    } else {
      return failure();
    }
  }
  ResultLayout layout;
  for (auto output : returnOp.getOperands()) {
    auto reg = registers.find(output);
    if (reg == registers.end()) {
      return failure();
    }
    for (auto site : reg->second.sites) {
      if (!site && !reg->second.zeroInitialized) {
        return failure();
      }
      layout.bits.push_back(site);
    }
  }

  return TerminalReadoutPlan{
      function,
      returnOp,
      std::move(measuredSites),
      std::move(oldReadout),
      std::move(layout),
  };
}

void scheduleTerminalReadout(ModuleOp moduleOp) {
  auto plan = analyzeTerminalReadout(moduleOp);
  if (failed(plan)) {
    return;
  }
  for (auto* operation : plan->oldReadout) {
    if (isa<qc::MeasureOp, cbit::StoreOp>(operation)) {
      operation->moveBefore(plan->returnOp);
    }
  }
}

llvm::Expected<std::optional<ResultLayout>>
normalizeTerminalReadout(ModuleOp moduleOp) {
  llvm::DenseSet<Value> measuredQubits;
  llvm::DenseSet<uint64_t> measuredIndices;
  bool nativeAfterMeasurement = false;
  auto entry = mqt::getEntryPoint(moduleOp);
  if (!entry) {
    return std::nullopt;
  }
  const auto nativeReuse = entry.walk([&](Operation* operation) {
    if (auto measurement = dyn_cast<qc::MeasureOp>(operation)) {
      measuredQubits.insert(measurement.getQubit());
      if (auto site = measurement.getQubit().getDefiningOp<qc::StaticOp>()) {
        measuredIndices.insert(site.getIndex());
      }
    } else if (auto call = dyn_cast<func::CallOp>(operation)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (callee && callee.isExternal()) {
        nativeAfterMeasurement |= !measuredQubits.empty();
        for (auto operand : call.getOperands()) {
          auto site = operand.getDefiningOp<qc::StaticOp>();
          if (measuredQubits.contains(operand) ||
              (site && measuredIndices.contains(site.getIndex()))) {
            return WalkResult::interrupt();
          }
        }
      }
    }
    return WalkResult::advance();
  });
  if (nativeReuse.wasInterrupted()) {
    return llvm::createStringError(
        "native calls cannot reuse a measured qubit");
  }
  auto plan = analyzeTerminalReadout(moduleOp);
  if (failed(plan)) {
    if (nativeAfterMeasurement) {
      return llvm::createStringError(
          "native calls after measurements require static terminal readout");
    }
    return std::nullopt;
  }
  auto& [function, returnOp, measuredSites, oldReadout, layout] = *plan;
  if (measuredSites.empty()) {
    return std::nullopt;
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
  return std::move(layout);
}

} // namespace mlir
