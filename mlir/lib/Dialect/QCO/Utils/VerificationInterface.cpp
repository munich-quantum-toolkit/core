/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/VerificationInterface.h"

#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>

namespace mlir::qco {
namespace {

constexpr size_t MAX_WIRES =
    static_cast<size_t>(std::numeric_limits<dd::Qubit>::max()) + 1;
constexpr size_t UNASSIGNED = std::numeric_limits<size_t>::max();

} // namespace

static std::string outputLabel(const DDOutputBit& bit) {
  auto label = "result" + std::to_string(bit.resultIndex);
  if (!bit.name.empty()) {
    label += ":" + bit.name;
  }
  return label + "[" + std::to_string(bit.bitIndex) + "]";
}

FailureOr<VerificationInterface>
resolveVerificationInterface(func::FuncOp func,
                             const DDResolvedExecution& execution,
                             const VerificationContract& contract,
                             std::optional<size_t> expectedSourceCount) {
  if (execution.status != DDTraceStatus::Complete) {
    return func.emitError(
        "verification requires a complete resolved execution");
  }
  const auto moduleOp = func->getParentOfType<ModuleOp>();
  if ((moduleOp == nullptr) || func.isDeclaration() ||
      !func.getBody().hasOneBlock()) {
    return func.emitError("verification requires a one-block module function");
  }
  if (execution.numQubits > MAX_WIRES) {
    return func.emitError("verification wire count exceeds the DD index range");
  }

  auto roots = execution.roots;
  llvm::sort(roots, [](const DDWireRoot& left, const DDWireRoot& right) {
    return std::tie(left.kind, left.argumentIndex, left.allocationIndex,
                    left.staticId, left.elementIndex) <
           std::tie(right.kind, right.argumentIndex, right.allocationIndex,
                    right.staticId, right.elementIndex);
  });
  std::vector<bool> seen(execution.numQubits);
  for (const auto& root : roots) {
    if (root.wire >= execution.numQubits || seen[root.wire]) {
      return func.emitError("resolved quantum roots must have distinct wires");
    }
    seen[root.wire] = true;
  }
  if (llvm::is_contained(seen, false)) {
    return func.emitError("resolved quantum roots must cover every trace wire");
  }

  std::optional<mqt::QubitLayout> layout;
  if (const auto attr = moduleOp->getAttr("mqt.layout")) {
    auto parsed = mqt::QubitLayout::fromAttr(
        attr, [&] { return func.emitError("invalid verification layout: "); });
    if (failed(parsed)) {
      return failure();
    }
    layout = std::move(*parsed);
    if (layout->initial.size() > MAX_WIRES) {
      return func.emitError("verification layout exceeds the DD index range");
    }
  }

  std::optional<size_t> sourceCount;
  if (const auto attr = moduleOp->getAttr(mqt::kSourceQubitCountAttr)) {
    const auto count = dyn_cast<IntegerAttr>(attr);
    if (!count || !count.getType().isSignlessInteger(64) ||
        count.getInt() < 0 || std::cmp_greater(count.getInt(), MAX_WIRES)) {
      return func.emitError("invalid verification source qubit count");
    }
    sourceCount = static_cast<size_t>(count.getInt());
  }
  const auto hasArguments = llvm::any_of(roots, [](const auto& root) {
    return root.kind == DDWireRoot::Kind::Argument;
  });
  const auto hasStaticRoots = llvm::any_of(roots, [](const auto& root) {
    return root.kind == DDWireRoot::Kind::Static;
  });
  if (!layout && !sourceCount && moduleOp->hasAttr("mqt.target_env") &&
      (!hasArguments || hasStaticRoots)) {
    return func.emitError(
        "target program has no input layout provenance for verification");
  }
  if (layout && layout->sites && sourceCount) {
    return func.emitError("placed layout must not retain source qubit tags");
  }

  std::vector<size_t> input;
  std::vector<size_t> output;
  std::vector<bool> argumentSlots;
  size_t nextWire = execution.numQubits;
  const auto fresh = [&] { return nextWire++; };

  if (sourceCount) {
    if (llvm::any_of(roots, [](const auto& root) {
          return root.kind != DDWireRoot::Kind::Allocation;
        })) {
      return func.emitError("source qubit tags require entry allocation roots");
    }
    SmallVector<Operation*> allocations;
    for (Operation& operation : func.getBody().front()) {
      if (isa<AllocOp, qtensor::AllocOp>(operation)) {
        allocations.push_back(&operation);
      }
    }
    input.assign(*sourceCount, UNASSIGNED);
    SmallVector<size_t> circuitSources;
    for (const auto& root : roots) {
      if (root.allocationIndex >= allocations.size()) {
        return func.emitError("resolved allocation identity is out of range");
      }
      auto labels =
          allocations[root.allocationIndex]->getAttrOfType<DenseI64ArrayAttr>(
              mqt::kSourceQubitIndicesAttr);
      if (!labels || std::cmp_greater_equal(root.elementIndex, labels.size())) {
        return func.emitError("entry allocation lost its source qubit labels");
      }
      const auto source = labels[root.elementIndex];
      if (source < 0 || std::cmp_greater_equal(source, *sourceCount) ||
          input[source] != UNASSIGNED) {
        return func.emitError(
            "source qubit labels must be distinct input slots");
      }
      input[source] = root.wire;
      circuitSources.push_back(static_cast<size_t>(source));
    }
    for (auto& wire : input) {
      if (wire == UNASSIGNED) {
        wire = fresh();
      }
    }
    output = input;
    if (layout) {
      if (layout->initial.size() != roots.size() || !layout->routing ||
          std::cmp_not_equal(layout->inputCount, roots.size()) ||
          !llvm::equal(
              layout->initial,
              llvm::seq<int64_t>(0, static_cast<int64_t>(roots.size())))) {
        return func.emitError(
            "source-tagged circuit layout must retain its wire order");
      }
      for (size_t wire = 0; wire < circuitSources.size(); ++wire) {
        output[circuitSources[wire]] =
            input[circuitSources[(*layout->routing)[wire]]];
      }
    }
    argumentSlots.assign(*sourceCount, false);
  } else if (layout) {
    if (hasArguments) {
      return func.emitError(
          "layout and quantum argument identities cannot be combined");
    }
    sourceCount = static_cast<size_t>(layout->inputCount);
    std::vector<size_t> positions(layout->initial.size(), UNASSIGNED);
    if (layout->sites) {
      std::map<uint64_t, size_t> positionForSite;
      for (auto [position, site] : llvm::enumerate(*layout->sites)) {
        positionForSite.emplace(static_cast<uint64_t>(site), position);
      }
      for (const auto& root : roots) {
        const auto found = positionForSite.find(root.staticId);
        if (root.kind != DDWireRoot::Kind::Static ||
            found == positionForSite.end() ||
            positions[found->second] != UNASSIGNED) {
          return func.emitError(
              "placed roots do not match the layout site identities");
        }
        positions[found->second] = root.wire;
      }
    } else {
      const bool staticRoots = llvm::any_of(roots, [](const auto& root) {
        return root.kind == DDWireRoot::Kind::Static;
      });
      if (!staticRoots && roots.size() != positions.size()) {
        return func.emitError(
            "circuit layout no longer matches allocation slots");
      }
      for (auto [index, root] : llvm::enumerate(roots)) {
        const auto position = staticRoots ? root.staticId : index;
        if ((staticRoots && root.kind != DDWireRoot::Kind::Static) ||
            position >= positions.size() || positions[position] != UNASSIGNED) {
          return func.emitError("circuit roots do not match layout positions");
        }
        positions[position] = root.wire;
      }
    }
    for (auto& wire : positions) {
      if (wire == UNASSIGNED) {
        wire = fresh();
      }
    }
    for (const auto initial : layout->initial) {
      input.push_back(positions[initial]);
      output.push_back(
          positions[layout->routing ? (*layout->routing)[initial] : initial]);
    }
    argumentSlots.assign(*sourceCount, false);
  } else {
    size_t staticWidth = 0;
    for (const auto& root : roots) {
      if (root.kind == DDWireRoot::Kind::Static) {
        if (root.staticId >= MAX_WIRES || hasArguments) {
          return func.emitError(
              "unsupported static input identity for verification");
        }
        staticWidth =
            std::max(staticWidth, static_cast<size_t>(root.staticId) + 1);
      }
    }
    input.assign(staticWidth, UNASSIGNED);
    argumentSlots.assign(staticWidth, false);
    for (const auto& root : roots) {
      if (root.kind == DDWireRoot::Kind::Static) {
        if (input[root.staticId] != UNASSIGNED) {
          return func.emitError("static input identities must be distinct");
        }
        input[root.staticId] = root.wire;
      } else {
        input.push_back(root.wire);
        argumentSlots.push_back(root.kind == DDWireRoot::Kind::Argument);
      }
    }
    for (auto& wire : input) {
      if (wire == UNASSIGNED) {
        wire = fresh();
      }
    }
    sourceCount = input.size();
    output = input;
  }

  if (nextWire > MAX_WIRES || input.size() != nextWire) {
    return func.emitError(
        "verification interface exceeds its resolved wire domain");
  }
  if ((expectedSourceCount && *expectedSourceCount != *sourceCount) ||
      (contract.numInputs && *contract.numInputs != *sourceCount)) {
    return func.emitError() << "source input count " << *sourceCount
                            << " does not match the reference interface";
  }

  VerificationInterface result;
  result.numInputs = *sourceCount;
  result.numQubits = nextWire;
  result.zeroInputs.assign(nextWire, true);
  for (size_t slot = 0; slot < result.numInputs; ++slot) {
    result.zeroInputs[slot] =
        contract.domain == VerificationDomain::InitializedProgram &&
        !argumentSlots[slot];
  }
  std::set<size_t> zeroSlots;
  for (const size_t slot : contract.zeroAncillas) {
    if (slot >= result.numInputs || !zeroSlots.insert(slot).second) {
      return func.emitError(
          "zero ancillas must name distinct source input slots");
    }
    result.zeroInputs[slot] = true;
  }

  if (!execution.quantumReturns.empty()) {
    const size_t freeInputs = static_cast<size_t>(std::count(
        result.zeroInputs.begin(),
        result.zeroInputs.begin() + static_cast<ptrdiff_t>(result.numInputs),
        false));
    const bool allSources = execution.quantumReturns.size() == result.numInputs;
    if (!allSources && execution.quantumReturns.size() != freeInputs) {
      return func.emitError(
          "quantum returns must cover every non-ancillary source output");
    }
    if (!layout) {
      output.assign(nextWire, UNASSIGNED);
    }
    seen.assign(nextWire, false);
    size_t returned = 0;
    for (size_t slot = 0; slot < result.numInputs; ++slot) {
      if (!allSources && result.zeroInputs[slot]) {
        result.cleanOutputs.push_back(static_cast<dd::Qubit>(slot));
        continue;
      }
      const auto wire = execution.quantumReturns[returned++];
      if (wire >= execution.numQubits || seen[wire]) {
        return func.emitError(
            "quantum output wires must be distinct resolved wires");
      }
      if (layout && output[slot] != wire) {
        return func.emitError(
            "quantum return identities disagree with layout outputs");
      }
      output[slot] = wire;
      seen[wire] = true;
    }
    auto remaining = input.begin();
    for (auto& wire : output) {
      if (wire == UNASSIGNED) {
        while (seen[*remaining]) {
          ++remaining;
        }
        wire = *remaining++;
        seen[wire] = true;
      }
    }
  }
  for (size_t slot = result.numInputs; slot < nextWire; ++slot) {
    result.cleanOutputs.push_back(static_cast<dd::Qubit>(slot));
  }
  for (auto wire : input) {
    result.inputWires.push_back(static_cast<dd::Qubit>(wire));
  }
  for (auto wire : output) {
    result.outputWires.push_back(static_cast<dd::Qubit>(wire));
  }

  std::map<std::string, DDOutputBit> labeled;
  std::set<std::pair<size_t, size_t>> destinations;
  for (const auto& bit : execution.outputs) {
    if (bit.wire.has_value() == bit.constant.has_value() ||
        (bit.wire && *bit.wire >= execution.numQubits) ||
        !destinations.emplace(bit.resultIndex, bit.bitIndex).second ||
        !labeled.emplace(outputLabel(bit), bit).second) {
      return func.emitError(
          "classical outputs need distinct destinations and defined values");
    }
  }
  if (contract.observedOutputs.empty()) {
    for (auto& [label, bit] : labeled) {
      result.outputLabels.push_back(label);
      result.outputs.push_back(std::move(bit));
    }
  } else {
    std::set<std::string> selected;
    for (const auto& label : contract.observedOutputs) {
      const auto found = labeled.find(label);
      if (found == labeled.end() || !selected.insert(label).second) {
        return func.emitError()
               << "selected output '" << label << "' is missing or duplicated";
      }
      result.outputLabels.push_back(label);
      result.outputs.push_back(found->second);
    }
  }
  return result;
}

} // namespace mlir::qco
