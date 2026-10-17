/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Transforms/Mapping/Placement.h"

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace mlir::qco {

RegionConnectivityAnalysis::RegionConnectivityAnalysis(Operation* operation) {
  if (operation->getNumRegions() != 1) {
    operation->emitError(
        "connectivity analysis requires an operation with one region");
    return;
  }
  connectivity_ = buildConnectivityGraph(operation->getRegion(0));
}

FailureOr<RegionConnectivity> buildConnectivityGraph(Region& region) {
  if (!llvm::hasSingleElement(region)) {
    return emitError(region.getLoc(),
                     "connectivity requires a single-block region");
  }
  RegionConnectivity result;
  // collectWireOrigins does not track identities within tensors.
  DenseMap<Value, SmallVector<size_t>> wires;
  llvm::SetVector<std::pair<size_t, size_t>> edges;
  constexpr auto emptySlot = std::numeric_limits<size_t>::max();

  const auto addRoot = [&](Value value) -> LogicalResult {
    const bool scalar = isa<QubitType>(value.getType());
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!scalar && (!tensor || !tensor.hasStaticShape())) {
      return emitError(value.getLoc(),
                       "connectivity requires statically sized qubit tensors");
    }
    const int64_t size = scalar ? 1 : tensor.getNumElements();
    auto& indices = wires[value];
    for (int64_t i = 0; i < size; ++i) {
      indices.push_back(result.qubits.size());
      result.qubits.push_back(
          {.root = value, .slot = scalar ? std::nullopt : std::optional(i)});
    }
    return success();
  };
  const auto consume = [&](Value value) {
    auto indices = std::move(wires.at(value));
    wires.erase(value);
    return indices;
  };
  for (Value argument : quantumValues(region.getArguments())) {
    if (failed(addRoot(argument))) {
      return failure();
    }
  }
  struct Frame {
    Block::iterator next;
    Block::iterator end;
    Operation* pendingBranch = nullptr;
  };
  SmallVector<Frame> stack;
  stack.push_back({
      .next = region.front().begin(),
      .end = region.front().end(),
  });
  // Retain entry identities until all branch joins and backedges are checked.
  DenseMap<Value, SmallVector<size_t>> regionInputs;
  llvm::SmallPtrSet<Region*, 8> visited;
  const auto forward = [&](RegionBranchOpInterface branch,
                           RegionBranchPoint point) -> LogicalResult {
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(point, successors);
    for (auto successor : llvm::reverse(successors)) {
      auto inputs = branch.getSuccessorInputs(successor);
      auto operands = branch.getSuccessorOperands(point, successor);
      for (auto [operand, input] : llvm::zip_equal(operands, inputs)) {
        if (!isLinearQubitType(input.getType())) {
          continue;
        }
        auto known = wires.find(operand);
        if (known == wires.end()) {
          return branch.emitError("cannot identify a forwarded region qubit");
        }
        auto indices = known->second;
        auto& bindings = isa<BlockArgument>(input) ? regionInputs : wires;
        auto [it, inserted] = bindings.try_emplace(input, indices);
        if (!inserted && it->second != indices) {
          return branch.emitError(
              "connectivity requires consistent qubit and "
              "tensor-slot identities at region joins and backedges");
        }
        if (inserted && isa<BlockArgument>(input)) {
          wires[input] = std::move(indices);
        }
      }
      if (auto* body = successor.getSuccessor();
          body && visited.insert(body).second) {
        if (!llvm::hasSingleElement(*body)) {
          return branch.emitError("connectivity requires single-block regions");
        }
        if (llvm::any_of(quantumValues(body->getArguments()),
                         [&](Value value) { return !wires.contains(value); })) {
          return branch.emitError("cannot identify a region input qubit");
        }
        stack.push_back({
            .next = body->front().begin(),
            .end = body->front().end(),
        });
      }
    }
    return success();
  };
  while (!stack.empty()) {
    auto& frame = stack.back();
    if (auto* branch = std::exchange(frame.pendingBranch, nullptr)) {
      for (Value output : quantumValues(branch->getResults())) {
        if (!wires.contains(output)) {
          return branch->emitError("cannot identify a region output qubit");
        }
      }
      for (Region& body : branch->getRegions()) {
        for (Value argument : body.getArguments()) {
          regionInputs.erase(argument);
        }
        visited.erase(&body);
      }
    }
    if (frame.next == frame.end) {
      stack.pop_back();
      continue;
    }
    Operation& op = *frame.next++;
    if (isa<CallOp>(op)) {
      return op.emitError("inline quantum calls before connectivity analysis");
    }
    for (Value operand : quantumValues(op.getOperands())) {
      if (wires.contains(operand)) {
        continue;
      }
      if (operand.getParentRegion() == op.getParentRegion() ||
          failed(addRoot(operand))) {
        return op.emitError("cannot identify an input qubit for connectivity");
      }
    }
    if (auto unitary = dyn_cast<UnitaryOpInterface>(op)) {
      if (!isa<BarrierOp>(op)) {
        if (unitary.getNumQubits() > 2) {
          return op.emitError("decompose to one- and two-qubit gates before "
                              "connectivity analysis");
        }
        if (unitary.isTwoQubit()) {
          const auto a = wires.at(unitary.getInputQubit(0)).front();
          const auto b = wires.at(unitary.getInputQubit(1)).front();
          edges.insert(std::minmax(a, b));
        }
      }
      for (Value input : unitary.getInputQubits()) {
        wires[unitary.getOutputForInput(input)] = consume(input);
      }
    } else if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      frame.pendingBranch = &op;
      if (failed(forward(branch, RegionBranchPoint::parent()))) {
        return failure();
      }
      for (Value operand : quantumValues(op.getOperands())) {
        wires.erase(operand);
      }
    } else if (isa<AllocOp, StaticOp, qtensor::AllocOp>(op)) {
      if (failed(addRoot(op.getResult(0)))) {
        return failure();
      }
    } else if (auto measure = dyn_cast<MeasureOp>(op)) {
      wires[measure.getQubitOut()] = consume(measure.getQubitIn());
    } else if (auto reset = dyn_cast<ResetOp>(op)) {
      wires[reset.getQubitOut()] = consume(reset.getQubitIn());
    } else if (auto tensor = dyn_cast<qtensor::FromElementsOp>(op)) {
      SmallVector<size_t> indices;
      for (Value qubit : tensor.getElements()) {
        indices.push_back(consume(qubit).front());
      }
      wires[tensor.getResult()] = std::move(indices);
    } else if (isa<qtensor::ExtractOp, qtensor::InsertOp>(op)) {
      auto extract = dyn_cast<qtensor::ExtractOp>(op);
      auto insert = dyn_cast<qtensor::InsertOp>(op);
      Value input = extract ? extract.getTensor() : insert.getDest();
      const auto index =
          getConstantIntValue(extract ? extract.getIndex() : insert.getIndex());
      auto indices = consume(input);
      if (!index || *index < 0 ||
          std::cmp_greater_equal(*index, indices.size())) {
        return op.emitError(
            "connectivity requires an in-bounds constant tensor index");
      }
      auto& slot = indices[static_cast<size_t>(*index)];
      if (extract) {
        if (slot == emptySlot) {
          return op.emitError("cannot extract an empty slot for connectivity");
        }
        wires[extract.getResult()] = {std::exchange(slot, emptySlot)};
        wires[extract.getOutTensor()] = std::move(indices);
      } else {
        if (slot != emptySlot) {
          return op.emitError(
              "cannot insert into an occupied slot for connectivity");
        }
        slot = consume(insert.getScalar()).front();
        wires[insert.getResult()] = std::move(indices);
      }
    } else if (isa<SinkOp, qtensor::DeallocOp>(op) ||
               op.hasTrait<OpTrait::IsTerminator>()) {
      if (op.hasTrait<OpTrait::IsTerminator>() && stack.size() > 1) {
        auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op);
        if (!terminator) {
          return op.emitError(
              "connectivity requires a region branch terminator");
        }
        if (failed(forward(cast<RegionBranchOpInterface>(op.getParentOp()),
                           terminator))) {
          return failure();
        }
      }
      for (Value operand : op.getOperands()) {
        wires.erase(operand);
      }
    } else if (op.getNumRegions() != 0 ||
               llvm::any_of(op.getOperandTypes(), isLinearQubitType) ||
               llvm::any_of(op.getResultTypes(), isLinearQubitType)) {
      return op.emitError(
          "unsupported connectivity operation; lower unsupported "
          "regions and inline quantum calls");
    }
  }
  result.graph =
      Graph(llvm::to_vector(llvm::seq<size_t>(0, result.qubits.size())));
  for (const auto& [a, b] : edges) {
    result.graph.addEdge(a, b);
    result.graph.addEdge(b, a);
  }
  return result;
}

namespace {
using Pattern = CompilerTarget::TopologyPatterns::Type;
using PatternGroup = CompilerTarget::TopologyPatterns::Group;
} // namespace

std::optional<SmallVector<size_t, 4>>
matchTopologyPattern(const Graph& graph, const PatternGroup& pattern) {
  const auto arity = pattern.arity();
  if (graph.getNumNodes() < arity) {
    return std::nullopt;
  }
  const auto nodes = graph.getNodes();
  SmallVector<size_t, 4> match;
  // All templates have a connected prefix, so extend along its frontier.
  const auto search = [&](auto&& self) -> bool {
    const auto depth = match.size();
    if (depth == arity) {
      return true;
    }
    ArrayRef<size_t> candidateRange = nodes;
    const auto neighbours = pattern.neighbours(depth);
    for (const auto previous : neighbours) {
      if (previous < depth) {
        candidateRange = graph.getNeighbours(match[previous]);
        break;
      }
    }
    auto candidates = llvm::to_vector(candidateRange);
    llvm::sort(candidates);
    for (const auto candidate : candidates) {
      if (llvm::is_contained(match, candidate) ||
          graph.getDegree(candidate) < neighbours.size()) {
        continue;
      }
      if (llvm::any_of(neighbours, [&](size_t other) {
            return other < depth && !graph.hasEdge(candidate, match[other]);
          })) {
        continue;
      }
      size_t patternFrontier = 0;
      size_t patternRemaining = 0;
      for (const auto next : neighbours) {
        if (next > depth) {
          ++patternRemaining;
          patternFrontier += llvm::any_of(pattern.neighbours(next),
                                          [&](size_t n) { return n < depth; });
        }
      }
      size_t graphFrontier = 0;
      size_t graphRemaining = 0;
      for (const auto next : graph.getNeighbours(candidate)) {
        if (!llvm::is_contained(match, next)) {
          ++graphRemaining;
          graphFrontier += llvm::any_of(
              match, [&](size_t n) { return graph.hasEdge(next, n); });
        }
      }
      // Non-induced matches may map new pattern vertices into the graph
      // frontier.
      if (patternFrontier > graphFrontier ||
          patternRemaining > graphRemaining) {
        continue;
      }
      match.push_back(candidate);
      if (self(self)) {
        return true;
      }
      match.pop_back();
    }
    return false;
  };
  if (search(search)) {
    return match;
  }
  return std::nullopt;
}

FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(Graph graph, ArrayRef<PatternGroup> available) {
  SmallVector<TopologyPatternMatch> slices;
  // Removing edges cannot create a new non-induced match of an exhausted type.
  for (const auto type :
       {Pattern::FourCycle, Pattern::Star, Pattern::QubitPair}) {
    const auto* pattern = llvm::find_if(available, [type](const auto& group) {
      return group.type() == type && group.size() != 0;
    });
    if (pattern == available.end()) {
      continue;
    }
    while (auto match = matchTopologyPattern(graph, *pattern)) {
      for (const auto& [a, b] : pattern->edges()) {
        graph.removeEdge((*match)[a], (*match)[b]);
        graph.removeEdge((*match)[b], (*match)[a]);
      }
      slices.push_back({.type = type, .vertices = std::move(*match)});
    }
  }
  if (llvm::any_of(graph.getNodes(),
                   [&](size_t n) { return graph.getDegree(n) != 0; })) {
    return failure();
  }
  return slices;
}

} // namespace mlir::qco
