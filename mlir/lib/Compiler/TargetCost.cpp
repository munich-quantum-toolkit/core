/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetCost.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <utility>

namespace mlir {
namespace {

struct GateSignature {
  llvm::StringLiteral name;
  size_t numParameters;
};

} // namespace

[[nodiscard]] static GateSignature
gateSignature(const CompilerTarget::GateKind gate) {
  using Gate = CompilerTarget::GateKind;
  constexpr std::array signatures{
      std::pair{Gate::RXX, GateSignature{"rxx", 1}},
      std::pair{Gate::RYY, GateSignature{"ryy", 1}},
      std::pair{Gate::RZX, GateSignature{"rzx", 1}},
      std::pair{Gate::RZZ, GateSignature{"rzz", 1}},
      std::pair{Gate::ISWAP, GateSignature{"iswap", 0}},
      std::pair{Gate::CZ, GateSignature{"cz", 0}},
      std::pair{Gate::CX, GateSignature{"cx", 0}},
      std::pair{Gate::ECR, GateSignature{"ecr", 0}},
  };
  const llvm::ArrayRef<std::pair<Gate, GateSignature>> signatureList{
      signatures};
  const auto* const signature =
      llvm::find_if(signatureList, [gate](const auto& candidate) {
        return candidate.first == gate;
      });
  assert(signature != signatureList.end() &&
         "routing costs require a two-qubit gate");
  return signature->second;
}

TargetGateCosts::TargetGateCosts(const CompilerTarget& target,
                                 const CompilerTarget::GateKind gate)
    : target_(target) {
  constexpr auto unavailable = std::numeric_limits<float>::infinity();
  if (!target_.hasExplicitTopology()) {
    if (!target_.hasExplicitOperations()) {
      return;
    }

    const auto [name, numParameters] = gateSignature(gate);
    defaultAdjacentCost_ = unavailable;
    uniform_ = false;
    for (const auto& operation : target_.operations()) {
      if (operation.canonicalName() != name || operation.numQubits() != 2 ||
          operation.numParameters() != numParameters) {
        continue;
      }
      if (!operation.hasExplicitSiteTuples()) {
        defaultAdjacentCost_ = 0.F;
        costs_.clear();
        uniform_ = true;
        return;
      }
      for (const auto& tuple : operation.siteTuples()) {
        assert(tuple.sites().size() == 2 &&
               "two-qubit gate must have two-site tuples");
        const auto source = tuple.sites()[0];
        const auto target = tuple.sites()[1];
        costs_.insert_or_assign({source, target}, 0.F);
        const std::array reverseSites{target, source};
        costs_.insert_or_assign(
            {target, source}, target_.supports(gate, reverseSites) ? 0.F : 1.F);
      }
    }
    const auto numQubits = target_.numQubits();
    if (costs_.size() == numQubits * (numQubits - 1) &&
        llvm::all_of(costs_,
                     [](const auto& cost) { return cost.second == 0.F; })) {
      defaultAdjacentCost_ = 0.F;
      costs_.clear();
      uniform_ = true;
    }
    return;
  }

  for (size_t source = 0; source < target_.numQubits(); ++source) {
    target_.forEachNeighbour(source, [&](const size_t target) {
      if (target < source) {
        return;
      }

      const auto sourceSite = target_.siteForVertex(source);
      const auto targetSite = target_.siteForVertex(target);
      const std::array forwardSites{sourceSite, targetSite};
      const std::array reverseSites{targetSite, sourceSite};
      const bool forward = target_.supports(gate, forwardSites);
      const bool reverse = target_.supports(gate, reverseSites);

      if (!forward) {
        costs_.try_emplace(CompilerTarget::Coupling{sourceSite, targetSite},
                           reverse ? 1.F : unavailable);
      }
      if (!reverse) {
        costs_.try_emplace(CompilerTarget::Coupling{targetSite, sourceSite},
                           forward ? 1.F : unavailable);
      }
    });
  }
  uniform_ = costs_.empty();
}

float TargetGateCosts::routingCostBetween(const size_t source,
                                          const size_t target) const {
  assert(source < target_.numQubits() && target < target_.numQubits() &&
         "compiler target vertex is out of range");
  if (source == target) {
    return 0.F;
  }
  const auto distance = target_.distanceBetween(source, target);
  if (distance > 1) {
    return static_cast<float>(distance - 1);
  }
  const CompilerTarget::Coupling coupling{target_.siteForVertex(source),
                                          target_.siteForVertex(target)};
  if (const auto found = costs_.find(coupling); found != costs_.end()) {
    return found->second;
  }
  return defaultAdjacentCost_;
}

bool TargetGateCosts::isUniform() const noexcept { return uniform_; }

} // namespace mlir
