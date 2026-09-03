/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/MappingTarget.h"

#include <array>
#include <cstddef>

namespace mlir {

[[nodiscard]] static constexpr bool
isSwapInvariant(CompilerTarget::GateKind gate) {
  using Gate = CompilerTarget::GateKind;
  switch (gate) {
  case Gate::CZ:
  case Gate::ISWAP:
  case Gate::RXX:
  case Gate::RYY:
  case Gate::RZZ:
    return true;
  default:
    return false;
  }
}

MappingTarget::MappingTarget(const CompilerTarget& target) : target_(target) {
  const auto basis = target_.synthesisBasis();
  if (!basis || isSwapInvariant(basis->entangler)) {
    return;
  }

  for (size_t source = 0; source < target_.numSites(); ++source) {
    target_.forEachNeighbour(source, [&](size_t targetVertex) {
      if (targetVertex < source) {
        return;
      }

      const auto sourceSite = target_.siteForVertex(source);
      const auto targetSite = target_.siteForVertex(targetVertex);
      const std::array forwardSites{sourceSite, targetSite};
      const std::array reverseSites{targetSite, sourceSite};
      const bool forward = target_.supports(basis->entangler, forwardSites);
      const bool reverse = target_.supports(basis->entangler, reverseSites);

      if (!forward) {
        penalizedDirections_.insert(
            CompilerTarget::Coupling{sourceSite, targetSite});
      }
      if (!reverse) {
        penalizedDirections_.insert(
            CompilerTarget::Coupling{targetSite, sourceSite});
      }
    });
  }
}

const CompilerTarget& MappingTarget::compilerTarget() const noexcept {
  return target_;
}

size_t MappingTarget::numSites() const noexcept { return target_.numSites(); }

size_t MappingTarget::maxDegree() const noexcept { return target_.maxDegree(); }

size_t MappingTarget::distanceBetween(size_t source, size_t target) const {
  return target_.distanceBetween(source, target);
}

void MappingTarget::forEachNeighbour(
    size_t vertex, llvm::function_ref<void(size_t)> callback) const {
  target_.forEachNeighbour(vertex, callback);
}

float MappingTarget::pathCostBetween(size_t source, size_t target) const {
  if (source == target) {
    return 0.F;
  }
  const auto distance = target_.distanceBetween(source, target);
  if (distance > 1) {
    return static_cast<float>(distance - 1);
  }

  const CompilerTarget::Coupling coupling{target_.siteForVertex(source),
                                          target_.siteForVertex(target)};
  return penalizedDirections_.contains(coupling) ? 1.F : 0.F;
}

bool MappingTarget::isExecutable(size_t source, size_t target) const {
  return source != target && pathCostBetween(source, target) == 0.F;
}

} // namespace mlir
