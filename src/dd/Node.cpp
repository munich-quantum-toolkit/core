/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Node.hpp"

#include "dd/ComplexNumbers.hpp"

#include <cassert>
#include <cstdint>
#include <utility>

namespace dd {

void dNode::setDensityMatrixNodeFlag(const bool densityMatrix) noexcept {
  if (densityMatrix) {
    flags = (flags | static_cast<std::uint8_t>(8U));
  } else {
    flags = (flags & static_cast<std::uint8_t>(~8U));
  }
}

std::uint8_t dNode::alignDensityNodeNode(dNode*& p) noexcept {
  const auto flags = static_cast<std::uint8_t>(getDensityMatrixTempFlags(p));
  // Get an aligned node
  alignDensityNode(p);

  if (dNode::isTerminal(p)) {
    return 0U;
  }

  if (isNonReduceTempFlagSet(flags) && !isConjugateTempFlagSet(flags)) {
    // nothing more to do for first edge path (inherited by all child paths)
    return flags;
  }

  if (!isConjugateTempFlagSet(flags)) {
    p->e[2].w = ComplexNumbers::conj(p->e[2].w);
    setConjugateTempFlagTrue(p->e[2].p);
    // Mark the first edge
    setNonReduceTempFlagTrue(p->e[1].p);

    for (auto& edge : p->e) {
      setDensityMatTempFlagTrue(edge.p);
    }

  } else {
    std::swap(p->e[2], p->e[1]);
    for (auto& edge : p->e) {
      edge.w = ComplexNumbers::conj(edge.w);
      setConjugateTempFlagTrue(edge.p);
      setDensityMatTempFlagTrue(edge.p);
    }
  }
  return flags;
}

void dNode::getAlignedNodeRevertModificationsOnSubEdges(dNode* p) noexcept {
  // Get an aligned node and revert the modifications on the sub edges
  alignDensityNode(p);

  for (auto& edge : p->e) {
    // remove the set properties from the node pointers of edge.p->e
    alignDensityNode(edge.p);
  }

  if (isNonReduceTempFlagSet(p->flags) && !isConjugateTempFlagSet(p->flags)) {
    // nothing more to do for a first edge path
    return;
  }

  if (!isConjugateTempFlagSet(p->flags)) {
    p->e[2].w = ComplexNumbers::conj(p->e[2].w);
    return;
  }
  for (auto& edge : p->e) {
    edge.w = ComplexNumbers::conj(edge.w);
  }
  std::swap(p->e[2], p->e[1]);
}

void dNode::applyDmChangesToNode(dNode*& p) noexcept {
  if (isDensityMatrixTempFlagSet(p)) {
    const auto tmp = alignDensityNodeNode(p);
    if (p == nullptr) {
      return;
    }
    assert(getDensityMatrixTempFlags(p->flags) == 0);
    p->flags = p->flags | tmp;
  }
}

void dNode::revertDmChangesToNode(dNode*& p) noexcept {
  if (!dNode::isTerminal(p) && isDensityMatrixTempFlagSet(p->flags)) {
    getAlignedNodeRevertModificationsOnSubEdges(p);
    p->unsetTempDensityMatrixFlags();
  }
}

} // namespace dd
