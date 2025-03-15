/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/CachedEdge.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/LinkedListBase.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

namespace dd {

/**
 * @brief Base class for all DD nodes
 * @details This class is used to store common information for all DD nodes. The
 * `flags` field is an exception, but it allows to reuse functionality for
 * different node types. Data Layout |8|4|2|1| = 15B (space for one more byte)
 */
struct NodeBase : public LLBase {
  RefCount ref = 0; // reference count
  Qubit v{};        // variable index

  std::uint8_t flags =
      0; // not required for all node types, but padding is required either way
  // 32 = unused (was used to mark a node which is symmetric)
  // 16 = unused (was used to mark a node resembling the identity)
  // 8 = marks a reduced dm node,
  // 4 = marks a dm (tmp flag),
  // 2 = mark first path edge (tmp flag),
  // 1 = mark path is conjugated (tmp flag)

  NodeBase* next() const noexcept {
    return static_cast<NodeBase*>(LLBase::next());
  }

  static bool isTerminal(const NodeBase* p) noexcept {
    // some nodes (dNode) use bottom bits to store flags. As all nodes are >= 8
    // byte aligned, so these bits can be ignored in all cases
    return (reinterpret_cast<std::uintptr_t>(p) & (~7ULL)) == 0ULL;
  }
  static constexpr NodeBase* getTerminal() noexcept { return nullptr; }
};

/**
 * @brief A vector DD node
 * @details Data Layout 8|4|2|1|1 (padding)|24|24 = 64B
 */
struct vNode : public NodeBase {      // NOLINT(readability-identifier-naming)
  std::array<Edge<vNode>, RADIX> e{}; // edges out of this node

  vNode* next() const noexcept { return static_cast<vNode*>(NodeBase::next()); }

  static constexpr vNode* getTerminal() noexcept { return nullptr; }
};
using vEdge = Edge<vNode>;
using vCachedEdge = CachedEdge<vNode>;

/**
 * @brief A matrix DD node
 * @details Data Layout 8|4|2|1|1 (padding)|24|24|24|24|= 112B
 */
struct mNode : public NodeBase {      // NOLINT(readability-identifier-naming)
  std::array<Edge<mNode>, NEDGE> e{}; // edges out of this node

  mNode* next() const noexcept { return static_cast<mNode*>(NodeBase::next()); }

  static constexpr mNode* getTerminal() noexcept { return nullptr; }
};
using mEdge = Edge<mNode>;
using mCachedEdge = CachedEdge<mNode>;

/**
 * @brief A density matrix DD node
 * @details Data Layout 8|4|2|1|1 (padding)|24|24|24|24|8|4|2|1| = 112B
 */
struct dNode : public NodeBase {      // NOLINT(readability-identifier-naming)
  std::array<Edge<dNode>, NEDGE> e{}; // edges out of this node

  dNode* next() const noexcept { return static_cast<dNode*>(NodeBase::next()); }

  static constexpr dNode* getTerminal() noexcept { return nullptr; }

  [[nodiscard]] [[maybe_unused]] static constexpr bool
  tempDensityMatrixFlagsEqual(const std::uint8_t a,
                              const std::uint8_t b) noexcept {
    return getDensityMatrixTempFlags(a) == getDensityMatrixTempFlags(b);
  }

  [[nodiscard]] static constexpr bool
  isConjugateTempFlagSet(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 0)) != 0U;
  }
  [[nodiscard]] static constexpr bool
  isNonReduceTempFlagSet(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 1)) != 0U;
  }
  [[nodiscard]] static constexpr bool
  isDensityMatrixTempFlagSet(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 2)) != 0U;
  }
  [[nodiscard]] static bool
  isDensityMatrixNode(const std::uintptr_t p) noexcept {
    return (p & (1ULL << 3)) != 0U;
  }

  [[nodiscard]] static bool isConjugateTempFlagSet(const dNode* p) noexcept {
    return isConjugateTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static bool isNonReduceTempFlagSet(const dNode* p) noexcept {
    return isNonReduceTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static bool
  isDensityMatrixTempFlagSet(const dNode* p) noexcept {
    return isDensityMatrixTempFlagSet(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static bool isDensityMatrixNode(const dNode* p) noexcept {
    return isDensityMatrixNode(reinterpret_cast<std::uintptr_t>(p));
  }

  static void setConjugateTempFlagTrue(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 0));
  }
  static void setNonReduceTempFlagTrue(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 1));
  }
  static void setDensityMatTempFlagTrue(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) |
                                 (1ULL << 2));
  }
  static void alignDensityNode(dNode*& p) noexcept {
    p = reinterpret_cast<dNode*>(reinterpret_cast<std::uintptr_t>(p) & (~7ULL));
  }

  [[nodiscard]] static std::uintptr_t
  getDensityMatrixTempFlags(dNode*& p) noexcept {
    return getDensityMatrixTempFlags(reinterpret_cast<std::uintptr_t>(p));
  }
  [[nodiscard]] static constexpr std::uintptr_t
  getDensityMatrixTempFlags(const std::uintptr_t a) noexcept {
    return a & (7ULL);
  }

  constexpr void unsetTempDensityMatrixFlags() noexcept {
    flags = flags & static_cast<std::uint8_t>(~7U);
  }

  void setDensityMatrixNodeFlag(bool densityMatrix) noexcept;

  static std::uint8_t alignDensityNodeNode(dNode*& p) noexcept;

  static void getAlignedNodeRevertModificationsOnSubEdges(dNode* p) noexcept;

  static void applyDmChangesToNode(dNode*& p) noexcept;

  static void revertDmChangesToNode(dNode*& p) noexcept;
};
using dEdge = Edge<dNode>;
using dCachedEdge = CachedEdge<dNode>;

static inline dEdge densityFromMatrixEdge(const mEdge& e) {
  return dEdge{reinterpret_cast<dNode*>(e.p), e.w};
}

/**
 * @brief Increment the reference count of a node.
 * @details This function increments the reference count of a node. If the
 * reference count has saturated (i.e. reached the maximum value of RefCount)
 * the reference count is not incremented.
 * @param p A pointer to the node to increment the reference count of.
 * @returns Whether the reference count was incremented.
 * @note Typically, you do not want to call this function directly. Instead,
 * use the UniqueTable::incRef(Node*) function.
 */
[[nodiscard]] static inline bool incRef(NodeBase* p) noexcept {
  if (p == nullptr || p->ref == std::numeric_limits<RefCount>::max()) {
    return false;
  }
  ++p->ref;
  return true;
}

/**
 * @brief Decrement the reference count of a node.
 * @details This function decrements the reference count of a node. If the
 * reference count has saturated (i.e. reached the maximum value of RefCount)
 * the reference count is not decremented.
 * @param p A pointer to the node to decrement the reference count of.
 * @returns Whether the reference count was decremented.
 * @note Typically, you do not want to call this function directly. Instead,
 * use the UniqueTable::decRef(Node*) function.
 */
[[nodiscard]] static inline bool decRef(NodeBase* p) noexcept {
  if (p == nullptr || p->ref == std::numeric_limits<RefCount>::max()) {
    return false;
  }
  assert(p->ref != 0 &&
         "Reference count of Node must not be zero before decrement");
  --p->ref;
  return true;
}

} // namespace dd
