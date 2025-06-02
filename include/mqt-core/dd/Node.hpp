/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
 * @brief Base class for all DD nodes.
 * @details This class is used to store common information for all DD nodes.
 * The `flags` makes the implicit padding explicit and can be used for storing
 * node properties.
 * Data Layout (8)|(4|2|2) = 16B.
 */
struct NodeBase : LLBase {
  /// Variable index
  Qubit v{};

  /**
   * @brief Flags for node properties
   * @details Not required for all node types, but padding is required either
   * way.
   *
   * 0b10000 = mark flag used for mark-and-sweep garbage collection,
   *  0b1000 = marks a reduced dm node,
   *   0b100 = marks a dm (tmp flag),
   *    0b10 = mark first path edge (tmp flag),
   *     0b1 = mark path is conjugated (tmp flag)
   */
  std::uint16_t flags = 0;

  /// Mark flag used for mark-and-sweep garbage collection
  static constexpr std::uint16_t MARK_FLAG = 0b10000U;

  [[nodiscard]] bool marked() const noexcept {
    return (flags & MARK_FLAG) != 0U;
  }
  void mark() noexcept { flags |= MARK_FLAG; }
  void unmark() noexcept { flags &= static_cast<std::uint16_t>(~MARK_FLAG); }

  /// Getter for the next object.
  [[nodiscard]] NodeBase* next() const noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
    return static_cast<NodeBase*>(next_);
  }

  /**
   * @brief Check if a node is terminal
   * @details Generally, a node is terminal if it is nullptr.
   * Some nodes (dNode) encode additional information in the least significant
   * bits of the pointer. These bits are masked out before checking for
   * terminal nodes.
   * @param p The node to check
   * @return true if the node is terminal, false otherwise.
   */
  [[nodiscard]] static bool isTerminal(const NodeBase* p) noexcept {
    return (reinterpret_cast<std::uintptr_t>(p) & (~7ULL)) == 0ULL;
  }
  static constexpr NodeBase* getTerminal() noexcept { return nullptr; }
};

/**
 * @brief A vector DD node
 * @details Data Layout (8)|(4|2|2)|(24|24) = 64B
 */
struct vNode final : NodeBase {       // NOLINT(readability-identifier-naming)
  std::array<Edge<vNode>, RADIX> e{}; // edges out of this node

  /// Getter for the next object
  [[nodiscard]] vNode* next() const noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
    return static_cast<vNode*>(next_);
  }
  /// Getter for the terminal object
  static constexpr vNode* getTerminal() noexcept { return nullptr; }
};
using vEdge = Edge<vNode>;
using vCachedEdge = CachedEdge<vNode>;
using VectorDD = vEdge;

/**
 * @brief A matrix DD node
 * @details Data Layout (8)|(4|2|2)|(24|24|24|24) = 112B
 */
struct mNode final : NodeBase {       // NOLINT(readability-identifier-naming)
  std::array<Edge<mNode>, NEDGE> e{}; // edges out of this node

  /// Getter for the next object
  [[nodiscard]] mNode* next() const noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
    return static_cast<mNode*>(next_);
  }
  /// Getter for the terminal object
  static constexpr mNode* getTerminal() noexcept { return nullptr; }
};
using mEdge = Edge<mNode>;
using mCachedEdge = CachedEdge<mNode>;
using MatrixDD = mEdge;

/**
 * @brief A density matrix DD node
 * @details Data Layout (8)|(4|2|2)|(24|24|24|24) = 112B
 */
struct dNode final : NodeBase {       // NOLINT(readability-identifier-naming)
  std::array<Edge<dNode>, NEDGE> e{}; // edges out of this node

  /// Getter for the next object
  [[nodiscard]] dNode* next() const noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
    return static_cast<dNode*>(next_);
  }
  /// Getter for the terminal object
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
using DensityMatrixDD = dEdge;

static inline dEdge densityFromMatrixEdge(const mEdge& e) {
  return dEdge{reinterpret_cast<dNode*>(e.p), e.w};
}

} // namespace dd
