/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

namespace dd {

/**
 * @brief Base class for all DD nodes.
 * @details This class is used to store common information for all DD nodes.
 * The `flags` makes the implicit padding explicit and can be used for storing
 * node properties.
 * Data Layout (8)|(2|2|4) = 16B.
 */
struct NodeBase : LLBase {
  /// Variable index
  Qubit v{};

  /**
   * @brief Flags for node properties
   * @details Not required for all node types, but padding is required either
   * way.
   *
   * 0b1 = mark flag used for mark-and-sweep garbage collection
   */
  std::uint16_t flags = 0;

  /// Mark flag used for mark-and-sweep garbage collection
  static constexpr std::uint16_t MARK_FLAG = 0b1U;

  /// @brief Whether a node is marked as used.
  [[nodiscard]] bool isMarked() const noexcept {
    return (flags & MARK_FLAG) != 0U;
  }

  /// @brief Mark the node as used.
  void mark() noexcept { flags |= MARK_FLAG; }

  /// @brief Unmark the node.
  void unmark() noexcept { flags &= static_cast<uint16_t>(~MARK_FLAG); }

  /// Getter for the next object.
  [[nodiscard]] NodeBase* next() const noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
    return static_cast<NodeBase*>(next_);
  }

  /**
   * @brief Check if a node is terminal
   * @param p The node to check
   * @return true if the node is terminal, false otherwise.
   */
  [[nodiscard]] static constexpr bool isTerminal(const NodeBase* p) noexcept {
    return p == nullptr;
  }
  static constexpr NodeBase* getTerminal() noexcept { return nullptr; }
};

static_assert(sizeof(NodeBase) == 16);
static_assert(alignof(NodeBase) == 8);

/**
 * @brief A vector DD node
 * @details Data Layout (8)|(2|2|4)|(24|24) = 64B
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
 * @details Data Layout (8)|(2|2|4)|(24|24|24|24) = 112B
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

} // namespace dd
