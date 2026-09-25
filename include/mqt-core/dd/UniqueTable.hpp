/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file UniqueTable.hpp
/// Data structure for uniquely storing DD nodes

#pragma once

#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/statistics/UniqueTableStatistics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <ranges>
#include <type_traits>
#include <vector>

namespace dd {

/// Data structure for uniquely storing DD nodes
class UniqueTable {
public:
  /// The initial garbage collection limit.
  ///
  /// The initial garbage collection limit is the number of entries that
  /// must be present in the table before garbage collection is triggered.
  /// Increasing this number reduces the number of garbage collections, but
  /// increases the memory usage.
  static constexpr std::size_t INITIAL_GC_LIMIT = 131072U;

  struct UniqueTableConfig {
    /// The number of variables
    std::size_t nVars = 0U;

    /// Initial buckets per level (must be a power of two).
    std::size_t nBuckets = 64U;

    /// The initial garbage collection limit
    std::size_t initialGCLimit = INITIAL_GC_LIMIT;

    /// Per-level bucket ceiling; must be a power of two and at least nBuckets.
    /// Set equal to nBuckets for fixed sizing. clear() retains grown
    /// capacities.
    size_t maxBuckets = 1048576U;
  };

  /// Validate capacities and create a table. The manager must allocate the
  /// node type used in lookup and outlive the table.
  [[nodiscard]] static mlir::FailureOr<UniqueTable>
  create(MemoryManager& manager, const UniqueTableConfig& config);

  void resize(std::size_t nVars);

  /// The hash function for the hash table.
  ///
  /// The hash function just combines the hashes of the edges of the
  /// node. The hash value is masked to ensure that it is in the range
  /// [0, number of buckets at p.v - 1].
  /// @pre p.v names an allocated level.
  /// @param p The node to hash.
  /// @returns The hash value of the node.
  template <class Node> [[nodiscard]] std::size_t hash(const Node& p) const {
    static_assert(std::is_base_of_v<NodeBase, Node>,
                  "Node must be derived from NodeBase");
    const auto mask = tables[p.v].size() - 1;
    std::size_t key = 0U;
    for (const auto& succ : p.e) {
      hashCombine(key, std::hash<Edge<Node>>{}(succ));
    }
    key &= mask;
    return key;
  }

  template <class Node>
  [[nodiscard]] static bool nodesAreEqual(const Node& p, const Node& q) {
    return p.e == q.e;
  }

  // Lookup a node in the unique table for the appropriate variable and insert
  // it if it has not been found. Only normalized nodes shall be stored.
  template <class Node> [[nodiscard]] Node* lookup(Node* p) {
    static_assert(std::is_base_of_v<NodeBase, Node>,
                  "Node must be derived from NodeBase");
    // there are unique terminal nodes
    if (NodeBase::isTerminal(p)) {
      return p;
    }

    auto key = hash(*p);
    const auto v = p->v;
    ++stats[v].lookups;

    // search bucket in table corresponding to hashed value for the given node
    // and return it if found.
    if (auto* hashedNode = searchTable(*p, key);
        !Node::isTerminal(hashedNode)) {
      return hashedNode;
    }

    /// Grow only this populated level; node addresses and roots stay valid.
    if (stats[v].numEntries >= tables[v].size() &&
        tables[v].size() < cfg.maxBuckets) {
      grow<Node>(v);
      key = hash(*p);
    }

    // if node not found → add it to front of unique table bucket
    p->setNext(tables[v][key]);
    tables[v][key] = p;
    stats[v].trackInsert();
    ++entryCount_;

    return p;
  }

  /// Get a reference to the tables.
  /// Bucket storage may change after insertion or resize; node addresses stay
  /// valid.
  [[nodiscard]] const auto& getTables() const { return tables; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Get a reference to individual statistics
  [[nodiscard]] const UniqueTableStatistics&
  getStats(std::size_t idx) const noexcept;

  /// Get the total number of entries
  [[nodiscard]] std::size_t getNumEntries() const noexcept;

  /// Count the number of marked entries
  [[nodiscard]] std::size_t countMarkedEntries() const noexcept;

  /// Determine whether the table possibly requires garbage collection.
  [[nodiscard]] bool possiblyNeedsCollection() const;

  std::size_t garbageCollect(bool force = false);

  void clear();

  template <class Node> void print() const {
    static_assert(std::is_base_of_v<NodeBase, Node>,
                  "Node must be derived from NodeBase");
    auto q = cfg.nVars - 1U;
    for (const auto& table : std::ranges::reverse_view(tables)) {
      std::cout << "\tq" << q << ":" << "\n";
      for (std::size_t key = 0; key < table.size(); ++key) {
        const auto* p = static_cast<Node*>(table[key]);
        if (p != nullptr) {
          std::cout << "\tkey=" << key << ": ";
        }

        while (p != nullptr) {
          std::cout << "\t\t" << std::hex
                    << reinterpret_cast<std::uintptr_t>(p);
          for (const auto& e : p->e) {
            std::cout << " p" << reinterpret_cast<std::uintptr_t>(e.p) << "(r"
                      << reinterpret_cast<std::uintptr_t>(e.w.r) << " i"
                      << reinterpret_cast<std::uintptr_t>(e.w.i) << ")";
          }
          std::cout << std::dec << "\n";
          p = p->next();
        }
      }
      --q;
    }
  }

private:
  friend class Package;
  UniqueTable(MemoryManager& manager, const UniqueTableConfig& config);
  [[nodiscard]] static mlir::LogicalResult checkCapacity(size_t initial,
                                                         size_t maximum);

  /// Typedef for a bucket in the table
  using Bucket = NodeBase*;
  /// Typedef for the table
  using Table = std::vector<Bucket>;

  UniqueTableConfig cfg;

  /// The current garbage collection limit
  std::size_t gcLimit;

  /// A pointer to the memory manager for the nodes stored in the table.
  MemoryManager* memoryManager;

  /// The actual tables (one for each variable)
  ///
  /// Each hash table is an array of buckets. Each bucket is a linked
  /// list of entries. The linked list is implemented by using the next pointer
  /// of the entries.
  std::vector<Table> tables;

  /// A collection of statistics
  std::vector<UniqueTableStatistics> stats;

  /// Total entries across all levels, used by per-operation collection checks.
  std::size_t entryCount_ = 0U;

  template <class Node> void grow(const size_t v) {
    Table old(tables[v].size() * 2U);
    tables[v].swap(old);
    for (auto* bucket : old) {
      auto* node = static_cast<Node*>(bucket);
      while (node != nullptr) {
        auto* next = node->next();
        const auto key = hash(*node);
        node->setNext(tables[v][key]);
        tables[v][key] = node;
        node = next;
      }
    }
    stats[v].numBuckets = tables[v].size();
  }

  /// Search for a node in the hash table with the given key.
  /// @param p The node to search for.
  /// @param key The hashed value used to search the table.
  /// @returns A pointer to the node if found or Node::getTerminal() otherwise.
  template <class Node>
  [[nodiscard]] Node* searchTable(Node& p, const std::size_t& key) {
    static_assert(std::is_base_of_v<NodeBase, Node>,
                  "Node must be derived from NodeBase");
    const auto v = p.v;
    Node* bucket = static_cast<Node*>(tables[v][key]);
    while (bucket != nullptr) {
      if (nodesAreEqual(p, *bucket)) {
        // Match found
        if (&p != bucket) {
          // put node pointed to by p on available chain
          memoryManager->returnEntry(p);
        }
        ++stats[v].hits;
        return bucket;
      }
      ++stats[v].collisions;
      bucket = bucket->next();
    }

    // Node not found in bucket
    return Node::getTerminal();
  }
};

} // namespace dd
