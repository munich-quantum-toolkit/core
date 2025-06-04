/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file UniqueTable.hpp
 * @brief Data structure for uniquely storing DD nodes
 */

#pragma once

#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/statistics/UniqueTableStatistics.hpp"
#include "ir/Definitions.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <vector>

namespace dd {

/**
 * @brief Data structure for uniquely storing DD nodes
 */
class UniqueTable {
public:
  /**
   * @brief The initial garbage collection limit.
   * @details The initial garbage collection limit is the number of entries that
   * must be present in the table before garbage collection is triggered.
   * Increasing this number reduces the number of garbage collections, but
   * increases the memory usage.
   */
  static constexpr std::size_t INITIAL_GC_LIMIT = 131072U;

  struct UniqueTableConfig {
    /// The number of variables
    std::size_t nVars = 0U;

    /// The number of hash buckets to use (has to be a power of two)
    std::size_t nBuckets = 32768;

    /// The initial garbage collection limit
    std::size_t initialGCLimit = INITIAL_GC_LIMIT;
  };

  /**
   * @brief The default constructor
   * @param manager The memory manager to use
   * @param config The configuration for the unique table
   * @details The MemoryManager shall be constructed from the same type that the
   * unique table is then used for in the lookup method.
   */
  UniqueTable(MemoryManager& manager, const UniqueTableConfig& config);

  void resize(std::size_t nVars);

  /**
   * @brief The hash function for the hash table.
   * @details The hash function just combines the hashes of the edges of the
   * node. The hash value is masked to ensure that it is in the range
   * [0, nBuckets - 1].
   * @param p The node to hash.
   * @returns The hash value of the node.
   */
  template <class Node> [[nodiscard]] std::size_t hash(const Node& p) const {
    static_assert(std::is_base_of_v<NodeBase, Node>,
                  "Node must be derived from NodeBase");
    const std::size_t mask = cfg.nBuckets - 1;
    std::size_t key = 0U;
    for (const auto& succ : p.e) {
      qc::hashCombine(key, std::hash<Edge<Node>>{}(succ));
    }
    key &= mask;
    return key;
  }

  template <class Node>
  [[nodiscard]] static bool nodesAreEqual(const Node& p, const Node& q) {
    if constexpr (std::is_same_v<Node, dNode>) {
      return (p.e == q.e && (p.flags == q.flags));
    } else {
      return p.e == q.e;
    }
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

    const auto key = hash(*p);
    const auto v = p->v;
    ++stats[v].lookups;

    // search bucket in table corresponding to hashed value for the given node
    // and return it if found.
    if (auto* hashedNode = searchTable(*p, key);
        !Node::isTerminal(hashedNode)) {
      return hashedNode;
    }

    // if node not found -> add it to front of unique table bucket
    p->setNext(tables[v][key]);
    tables[v][key] = p;
    stats[v].trackInsert();

    return p;
  }

  /// Get a reference to the table
  [[nodiscard]] const auto& getTables() const { return tables; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Get a reference to individual statistics
  [[nodiscard]] const UniqueTableStatistics&
  getStats(std::size_t idx) const noexcept;

  /// Get a JSON object with the statistics
  [[nodiscard]] nlohmann::basic_json<>
  getStatsJson(bool includeIndividualTables = false) const;

  /// Get the total number of entries
  [[nodiscard]] std::size_t getNumEntries() const noexcept;

  /// Get the total number of active entries
  [[nodiscard]] std::size_t getNumActiveEntries() const noexcept;

  /// Get the peak total number of active entries
  [[nodiscard]] std::size_t getPeakNumActiveEntries() const noexcept;

  /**
   * @brief Determine whether the table possibly requires garbage collection.
   */
  [[nodiscard]] bool possiblyNeedsCollection() const;

  std::size_t garbageCollect(bool force = false);

  void clear();

  template <class Node> void print() {
    static_assert(std::is_base_of_v<NodeBase, Node>,
                  "Node must be derived from NodeBase");
    auto q = cfg.nVars - 1U;
    for (auto it = tables.rbegin(); it != tables.rend(); ++it) {
      auto& table = *it;
      std::cout << "\tq" << q << ":"
                << "\n";
      for (std::size_t key = 0; key < table.size(); ++key) {
        auto* p = static_cast<Node*>(table[key]);
        if (p != nullptr) {
          std::cout << "\tkey=" << key << ": ";
        }

        while (p != nullptr) {
          std::cout << "\t\t" << std::hex << reinterpret_cast<std::uintptr_t>(p)
                    << std::dec;
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
  /// Typedef for a bucket in the table
  using Bucket = NodeBase*;
  /// Typedef for the table
  using Table = std::vector<Bucket>;

  UniqueTableConfig cfg;

  /// The current garbage collection limit
  std::size_t gcLimit;

  /// A pointer to the memory manager for the nodes stored in the table.
  MemoryManager* memoryManager;

  /**
   * @brief The actual tables (one for each variable)
   * @details Each hash table is an array of buckets. Each bucket is a linked
   * list of entries. The linked list is implemented by using the next pointer
   * of the entries.
   */
  std::vector<Table> tables;

  /// A collection of statistics
  std::vector<UniqueTableStatistics> stats;

  /**
   * @brief Search for a node in the hash table with the given key.
   * @param p The node to search for.
   * @param key The hashed value used to search the table.
   * @returns A pointer to the node if found or Node::getTerminal() otherwise.
   */
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
