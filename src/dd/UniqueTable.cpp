/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/UniqueTable.hpp"

#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"

#include <algorithm>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>

namespace dd {

UniqueTable::UniqueTable(MemoryManager& manager,
                         const UniqueTableConfig& config)
    : cfg(config), gcLimit(config.initialGCLimit), memoryManager(&manager),
      tables(config.nVars), stats(config.nVars) {
  for (auto& stat : stats) {
    stat.entrySize = sizeof(Bucket);
    stat.numBuckets = cfg.nBuckets;
  }
}

void UniqueTable::resize(const std::size_t nVars) {
  cfg.nVars = nVars;
  tables.resize(nVars, Table(cfg.nBuckets));
  // TODO: if the new size is smaller than the old one we might have to
  // release the unique table entries for the superfluous variables
  stats.resize(nVars);
  for (auto& stat : stats) {
    stat.entrySize = sizeof(Bucket);
    stat.numBuckets = cfg.nBuckets;
  }
}

bool UniqueTable::possiblyNeedsCollection() const {
  return getNumEntries() >= gcLimit;
}

std::size_t UniqueTable::garbageCollect(const bool force) {
  const std::size_t numEntriesBefore = getNumEntries();
  if ((!force && numEntriesBefore < gcLimit) || numEntriesBefore == 0U) {
    return 0U;
  }

  std::size_t v = 0U;
  for (auto& table : tables) {
    auto& stat = stats[v];
    ++stat.gcRuns;
    for (auto& bucket : table) {
      NodeBase* p = bucket;
      NodeBase* lastp = nullptr;
      while (p != nullptr) {
        if (!p->marked()) {
          NodeBase* next = p->next();
          if (lastp == nullptr) {
            bucket = next;
          } else {
            lastp->setNext(next);
          }
          memoryManager->returnEntry(*p);
          p = next;
          --stat.numEntries;
        } else {
          p->unmark();
          lastp = p;
          p = p->next();
        }
      }
    }
    stat.numActiveEntries = stat.numEntries;
    ++v;
  }

  // The garbage collection limit changes dynamically depending on the number
  // of remaining (active) nodes. If it were not changed, garbage collection
  // would run through the complete table on each successive call once the
  // number of remaining entries reaches the garbage collection limit. It is
  // increased whenever the number of remaining entries is rather close to the
  // garbage collection threshold and decreased if the number of remaining
  // entries is much lower than the current limit.
  const auto numEntries = getNumEntries();
  if (numEntries > gcLimit / 10 * 9) {
    gcLimit = numEntries + cfg.initialGCLimit;
  }
  return numEntriesBefore - numEntries;
}

void UniqueTable::clear() {
  // clear unique table buckets
  for (auto& table : tables) {
    for (auto& bucket : table) {
      bucket = nullptr;
    }
  }
  gcLimit = cfg.initialGCLimit;
  for (auto& stat : stats) {
    stat.reset();
  }
};

const UniqueTableStatistics&
UniqueTable::getStats(const std::size_t idx) const noexcept {
  return stats.at(idx);
}

nlohmann::basic_json<>
UniqueTable::getStatsJson(const bool includeIndividualTables) const {
  if (std::all_of(stats.begin(), stats.end(),
                  [](const UniqueTableStatistics& stat) {
                    return stat.peakNumEntries == 0U;
                  })) {
    return "unused";
  }

  UniqueTableStatistics totalStats;
  for (const auto& stat : stats) {
    totalStats.entrySize = std::max(totalStats.entrySize, stat.entrySize);
    totalStats.numBuckets += stat.numBuckets;
    totalStats.numEntries += stat.numEntries;
    totalStats.peakNumEntries += stat.peakNumEntries;
    totalStats.collisions += stat.collisions;
    totalStats.hits += stat.hits;
    totalStats.lookups += stat.lookups;
    totalStats.inserts += stat.inserts;
    totalStats.numActiveEntries += stat.numActiveEntries;
    totalStats.peakNumActiveEntries += stat.peakNumActiveEntries;
    totalStats.gcRuns = std::max(totalStats.gcRuns, stat.gcRuns);
  }

  nlohmann::basic_json<> j;
  j["total"] = totalStats.json();
  if (includeIndividualTables) {
    std::size_t v = 0U;
    for (const auto& stat : stats) {
      j[std::to_string(v)] = stat.json();
      ++v;
    }
  }
  return j;
}

std::size_t UniqueTable::getNumEntries() const noexcept {
  return std::accumulate(
      stats.begin(), stats.end(), std::size_t{0},
      [](const std::size_t& sum, const UniqueTableStatistics& stat) {
        return sum + stat.numEntries;
      });
}

std::size_t UniqueTable::getNumActiveEntries() const noexcept {
  return std::accumulate(
      stats.begin(), stats.end(), std::size_t{0},
      [](const std::size_t& sum, const UniqueTableStatistics& stat) {
        return sum + stat.numActiveEntries;
      });
}

std::size_t UniqueTable::getPeakNumActiveEntries() const noexcept {
  return std::accumulate(
      stats.begin(), stats.end(), std::size_t{0},
      [](const std::size_t& sum, const UniqueTableStatistics& stat) {
        return sum + stat.peakNumActiveEntries;
      });
}

} // namespace dd
