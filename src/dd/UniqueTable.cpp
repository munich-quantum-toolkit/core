/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/UniqueTable.hpp"

#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"

#include "statistics/StatisticsJson.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstddef>
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
  const auto oldSize = tables.size();
  for (auto i = nVars; i < oldSize; ++i) {
    entryCount_ -= stats[i].numEntries;
  }
  cfg.nVars = nVars;
  tables.resize(nVars);
  /// TODO: release entries for removed levels when shrinking populated tables.
  stats.resize(nVars);
  for (auto i = oldSize; i < nVars; ++i) {
    tables[i].resize(cfg.nBuckets);
    stats[i].entrySize = sizeof(Bucket);
    stats[i].numBuckets = cfg.nBuckets;
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
        if (!p->isMarked()) {
          NodeBase* next = p->next();
          if (lastp == nullptr) {
            bucket = next;
          } else {
            lastp->setNext(next);
          }
          memoryManager->returnEntry(*p);
          p = next;
          --stat.numEntries;
          --entryCount_;
        } else {
          lastp = p;
          p = p->next();
        }
      }
    }
    ++v;
  }

  /// Adapt the threshold to live entries so a mostly full table does not
  /// trigger a complete scan on every subsequent collection request.
  const auto numEntries = getNumEntries();
  if (numEntries > gcLimit / 10 * 9) {
    gcLimit = numEntries + cfg.initialGCLimit;
  }
  return numEntriesBefore - numEntries;
}

void UniqueTable::clear() {
  for (auto& table : tables) {
    for (auto& bucket : table) {
      bucket = nullptr;
    }
  }
  gcLimit = cfg.initialGCLimit;
  entryCount_ = 0U;
  for (auto& stat : stats) {
    stat.reset();
  }
};

const UniqueTableStatistics&
UniqueTable::getStats(const std::size_t idx) const noexcept {
  return stats.at(idx);
}

nlohmann::basic_json<> toJson(const UniqueTable& table,
                              const bool includeIndividualTables) {
  const auto& stats = table.getStats();
  if (std::ranges::all_of(stats, [](const UniqueTableStatistics& stat) {
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
    totalStats.gcRuns = std::max(totalStats.gcRuns, stat.gcRuns);
  }

  nlohmann::basic_json<> j;
  j["total"] = toJson(totalStats);
  if (includeIndividualTables) {
    std::size_t v = 0U;
    for (const auto& stat : stats) {
      j[std::to_string(v)] = toJson(stat);
      ++v;
    }
  }
  return j;
}

std::size_t UniqueTable::getNumEntries() const noexcept { return entryCount_; }

std::size_t UniqueTable::countMarkedEntries() const noexcept {
  std::size_t count = 0U;
  for (const auto& table : tables) {
    for (const auto* bucket : table) {
      const auto* p = bucket;
      while (p != nullptr) {
        if (p->isMarked()) {
          ++count;
        }
        p = p->next();
      }
    }
  }
  return count;
}

} // namespace dd
