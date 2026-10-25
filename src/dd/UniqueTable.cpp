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

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <string>
#include <system_error>

namespace dd {

llvm::Error UniqueTable::checkCapacity(const size_t initial,
                                       const size_t maximum) {
  if (!std::has_single_bit(initial) || !std::has_single_bit(maximum) ||
      maximum < initial) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Unique table capacities must be powers of two, with maximum "
        "at least the initial capacity.");
  }
  return llvm::Error::success();
}

llvm::Expected<UniqueTable>
UniqueTable::create(MemoryManager& manager, const UniqueTableConfig& config) {
  if (auto error = checkCapacity(config.nBuckets, config.maxBuckets)) {
    return error;
  }
  return UniqueTable(manager, config);
}

UniqueTable::UniqueTable(MemoryManager& manager,
                         const UniqueTableConfig& config)
    : cfg(config), gcLimit(config.initialGCLimit), memoryManager(&manager) {
  llvm::cantFail(resize(config.nVars));
}

llvm::Error UniqueTable::resize(const std::size_t nVars) {
  const auto oldSize = tables.size();
  if (nVars < oldSize) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "A unique table cannot be shrunk.");
  }
  tables.resize(nVars);
  stats.resize(nVars);
  for (auto i = oldSize; i < nVars; ++i) {
    tables[i].resize(cfg.nBuckets);
    stats[i].entrySize = sizeof(Bucket);
    stats[i].numBuckets = cfg.nBuckets;
  }
  cfg.nVars = nVars;
  return llvm::Error::success();
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

  // Adapt the threshold to live entries so a mostly full table does not
  // trigger a complete scan on every subsequent collection request.
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

llvm::Expected<const UniqueTableStatistics&>
UniqueTable::getStats(const std::size_t idx) const {
  if (idx >= stats.size()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::result_out_of_range),
        "Unique table level is out of range.");
  }
  return stats[idx];
}

void writeJson(llvm::json::OStream& json, const UniqueTable& table,
               const bool includeIndividualTables, const size_t activeEntries) {
  const auto& stats = table.getStats();
  if (std::ranges::all_of(stats, [](const UniqueTableStatistics& stat) {
        return stat.peakNumEntries == 0U;
      })) {
    json.value("unused");
    return;
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

  json.object([&] {
    json.attributeBegin("total");
    writeJson(json, totalStats, activeEntries);
    json.attributeEnd();
    if (includeIndividualTables) {
      size_t v = 0U;
      for (const auto& stat : stats) {
        json.attributeBegin(std::to_string(v));
        writeJson(json, stat);
        json.attributeEnd();
        ++v;
      }
    }
  });
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
