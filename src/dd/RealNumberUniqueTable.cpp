/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/RealNumberUniqueTable.hpp"

#include "dd/DDDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/RealNumber.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ostream>

namespace dd {

RealNumberUniqueTable::RealNumberUniqueTable(MemoryManager& manager,
                                             const std::size_t initialGCLim)
    : memoryManager(&manager), initialGCLimit(initialGCLim) {
  stats.entrySize = sizeof(Bucket);
  stats.numBuckets = NBUCKET;
  for (const auto& ival : immortals::get()) {
    RealNumber::immortalize(lookupNonNegative(ival));
  }
}

size_t RealNumberUniqueTable::hash(const fp val) const noexcept {
  assert(val >= 0.);
  static_assert(std::numeric_limits<fp>::is_iec559 &&
                std::numeric_limits<fp>::digits == 53 &&
                std::numeric_limits<fp>::max_exponent == 1024);
  /// Mask mantissa bits instead of converting val / cellWidth to an integer:
  /// that quotient can overflow for large finite values or tiny tolerances.
  const auto bits = std::bit_cast<uint64_t>(val);
  const auto exponent = static_cast<int>((bits >> 52U) & 2047U);
  const auto shift = cellExponent + 1075 - std::max(1, exponent);
  auto cell = bits;
  if (shift > 52) {
    cell = 0;
  } else if (shift > 0) {
    cell &= ~uint64_t{0} << static_cast<unsigned>(shift);
  }
  return murmur64(cell) & (table.size() - 1U);
}

RealNumber* RealNumberUniqueTable::lookup(const fp val) {
  // if the value is close enough to zero, return the zero entry (avoiding -0.0)
  if (RealNumber::approximatelyZero(val)) {
    return &constants::zero;
  }
  if (const auto sign = std::signbit(val); sign) {
    return RealNumber::getNegativePointer(lookupNonNegative(std::abs(val)));
  }
  return lookupNonNegative(val);
}

RealNumber* RealNumberUniqueTable::lookupNonNegative(const fp val) {
  assert(!std::isnan(val));
  assert(val > 0);

  if (RealNumber::approximatelyEquals(val, 1.0)) {
    return &constants::one;
  }

  if (RealNumber::approximatelyEquals(val, SQRT2_2)) {
    return &constants::sqrt2over2;
  }

  updateTolerance();
  ++stats.lookups;
  RealNumber* best = nullptr;
  auto distance = RealNumber::eps;
  const auto scan = [&](size_t key) {
    for (auto* entry = table[key]; entry != nullptr; entry = entry->next()) {
      const auto difference = std::abs(entry->value - val);
      if (difference <= distance && (best == nullptr || difference < distance ||
                                     entry->value < best->value)) {
        best = entry;
        distance = difference;
      } else {
        ++stats.collisions;
      }
      if (difference == 0.) {
        break;
      }
    }
  };
  auto key = hash(val);
  scan(key);
  if (distance != 0.) {
    const auto lower = hash(val - RealNumber::eps);
    const auto upper = hash(val + RealNumber::eps);
    if (lower != key) {
      scan(lower);
    }
    if (upper != key && upper != lower) {
      scan(upper);
    }
  }
  if (best != nullptr) {
    ++stats.hits;
    return best;
  }
  if (stats.numEntries >= table.size() && table.size() < MAX_BUCKETS) {
    rehash(2 * table.size(), cellExponent);
    key = hash(val);
  }
  auto* entry = memoryManager->get<RealNumber>();
  entry->value = val;
  entry->LLBase::setNext(table[key]);
  table[key] = entry;
  stats.trackInsert();
  return entry;
}

void RealNumberUniqueTable::rehash(const std::size_t size, const int exponent) {
  Table next(size);
  table.swap(next);
  cellExponent = exponent;
  for (auto* head : next) {
    for (auto* p = head; p != nullptr;) {
      auto* saved = p->next();
      const auto key = hash(p->value);
      p->setNext(table[key]);
      table[key] = p;
      p = saved;
    }
  }
  stats.numBuckets = table.size();
}

void RealNumberUniqueTable::updateTolerance() {
  if (indexedTolerance == RealNumber::eps) {
    return;
  }
  assert(std::isfinite(RealNumber::eps) && RealNumber::eps >= 0.);
  /// Eight to sixteen tolerances per cell keeps clustered chains short.
  const auto exponent =
      RealNumber::eps == 0. ? -1074 : std::ilogb(RealNumber::eps) + 4;
  if (exponent != cellExponent && stats.numEntries != 0) {
    rehash(table.size(), exponent);
  } else {
    cellExponent = exponent;
  }
  indexedTolerance = RealNumber::eps;
}

bool RealNumberUniqueTable::possiblyNeedsCollection() const noexcept {
  return stats.numEntries >= gcLimit;
}

std::size_t RealNumberUniqueTable::garbageCollect(const bool force) noexcept {
  // nothing to be done if garbage collection is not forced, and the limit has
  // not been reached, or the current count is minimal.
  if ((!force && !possiblyNeedsCollection()) || stats.numEntries == 0) {
    return 0;
  }

  ++stats.gcRuns;
  const auto before = stats.numEntries;
  for (auto& bucket : table) {
    RealNumber* curr = bucket;
    RealNumber* prev = nullptr;
    while (curr != nullptr) {
      if (!RealNumber::isImmortal(curr) && !RealNumber::isMarked(curr)) {
        RealNumber* next = curr->next();
        if (prev == nullptr) {
          bucket = next;
        } else {
          prev->setNext(next);
        }
        memoryManager->returnEntry(*curr);
        curr = next;
        --stats.numEntries;
      } else {
        prev = curr;
        curr = curr->next();
      }
    }
  }

  /// Adapt the threshold to live entries so a mostly full table does not
  /// trigger a complete scan on every subsequent collection request.
  if (stats.numEntries > gcLimit / 10 * 9) {
    gcLimit = stats.numEntries + initialGCLimit;
  } else if (stats.numEntries < gcLimit / 128) {
    gcLimit /= 2;
  }
  return before - stats.numEntries;
}

void RealNumberUniqueTable::clear() noexcept {
  for (auto& bucket : table) {
    bucket = nullptr;
  }
  gcLimit = initialGCLimit;
  stats.reset();
}

void RealNumberUniqueTable::print() const {
  const auto precision = std::cout.precision();
  std::cout.precision(std::numeric_limits<dd::fp>::max_digits10);
  for (std::size_t key = 0; key < table.size(); ++key) {
    const auto* p = table[key];
    if (p != nullptr) {
      std::cout << key << ": \n";
    }

    while (p != nullptr) {
      std::cout << "\t\t" << p->value << " "
                << reinterpret_cast<std::uintptr_t>(p) << "\n";
      p = p->next();
    }

    if (table[key] != nullptr) {
      std::cout << "\n";
    }
  }
  std::cout.precision(precision);
}

std::ostream& RealNumberUniqueTable::printBucketDistribution(std::ostream& os) {
  for (const auto* bucket : table) {
    if (bucket == nullptr) {
      os << "0\n";
      continue;
    }
    std::size_t bucketCount = 0;
    while (bucket != nullptr) {
      ++bucketCount;
      bucket = bucket->next();
    }
    os << bucketCount << "\n";
  }
  os << "\n";
  return os;
}

std::size_t RealNumberUniqueTable::countMarkedEntries() const noexcept {
  std::size_t count = 0U;
  for (const auto* bucket : table) {
    const auto* curr = bucket;
    while (curr != nullptr) {
      if (RealNumber::isMarked(curr)) {
        ++count;
      }
      curr = curr->next();
    }
  }
  return count;
}

} // namespace dd
