/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

namespace dd {

RealNumberUniqueTable::RealNumberUniqueTable(MemoryManager& manager,
                                             const std::size_t initialGCLim)
    : memoryManager(&manager), initialGCLimit(initialGCLim) {
  stats.entrySize = sizeof(Bucket);
  stats.numBuckets = NBUCKET;

  // add 1/2 to the table so that it is available for complex numbers
  static_cast<void>(lookupNonNegative(0.5L));
}

std::int64_t RealNumberUniqueTable::hash(const fp val) noexcept {
  static constexpr std::int64_t MASK = NBUCKET - 1;
  assert(val >= 0);
  const auto key = static_cast<std::int64_t>(std::nearbyint(val * MASK));
  return std::min<std::int64_t>(key, MASK);
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

RealNumber* RealNumberUniqueTable::markNumber(RealNumber* num) noexcept {
  if (constants::isStaticNumber(num)) {
    return num;
  }
  auto* ptr = RealNumber::getAlignedPointer(num);
  ptr->markEntry();
  return RealNumber::getMarkedPointer(num);
}

RealNumber* RealNumberUniqueTable::unmarkNumber(RealNumber* num) noexcept {
  if (constants::isStaticNumber(num)) {
    return num;
  }
  auto* ptr = RealNumber::getAlignedPointer(num);
  ptr->unmarkEntry();
  return RealNumber::clearMark(num);
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

  ++stats.lookups;
  const auto lowerKey = hash(val - RealNumber::eps);
  const auto upperKey = hash(val + RealNumber::eps);

  if (upperKey == lowerKey) {
    return findOrInsert(lowerKey, val);
  }

  // code below is to properly handle border cases |----(-|-)----|
  // in case a value close to a border is looked up,
  // only the last entry in the lower bucket and the first entry in the upper
  // bucket need to be checked

  const auto key = hash(val);

  RealNumber* pLower; // NOLINT(cppcoreguidelines-init-variables)
  RealNumber* pUpper; // NOLINT(cppcoreguidelines-init-variables)
  if (lowerKey != key) {
    pLower = tailTable[static_cast<std::size_t>(lowerKey)];
    pUpper = table[static_cast<std::size_t>(key)];
  } else {
    pLower = tailTable[static_cast<std::size_t>(key)];
    pUpper = table[static_cast<std::size_t>(upperKey)];
  }

  const bool lowerMatchFound =
      (pLower != nullptr &&
       RealNumber::approximatelyEquals(val, pLower->value));
  const bool upperMatchFound =
      (pUpper != nullptr &&
       RealNumber::approximatelyEquals(val, pUpper->value));

  if (lowerMatchFound && upperMatchFound) {
    ++stats.hits;
    const auto diffToLower = std::abs(pLower->value - val);
    const auto diffToUpper = std::abs(pUpper->value - val);
    // val is actually closer to p_lower than to p_upper
    if (diffToLower < diffToUpper) {
      return pLower;
    }
    return pUpper;
  }

  if (lowerMatchFound) {
    ++stats.hits;
    return pLower;
  }

  if (upperMatchFound) {
    ++stats.hits;
    return pUpper;
  }

  // Since no match was found, a new value needs to be added
  // Depending on which border of the bucket the value lies, a value either
  // needs to be inserted in the front or the back of the bucket.
  if (key == lowerKey) {
    return insertFront(key, val);
  }
  return insertBack(key, val);
}

bool RealNumberUniqueTable::possiblyNeedsCollection() const noexcept {
  return stats.numEntries >= gcLimit;
}

std::size_t RealNumberUniqueTable::garbageCollect(const bool force) noexcept {
  // nothing to be done if garbage collection is not forced, and the limit has
  // not been reached, or the current count is minimal (the complex table
  // always contains at least 0.5)
  if ((!force && !possiblyNeedsCollection()) || stats.numEntries <= 1) {
    return 0;
  }

  ++stats.gcRuns;
  const auto before = stats.numEntries;
  for (std::size_t key = 0; key < table.size(); ++key) {
    auto* curr = table[key];
    RealNumber* prev = nullptr;
    while (curr != nullptr) {
      auto* next = curr->next();
      const bool isMarked = curr->isMarked();
      const bool isHalf = RealNumber::approximatelyEquals(curr->value, 0.5L);
      if (!isMarked && !isHalf) {
        if (prev == nullptr) {
          table[key] = next;
        } else {
          prev->setNext(next);
        }
        memoryManager->returnEntry(*RealNumber::getAlignedPointer(curr));
        --stats.numEntries;
        curr = next;
      } else {
        curr->unmarkEntry();
        curr = RealNumber::clearMark(curr);
        if (prev == nullptr) {
          table[key] = curr;
        } else {
          prev->setNext(curr);
        }
        prev = curr;
        curr = next;
      }
    }
    tailTable[key] = prev;
  }

  if (stats.numEntries > gcLimit / 10 * 9) {
    gcLimit = stats.numEntries + initialGCLimit;
  } else if (stats.numEntries < gcLimit / 128) {
    gcLimit /= 2;
  }

  return before - stats.numEntries;
}

void RealNumberUniqueTable::clear() noexcept {
  // clear table buckets
  for (auto& bucket : table) {
    bucket = nullptr;
  }
  for (auto& entry : tailTable) {
    entry = nullptr;
  }
  gcLimit = initialGCLimit;
  stats.reset();
}

void RealNumberUniqueTable::print() const {
  const auto precision = std::cout.precision();
  std::cout.precision(std::numeric_limits<dd::fp>::max_digits10);
  for (std::size_t key = 0; key < table.size(); ++key) {
    auto* p = table[key];
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
  for (auto* bucket : table) {
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
  for (auto* bucket : table) {
    auto* curr = bucket;
    while (curr != nullptr) {
      if (curr->isMarked()) {
        ++count;
      }
      curr = curr->next();
    }
  }
  return count;
}

RealNumber* RealNumberUniqueTable::findOrInsert(const std::int64_t key,
                                                const fp val) {
  const auto k = static_cast<std::size_t>(key);
  auto* curr = table[k];
  if (curr == nullptr) {
    auto* entry = memoryManager->get<RealNumber>();
    entry->value = val;
    entry->setNext(curr);
    table[k] = entry;
    tailTable[k] = entry;
    stats.trackInsert();
    return entry;
  }

  auto* back = tailTable[k];
  if (back != nullptr && back->value <= val) {
    if (RealNumber::approximatelyEquals(val, back->value)) {
      ++stats.hits;
      return back;
    }
    ++stats.collisions;
    auto* entry = memoryManager->get<RealNumber>();
    entry->value = val;
    entry->setNext(nullptr);
    back->setNext(entry);
    tailTable[k] = entry;
    stats.trackInsert();
    return entry;
  }

  RealNumber* prev = nullptr;
  const fp valTol = val + RealNumber::eps;
  while (curr != nullptr && curr->value <= valTol) {
    if (RealNumber::approximatelyEquals(curr->value, val)) {
      // check if val is actually closer to the next element in the list (if
      // there is one)
      if (curr->next() != nullptr) {
        const auto& next = curr->next();
        // potential candidate in range
        if (valTol >= next->value) {
          const auto diffToCurr = std::abs(curr->value - val);
          const auto diffToNext = std::abs(next->value - val);
          // val is actually closer to next than to curr
          if (diffToNext < diffToCurr) {
            ++stats.hits;
            return next;
          }
        }
      }
      ++stats.hits;
      return curr;
    }
    ++stats.collisions;
    prev = curr;
    curr = curr->next();
  }

  auto* entry = memoryManager->get<RealNumber>();
  entry->value = val;

  if (prev == nullptr) {
    // add to front of bucket
    table[k] = entry;
  } else {
    prev->setNext(entry);
  }
  entry->setNext(curr);
  if (curr == nullptr) {
    tailTable[k] = entry;
  }
  stats.trackInsert();
  return entry;
}

RealNumber* RealNumberUniqueTable::insertFront(const std::int64_t key,
                                               const fp val) {
  auto* entry = memoryManager->get<RealNumber>();
  entry->value = val;

  auto* curr = table[static_cast<std::size_t>(key)];
  table[static_cast<std::size_t>(key)] = entry;
  entry->setNext(curr);
  if (curr == nullptr) {
    tailTable[static_cast<std::size_t>(key)] = entry;
  }
  stats.trackInsert();
  return entry;
}

RealNumber* RealNumberUniqueTable::insertBack(const std::int64_t key,
                                              const fp val) {
  auto* entry = memoryManager->get<RealNumber>();
  entry->value = val;
  entry->setNext(nullptr);

  auto* back = tailTable[static_cast<std::size_t>(key)];
  tailTable[static_cast<std::size_t>(key)] = entry;
  if (back == nullptr) {
    table[static_cast<std::size_t>(key)] = entry;
  } else {
    back->setNext(entry);
  }
  stats.trackInsert();
  return entry;
}

} // namespace dd
