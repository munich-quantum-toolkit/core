/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "dd/statistics/MemoryManagerStatistics.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "dd/statistics/TableStatistics.hpp"
#include "dd/statistics/UniqueTableStatistics.hpp"

#include "gtest/gtest.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <array>
#include <clocale>
#include <cstddef>
#include <string>

TEST(DDStatistics, SerializesIndependentlyOfNumericLocale) {
  const auto originalLocale = std::string(std::setlocale(LC_NUMERIC, nullptr));
  const auto restoreLocale = llvm::scope_exit(
      [&] { std::setlocale(LC_NUMERIC, originalLocale.c_str()); });

  dd::MemoryManagerStatistics memory(sizeof(dd::vNode));
  memory.numAllocated = 8;
  memory.trackUsedEntries(3);
  dd::UniqueTableStatistics table;
  table.numBuckets = 8;
  table.entrySize = sizeof(dd::vNode);
  table.lookups = 4;
  table.hits = 1;
  table.collisions = 1;
  table.numEntries = 2;
  dd::Package package(2);
  package.incRef(dd::makeZeroState(2, package));
  const auto serialize = [&] {
    return std::array{
        memory.toString(),
        table.dd::TableStatistics::toString(),
        table.toString(),
        dd::getStatisticsString(package, true),
        dd::getStatisticsString(package, false),
    };
  };

  std::setlocale(LC_NUMERIC, "C");
  const auto expected = serialize();
  if (std::setlocale(LC_NUMERIC, "de_DE.UTF-8") == nullptr &&
      std::setlocale(LC_NUMERIC, "de_DE.utf8") == nullptr) {
    GTEST_SKIP() << "A comma-decimal numeric locale is unavailable";
  }
  const auto actual = serialize();
  std::setlocale(LC_NUMERIC, "C");
  for (size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    auto parsed = llvm::json::parse(actual[i]);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError());
    EXPECT_EQ(*parsed, llvm::cantFail(llvm::json::parse(expected[i])));
  }
}
