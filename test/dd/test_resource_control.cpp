/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ComputeTable.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/ResourceControl.hpp"
#include "dd/UniqueTable.hpp"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <stop_token>
#include <thread>

namespace dd {
namespace {
template <class Callable>
void expectLimit(const Callable& callable, const ResourceLimit reason) {
  try {
    callable();
    FAIL() << "Expected a DD resource limit";
  } catch (const ResourceLimitExceeded& error) {
    EXPECT_EQ(error.reason(), reason);
  }
}

TEST(DDResourceControl, ChecksBeforeInitialAllocations) {
  // NOLINTNEXTLINE(misc-const-correctness): mutable in C++20.
  std::stop_source stop;
  stop.request_stop();
  ResourceControl cancelled({.stop = stop.get_token()});
  expectLimit([&] { const Package package(2, {}, &cancelled); },
              ResourceLimit::Cancelled);
  EXPECT_EQ(cancelled.allocatedBytes(), 0);
  EXPECT_EQ(cancelled.peakBytes(), 0);

  ResourceControl expired({.deadline = std::chrono::steady_clock::now()});
  expectLimit([&] { const Package package(2, {}, &expired); },
              ResourceLimit::Deadline);
  EXPECT_EQ(expired.allocatedBytes(), 0);

  ResourceControl tiny({.maxBytes = 1});
  expectLimit([&] { const Package package(2, {}, &tiny); },
              ResourceLimit::Memory);
  EXPECT_EQ(tiny.allocatedBytes(), 0);
}

TEST(DDResourceControl, AccountsConstructionAndLazyComputeStorage) {
  ResourceControl measured({});
  size_t initialBytes = 0;
  size_t operandBytes = 0;
  {
    Package package(2, {}, &measured);
    initialBytes = measured.allocatedBytes();
    EXPECT_GT(initialBytes, 0);
    const auto matrix = package.makeDDFromMatrix(CMat{{1., 2.}, {3., 4.}});
    operandBytes = measured.allocatedBytes();
    const auto product = package.multiply(matrix, matrix);
    const auto actual = getMatrix(product, 1);
    const CMat expected{{7., 10.}, {15., 22.}};
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 2; ++col) {
        EXPECT_NEAR(std::abs(actual[row][col] - expected[row][col]), 0., 1e-12);
      }
    }
    EXPECT_GT(measured.allocatedBytes(), initialBytes);
  }
  EXPECT_EQ(measured.allocatedBytes(), 0);
  EXPECT_GT(measured.peakBytes(), initialBytes);

  ResourceControl constructionLimit({.maxBytes = initialBytes - 1});
  expectLimit([&] { const Package package(2, {}, &constructionLimit); },
              ResourceLimit::Memory);
  EXPECT_EQ(constructionLimit.allocatedBytes(), 0);

  ResourceControl computeLimit({.maxBytes = operandBytes});
  expectLimit(
      [&] {
        Package package(2, {}, &computeLimit);
        const auto matrix = package.makeDDFromMatrix(CMat{{1., 2.}, {3., 4.}});
        static_cast<void>(package.multiply(matrix, matrix));
      },
      ResourceLimit::Memory);
  EXPECT_EQ(computeLimit.allocatedBytes(), 0);
  EXPECT_LE(computeLimit.peakBytes(), operandBytes);
}

TEST(DDResourceControl, CancellationDuringRecursiveConstruction) {
  // NOLINTNEXTLINE(misc-const-correctness): mutable in C++20.
  std::stop_source stop;
  ResourceControl control({.stop = stop.get_token()});
  size_t entries = 0;
  expectLimit(
      [&] {
        Package package(8, {}, &control);
        static_cast<void>(package.makeDDFromMatrix(
            256, [&](const size_t row, const size_t col) {
              if (++entries == 50) {
                stop.request_stop();
              }
              return std::complex<fp>{static_cast<fp>(row + col + 1), 0.};
            }));
      },
      ResourceLimit::Cancelled);
  EXPECT_GE(entries, 50);
  EXPECT_LT(entries, 256 * 256);
  EXPECT_EQ(control.allocatedBytes(), 0);
}

TEST(DDResourceControl, CountsOutstandingAllocationsTogether) {
  ResourceControl control({.maxBytes = 12});
  auto* first = control.allocate(8);
  expectLimit(
      [&] {
        auto* second = control.allocate(8);
        control.deallocate(second, 8);
      },
      ResourceLimit::Memory);
  EXPECT_EQ(control.allocatedBytes(), 8);
  EXPECT_EQ(control.peakBytes(), 8);
  control.deallocate(first, 8);
  EXPECT_EQ(control.allocatedBytes(), 0);
}

TEST(DDResourceControl, CountsNestedUniqueTableStorage) {
  ResourceControl control({.maxBytes = 4096});
  expectLimit(
      [&] {
        auto manager = MemoryManager::create<vNode>(1, &control);
        const UniqueTable table(manager, {.nVars = 1, .nBuckets = 1024});
      },
      ResourceLimit::Memory);
  EXPECT_EQ(control.allocatedBytes(), 0);
}

TEST(DDResourceControl, PreservesComputeEntryAlignment) {
  struct alignas(1024) Aligned {
    size_t value = 0;
  };
  ResourceControl control({});
  {
    ComputeTable<size_t, size_t, Aligned> table(2, &control);
    table.insert(1, 2, {3});
    const auto* value = table.lookup(1, 2);
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->value, 3);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(value) % alignof(Aligned), 0);
    EXPECT_GE(control.allocatedBytes(), 2 * sizeof(Aligned));
  }
  EXPECT_EQ(control.allocatedBytes(), 0);
}

TEST(DDResourceControl, ComputeTableResizeRetainsResource) {
  // NOLINTNEXTLINE(misc-const-correctness): mutable in C++20.
  std::stop_source stop;
  ResourceControl control({.stop = stop.get_token()});
  {
    ComputeTable<size_t, size_t, size_t> table(2, &control);
    table.insert(1, 2, 3);
    EXPECT_GT(control.allocatedBytes(), 0);
    table.resize(4);
    EXPECT_EQ(control.allocatedBytes(), 0);
    stop.request_stop();
    expectLimit([&] { table.insert(1, 2, 3); }, ResourceLimit::Cancelled);
  }
  EXPECT_EQ(control.allocatedBytes(), 0);
}

TEST(DDResourceControl, CancellationIsCheckedForCachedAndTerminalOperations) {
  const auto check = [](const auto& operation) {
    // NOLINTNEXTLINE(misc-const-correctness): mutable in C++20.
    std::stop_source stop;
    ResourceControl control({.stop = stop.get_token()});
    {
      Package package(1, {}, &control);
      stop.request_stop();
      expectLimit([&] { operation(package); }, ResourceLimit::Cancelled);
    }
    EXPECT_EQ(control.allocatedBytes(), 0);
  };
  check([](Package& package) {
    static_cast<void>(package.multiply(mEdge::one(), mEdge::one()));
  });
  check([](Package& package) {
    static_cast<void>(package.add(mEdge::one(), mEdge::one()));
  });
  check([](Package& package) {
    static_cast<void>(package.trace(mEdge::one(), 1));
  });
}

TEST(DDResourceControl, SharedBudgetAndIndependentCancellation) {
  ResourceControl measured({});
  size_t packageBytes = 0;
  {
    const Package package(2, {}, &measured);
    packageBytes = measured.allocatedBytes();
  }
  ResourceControl shared({.maxBytes = packageBytes * 2});
  std::promise<void> ready;
  std::jthread worker([&](const std::stop_token& stop) {
    try {
      const Package package(2, {}, &shared);
      ready.set_value();
      while (!stop.stop_requested()) {
        std::this_thread::yield();
      }
    } catch (...) {
      ready.set_exception(std::current_exception());
    }
  });
  ready.get_future().get();
  {
    const Package package(2, {}, &shared);
    EXPECT_EQ(shared.allocatedBytes(), packageBytes * 2);
    expectLimit([&] { const Package excess(1, {}, &shared); },
                ResourceLimit::Memory);
  }
  worker.request_stop();
  worker.join();
  EXPECT_EQ(shared.allocatedBytes(), 0);
  EXPECT_EQ(shared.peakBytes(), packageBytes * 2);

  // NOLINTNEXTLINE(misc-const-correctness): mutable in C++20.
  std::stop_source stop;
  stop.request_stop();
  ResourceControl cancelled({.stop = stop.get_token()});
  expectLimit([&] { const Package package(2, {}, &cancelled); },
              ResourceLimit::Cancelled);
  EXPECT_NO_THROW((Package(2, {}, &shared)));
}
} // namespace
} // namespace dd
