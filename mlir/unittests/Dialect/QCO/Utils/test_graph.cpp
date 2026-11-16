/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/Graph.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>

using namespace mlir::qco;

TEST(GraphTest, CycleExcludesPrefixAndPreservesEdgeDirection) {
  Graph graph({});
  graph.addEdge(7, 101);
  graph.addEdge(101, 503);
  graph.addEdge(503, 9001);
  graph.addEdge(9001, 101);

  const auto cycle = graph.findCycle();
  ASSERT_TRUE(cycle.has_value());
  ASSERT_EQ(cycle->size(), 3U);
  for (size_t i = 0; i < cycle->size(); ++i) {
    EXPECT_NE((*cycle)[i], 7U);
    EXPECT_TRUE(llvm::is_contained(graph.getNeighbours((*cycle)[i]),
                                   (*cycle)[(i + 1) % cycle->size()]));
  }
}

TEST(GraphTest, SelfLoopAndClearedEdges) {
  Graph graph({});
  graph.addEdge(42, 42);
  const auto cycle = graph.findCycle();
  ASSERT_TRUE(cycle.has_value());
  ASSERT_EQ(cycle->size(), 1U);
  EXPECT_EQ(cycle->front(), 42U);

  graph.clearEdges();
  EXPECT_FALSE(graph.findCycle().has_value());
  graph.addEdge(42, 99);
  EXPECT_FALSE(graph.findCycle().has_value());
}

TEST(GraphTest, AcyclicDiamondHasNoCycle) {
  Graph graph({});
  graph.addEdge(10, 20);
  graph.addEdge(10, 30);
  graph.addEdge(20, 40);
  graph.addEdge(30, 40);
  EXPECT_FALSE(graph.findCycle().has_value());
}
