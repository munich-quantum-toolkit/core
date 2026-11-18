/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Placement.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"

#include "gtest/gtest.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <set>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {
using Edges = std::set<std::pair<size_t, size_t>>;
using Layouts = SmallVector<SmallVector<size_t>>;
} // namespace

static Layouts bruteForceLayouts(const Graph& graph,
                                 const CompilerTarget& target) {
  auto nodes = graph.getNodes();
  llvm::sort(nodes);
  auto permutation = llvm::to_vector(llvm::seq<size_t>(0, target.numSites()));
  Layouts result;
  if (nodes.size() > permutation.size()) {
    return result;
  }
  do {
    bool valid = true;
    for (size_t a = 0; a < nodes.size(); ++a) {
      for (size_t b = a + 1; b < nodes.size(); ++b) {
        valid &= !llvm::is_contained(graph.getNeighbours(nodes[a]), nodes[b]) ||
                 target.areAdjacent(permutation[a], permutation[b]);
      }
    }
    if (valid) {
      result.emplace_back(permutation.begin(),
                          permutation.begin() + nodes.size());
    }
  } while (std::ranges::next_permutation(permutation).found);
  llvm::sort(result);
  result.erase(std::ranges::unique(result).begin(), result.end());
  return result;
}
TEST(TopologyEmbedding, ExhaustiveGraphsMatchBruteForce) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  // All labelled five-vertex queries include disconnected, cyclic, branched and
  // impossible embeddings. Sparse IDs exercise ordering independently of IDs.
  constexpr std::array<size_t, 5> nodes{2, 5, 11, 17, 23};
  for (const auto& edges : {
           SmallVector<CompilerTarget::Coupling>{
               {0, 1},
               {1, 2},
               {2, 3},
               {3, 4},
           },
           SmallVector<CompilerTarget::Coupling>{
               {0, 1},
               {0, 2},
               {0, 3},
               {0, 4},
           },
           SmallVector<CompilerTarget::Coupling>{
               {0, 1},
               {1, 2},
               {2, 3},
               {0, 3},
               {0, 2},
               {3, 4},
               {1, 4},
           },
       }) {
    const auto target = llvm::cantFail(CompilerTarget::create(
        5, CompilerTarget::Connectivity::fromCouplings(edges),
        CompilerTarget::NativeOperations::unrestricted()));
    for (size_t mask = 0; mask < (1U << 10U); ++mask) {
      SCOPED_TRACE(mask);
      Graph graph(nodes);
      size_t bit = 0;
      for (size_t a = 0; a < nodes.size(); ++a) {
        for (size_t b = a + 1; b < nodes.size(); ++b, ++bit) {
          if ((mask & (1U << bit)) != 0) {
            graph.addEdge(nodes[b], nodes[a]);
            graph.addEdge(nodes[a], nodes[b]);
          }
        }
      }
      const auto layouts = bruteForceLayouts(graph, target);
      const auto first = findTopologyLayout(graph, target, context, 100000);
      EXPECT_EQ(first.has_value(), !layouts.empty());
      if (first) {
        EXPECT_TRUE(llvm::is_contained(layouts, *first));
      }
    }
  }
}

TEST(TopologyEmbedding, ParallelOrderAndBoundaryCases) {
  MLIRContext serial(MLIRContext::Threading::DISABLED);
  MLIRContext parallel;
  for (const auto& connectivity : {
           CompilerTarget::Connectivity::allToAll(),
           CompilerTarget::Connectivity::fromCouplings(
               {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}, {3, 4}, {1, 4}}),
       }) {
    const auto target = llvm::cantFail(CompilerTarget::create(
        5, connectivity, CompilerTarget::NativeOperations::unrestricted()));
    for (const auto& edges : {
             Edges{},
             Edges{{2, 5}},
             Edges{{2, 5}, {11, 17}},
             Edges{{2, 5}, {5, 11}, {11, 17}, {2, 17}},
             Edges{{2, 5}, {2, 11}, {2, 17}, {17, 23}},
         }) {
      Graph graph({23, 17, 11, 5, 2});
      for (const auto& [a, b] : edges) {
        graph.addEdge(a, b);
        graph.addEdge(b, a);
      }
      const auto layouts = bruteForceLayouts(graph, target);
      const auto layout = findTopologyLayout(graph, target, parallel, 100000);
      EXPECT_EQ(layout, findTopologyLayout(graph, target, serial, 100000));
      EXPECT_EQ(layout.has_value(), !layouts.empty());
      if (layout) {
        EXPECT_TRUE(llvm::is_contained(layouts, *layout));
      }
    }
    EXPECT_EQ(findTopologyLayout(Graph({}), target, parallel, 0),
              std::optional(SmallVector<size_t>{}));
    EXPECT_FALSE(findTopologyLayout(Graph({0, 1}), target, parallel, 1));
    EXPECT_FALSE(findTopologyLayout(Graph({0, 1, 2, 3, 4, 5}), target, parallel,
                                    100000));
    const Graph idle({99});
    const auto layout = findTopologyLayout(idle, target, parallel, 100000);
    ASSERT_TRUE(layout);
    EXPECT_TRUE(llvm::is_contained(bruteForceLayouts(idle, target), *layout));
  }

  // Exercise parallel search beyond the small-input serial cutoff.
  SmallVector<CompilerTarget::Coupling> edges;
  for (size_t vertex = 0; vertex < 100; ++vertex) {
    if (vertex % 10 != 9) {
      edges.emplace_back(vertex, vertex + 1);
    }
    if (vertex + 10 < 100) {
      edges.emplace_back(vertex, vertex + 10);
    }
  }
  const auto target = llvm::cantFail(CompilerTarget::create(
      100, CompilerTarget::Connectivity::fromCouplings(edges),
      CompilerTarget::NativeOperations::unrestricted()));
  Graph path({5, 7, 11});
  for (const auto& [a, b] : Edges{{5, 7}, {7, 11}}) {
    path.addEdge(a, b);
    path.addEdge(b, a);
  }
  for (const bool cycle : {false, true}) {
    if (cycle) {
      // A triangle cannot embed in a bipartite grid.
      path.addEdge(5, 11);
      path.addEdge(11, 5);
    }
    for (const size_t budget : {0U, 2U, 100000U}) {
      const auto expected = findTopologyLayout(path, target, serial, budget);
      if (cycle || budget < 3) {
        EXPECT_FALSE(expected);
      } else {
        ASSERT_TRUE(expected);
        EXPECT_EQ(std::set<size_t>(expected->begin(), expected->end()).size(),
                  path.getNodes().size());
        EXPECT_TRUE(target.areAdjacent((*expected)[0], (*expected)[1]));
        EXPECT_TRUE(target.areAdjacent((*expected)[1], (*expected)[2]));
      }
      for (const unsigned workers : {1U, 4U}) {
        SCOPED_TRACE(workers);
        SCOPED_TRACE(budget);
        llvm::StdThreadPool pool(llvm::hardware_concurrency(workers));
        MLIRContext threaded(MLIRContext::Threading::DISABLED);
        threaded.setThreadPool(pool);
        EXPECT_EQ(findTopologyLayout(path, target, threaded, budget), expected);
      }
    }
  }
}

TEST(TopologyEmbedding, EnumerationStreamsCompleteLayoutsWithinBudget) {
  MLIRContext context;
  const auto target = llvm::cantFail(
      CompilerTarget::create(3, CompilerTarget::Connectivity::allToAll(),
                             CompilerTarget::NativeOperations::unrestricted()));
  Graph graph({7, 11});
  graph.addEdge(7, 11);
  graph.addEdge(11, 7);
  SmallVector<SmallVector<size_t>> layouts;
  const auto visit = [&](ArrayRef<size_t> layout) {
    layouts.emplace_back(layout.begin(), layout.end());
    return layouts.size() == 3;
  };
  enumerateTopologyLayouts(graph, target, context, 1, visit);
  EXPECT_TRUE(layouts.empty());
  enumerateTopologyLayouts(graph, target, context, 100, visit);
  ASSERT_EQ(layouts.size(), 3);
  for (const auto& layout : layouts) {
    ASSERT_EQ(layout.size(), 2);
    EXPECT_NE(layout[0], layout[1]);
    EXPECT_LT(layout[0], 3);
    EXPECT_LT(layout[1], 3);
  }
  llvm::sort(layouts);
  EXPECT_EQ(std::ranges::adjacent_find(layouts), layouts.end());
  const auto explicitTarget = llvm::cantFail(
      CompilerTarget::create(5,
                             CompilerTarget::Connectivity::fromCouplings(
                                 {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {3, 4}}),
                             CompilerTarget::NativeOperations::unrestricted()));
  Graph disconnected({2, 5, 11, 17});
  for (const auto [a, b] : {std::pair{2, 5}, std::pair{5, 11}}) {
    disconnected.addEdge(a, b);
    disconnected.addEdge(b, a);
  }
  layouts.clear();
  enumerateTopologyLayouts(disconnected, explicitTarget, context, 100000,
                           [&](ArrayRef<size_t> layout) {
                             layouts.emplace_back(layout.begin(), layout.end());
                             return false;
                           });
  llvm::sort(layouts);
  EXPECT_EQ(layouts, bruteForceLayouts(disconnected, explicitTarget));
}
