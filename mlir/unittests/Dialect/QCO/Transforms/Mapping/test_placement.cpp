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
#include "llvm/Support/LogicalResult.h"
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
using PatternType = CompilerTarget::TopologyPatterns::Type;
using Edges = std::set<std::pair<size_t, size_t>>;
constexpr std::array SMALL_PATTERN_TYPES{
    PatternType::FourCycle,
    PatternType::QubitPair,
    PatternType::Star,
};
constexpr std::array EXTENDED_PATTERN_TYPES{
    PatternType::FourCycle,  PatternType::QubitPair, PatternType::Star,
    PatternType::Chain4,     PatternType::Chain6,    PatternType::SixCycle,
    PatternType::EightCycle, PatternType::Grid2x3,   PatternType::Grid2x4,
};
class TopologyPatternMatching : public testing::Test {
protected:
  const CompilerTarget target = llvm::cantFail(
      CompilerTarget::create(4,
                             CompilerTarget::Connectivity::fromCouplings(
                                 {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}}),
                             CompilerTarget::NativeOperations::unrestricted()));
  const ArrayRef<CompilerTarget::TopologyPatterns::Group> patterns =
      *target.topologyPatterns();
};
} // namespace

static Edges getEdges(const Graph& graph) {
  Edges edges;
  for (const auto a : graph.getNodes()) {
    for (const auto b : graph.getNeighbours(a)) {
      if (a < b) {
        edges.emplace(a, b);
      }
    }
  }
  return edges;
}

static Edges matchEdges(PatternType type, ArrayRef<size_t> vertices) {
  Edges edges;
  const auto add = [&](size_t a, size_t b) {
    edges.insert(std::minmax(vertices[a], vertices[b]));
  };
  if (type == PatternType::FourCycle) {
    for (size_t i = 0; i < 4; ++i) {
      add(i, (i + 1) % 4);
    }
  } else {
    for (size_t i = 1; i < (type == PatternType::Star ? 4U : 2U); ++i) {
      add(0, i);
    }
  }
  return edges;
}

TEST(ConnectivityGraph, DirectedEdgeLookup) {
  EXPECT_EQ(Graph({}).getNumNodes(), 0);
  Graph graph({0, 1});
  EXPECT_EQ(graph.getNumNodes(), 2);
  graph.addEdge(0, 1);
  EXPECT_TRUE(graph.hasEdge(0, 1));
  EXPECT_FALSE(graph.hasEdge(1, 0));
  EXPECT_FALSE(graph.hasEdge(0, 0));
  EXPECT_FALSE(graph.hasEdge(2, 0));
  graph.removeEdge(0, 1);
  EXPECT_FALSE(graph.hasEdge(0, 1));
  EXPECT_EQ(graph.getNumNodes(), 2);
  graph.addEdge(1, 2);
  EXPECT_EQ(graph.getNumNodes(), 3);
  graph.clearEdges();
  EXPECT_EQ(graph.getNumNodes(), 3);
}

TEST_F(TopologyPatternMatching, ExhaustiveFiveVertexGraphs) {
  // Sparse IDs and reversed edge insertion also exercise determinism.
  constexpr std::array<size_t, 5> nodes{2, 5, 11, 17, 23};
  for (size_t mask = 0; mask < (1U << 10U); ++mask) {
    SCOPED_TRACE(mask);
    Graph graph(nodes);
    size_t bit = 0;
    for (size_t a = 0; a < nodes.size(); ++a) {
      for (size_t b = a + 1; b < nodes.size(); ++b, ++bit) {
        if ((mask & (1U << bit)) != 0) {
          graph.addEdge(nodes[a], nodes[b]);
          graph.addEdge(nodes[b], nodes[a]);
        }
      }
    }
    const auto edges = getEdges(graph);
    Graph reversed(nodes);
    for (const auto& [a, b] : llvm::reverse(edges)) {
      reversed.addEdge(b, a);
      reversed.addEdge(a, b);
    }
    for (const auto& pattern : patterns) {
      const auto type = pattern.type();
      auto permutation = nodes;
      std::optional<SmallVector<size_t, 4>> expected;
      do {
        const auto candidateEdges = matchEdges(type, permutation);
        if (std::ranges::includes(edges, candidateEdges)) {
          expected = SmallVector<size_t, 4>(
              permutation.begin(),
              permutation.begin() + (type == PatternType::QubitPair ? 2 : 4));
          break;
        }
      } while (std::ranges::next_permutation(permutation).found);
      EXPECT_EQ(matchTopologyPattern(graph, pattern), expected);
      EXPECT_EQ(matchTopologyPattern(reversed, pattern), expected);
    }
    auto slices = decomposeConnectivityGraph(graph, patterns);
    ASSERT_TRUE(succeeded(slices));
    Edges reconstructed;
    for (const auto& slice : *slices) {
      for (const auto& edge : matchEdges(slice.type, slice.vertices)) {
        EXPECT_TRUE(reconstructed.insert(edge).second);
      }
    }
    EXPECT_EQ(reconstructed, edges);
    EXPECT_EQ(getEdges(graph), edges);
  }
}

TEST_F(TopologyPatternMatching, PatternAvailabilityAndPreference) {
  Graph graph({0, 1, 2, 3, 4});
  EXPECT_TRUE(succeeded(decomposeConnectivityGraph(
      graph, ArrayRef<CompilerTarget::TopologyPatterns::Group>{})));
  EXPECT_FALSE(matchTopologyPattern(Graph({}), patterns[1]));
  for (const auto& [a, b] : Edges{{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}}) {
    graph.addEdge(a, b);
    graph.addEdge(b, a);
  }
  EXPECT_TRUE(failed(decomposeConnectivityGraph(graph, patterns.take_front())));
  const auto reversedPatterns = llvm::to_vector(llvm::reverse(patterns));
  auto slices = decomposeConnectivityGraph(graph, reversedPatterns);
  ASSERT_TRUE(succeeded(slices));
  ASSERT_EQ(slices->size(), 2);
  EXPECT_EQ((*slices)[0].type, PatternType::FourCycle);
  EXPECT_EQ((*slices)[1].type, PatternType::QubitPair);
  EXPECT_EQ((*slices)[1].vertices, (SmallVector<size_t, 4>{0, 2}));
  const auto pairTarget = llvm::cantFail(CompilerTarget::create(
      2, CompilerTarget::Connectivity::fromCouplings({{0, 1}}),
      CompilerTarget::NativeOperations::unrestricted()));
  auto pairs =
      decomposeConnectivityGraph(graph, *pairTarget.topologyPatterns());
  ASSERT_TRUE(succeeded(pairs));
  ASSERT_EQ(pairs->size(), 5);
  for (const auto& pair : *pairs) {
    EXPECT_EQ(pair.type, PatternType::QubitPair);
  }
}

TEST_F(TopologyPatternMatching, LargerPatternsReuseMatcherAndDecomposition) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  SmallVector<CompilerTarget::Coupling> couplings;
  for (size_t row = 0; row < 2; ++row) {
    for (size_t column = 0; column < 3; ++column) {
      couplings.emplace_back(row * 4 + column, row * 4 + column + 1);
    }
  }
  for (size_t column = 0; column < 4; ++column) {
    couplings.emplace_back(column, column + 4);
  }
  const auto largerTarget = llvm::cantFail(CompilerTarget::create(
      8, CompilerTarget::Connectivity::fromCouplings(couplings),
      CompilerTarget::NativeOperations::unrestricted()));
  for (const auto type : {
           PatternType::Chain4,
           PatternType::Chain6,
           PatternType::SixCycle,
           PatternType::EightCycle,
           PatternType::Grid2x3,
           PatternType::Grid2x4,
       }) {
    const auto* pattern = largerTarget.topologyPattern(type);
    ASSERT_NE(pattern, nullptr);
    ASSERT_NE(pattern->size(), 0);
    Graph graph(llvm::to_vector(llvm::seq<size_t>(0, pattern->arity())));
    for (const auto& [a, b] : pattern->edges()) {
      graph.addEdge(a, b);
      graph.addEdge(b, a);
    }
    auto match = matchTopologyPattern(graph, *pattern);
    ASSERT_TRUE(match);
    EXPECT_EQ(match->size(), pattern->arity());
    for (const auto& [a, b] : pattern->edges()) {
      EXPECT_TRUE(graph.hasEdge((*match)[a], (*match)[b]));
    }
    const std::array<const CompilerTarget::TopologyPatterns::Group*, 2> groups{
        nullptr,
        pattern,
    };
    auto slices = decomposeConnectivityGraph(graph, groups);
    ASSERT_TRUE(succeeded(slices));
    ASSERT_EQ(slices->size(), 1);
    EXPECT_EQ(slices->front().type, type);
    EXPECT_EQ(slices->front().vertices, *match);
    const auto layout =
        findTopologyPatternLayout(graph, largerTarget, context, 100000, {type});
    ASSERT_TRUE(layout);
    EXPECT_EQ(layout->size(), pattern->arity());
    EXPECT_EQ(std::set<size_t>(layout->begin(), layout->end()).size(),
              layout->size());
    for (const auto& [a, b] : pattern->edges()) {
      EXPECT_TRUE(largerTarget.areAdjacent((*layout)[a], (*layout)[b]));
    }
  }
}

namespace {
using Layouts = SmallVector<SmallVector<size_t>, 0>;
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
        valid &= !graph.hasEdge(nodes[a], nodes[b]) ||
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
TEST(TopologyPatternJoin, ExhaustiveGraphsMatchBruteForce) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  // All labelled five-vertex queries include disconnected, cyclic, star and
  // impossible joins. Sparse IDs exercise row ordering independently of IDs.
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
      for (const auto selected :
           {ArrayRef(SMALL_PATTERN_TYPES), ArrayRef(EXTENDED_PATTERN_TYPES)}) {
        const auto joined =
            findTopologyPatternLayout(graph, target, context, 100000, selected);
        EXPECT_EQ(joined.has_value(), !layouts.empty());
        if (joined) {
          EXPECT_TRUE(llvm::is_contained(layouts, *joined));
        }
      }
    }
  }
}

TEST(TopologyPatternJoin, ParallelOrderAndBoundaryCases) {
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
                  path.getNumNodes());
        EXPECT_TRUE(target.areAdjacent((*expected)[0], (*expected)[1]));
        EXPECT_TRUE(target.areAdjacent((*expected)[1], (*expected)[2]));
      }
      for (const unsigned workers : {1U, 2U, 4U, 8U}) {
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

TEST(TopologyPatternJoin, SelectedPatternsRespectBudgetsAndThreadCounts) {
  SmallVector<CompilerTarget::Coupling> couplings;
  for (size_t vertex = 0; vertex < 64; ++vertex) {
    if (vertex % 8 != 7) {
      couplings.emplace_back(vertex, vertex + 1);
    }
    if (vertex + 8 < 64) {
      couplings.emplace_back(vertex, vertex + 8);
    }
  }
  const auto target = llvm::cantFail(CompilerTarget::create(
      64, CompilerTarget::Connectivity::fromCouplings(couplings),
      CompilerTarget::NativeOperations::unrestricted()));
  MLIRContext serial(MLIRContext::Threading::DISABLED);
  for (const bool cycle : {false, true}) {
    Graph graph(llvm::to_vector(llvm::seq<size_t>(0, 9)));
    for (size_t vertex = 1; vertex < 9; ++vertex) {
      graph.addEdge(vertex - 1, vertex);
      graph.addEdge(vertex, vertex - 1);
    }
    if (cycle) {
      graph.addEdge(0, 8);
      graph.addEdge(8, 0);
    }
    for (const auto selected :
         {ArrayRef(SMALL_PATTERN_TYPES), ArrayRef(EXTENDED_PATTERN_TYPES)}) {
      for (const size_t budget : {0U, 1U, 64U, 256U, 100000U}) {
        const auto expected =
            findTopologyPatternLayout(graph, target, serial, budget, selected);
        if (cycle || budget == 0) {
          // An odd cycle cannot embed in a bipartite grid.
          EXPECT_FALSE(expected);
        } else if (budget == 100000) {
          ASSERT_TRUE(expected);
        }
        if (expected) {
          EXPECT_EQ(std::set<size_t>(expected->begin(), expected->end()).size(),
                    graph.getNumNodes());
          for (const auto& [a, b] : getEdges(graph)) {
            EXPECT_TRUE(target.areAdjacent((*expected)[a], (*expected)[b]));
          }
        }
        for (const unsigned workers : {1U, 2U, 4U, 8U}) {
          SCOPED_TRACE(workers);
          SCOPED_TRACE(budget);
          llvm::StdThreadPool pool(llvm::hardware_concurrency(workers));
          MLIRContext parallel(MLIRContext::Threading::DISABLED);
          parallel.setThreadPool(pool);
          EXPECT_EQ(findTopologyPatternLayout(graph, target, parallel, budget,
                                              selected),
                    expected);
        }
      }
    }
  }
  EXPECT_EQ(findTopologyPatternLayout(Graph({}), target, serial, 0,
                                      EXTENDED_PATTERN_TYPES),
            std::optional(SmallVector<size_t>{}));
  EXPECT_FALSE(findTopologyPatternLayout(
      Graph(llvm::to_vector(llvm::seq<size_t>(0, 65))), target, serial, 100000,
      EXTENDED_PATTERN_TYPES));
}

TEST(TopologyPatternJoin, UnavailablePatternsFallBackToCompletePairs) {
  SmallVector<CompilerTarget::Coupling> couplings;
  for (size_t a = 0; a < 20; ++a) {
    for (size_t b = a + 1; b < 20; ++b) {
      couplings.emplace_back(a, b);
    }
  }
  const auto target = llvm::cantFail(CompilerTarget::create(
      20, CompilerTarget::Connectivity::fromCouplings(couplings),
      CompilerTarget::NativeOperations::unrestricted()));
  ASSERT_EQ(target.topologyPattern(PatternType::Chain6), nullptr);
  Graph graph({0, 1, 2, 3, 4, 5});
  for (size_t vertex = 1; vertex < 6; ++vertex) {
    graph.addEdge(vertex - 1, vertex);
    graph.addEdge(vertex, vertex - 1);
  }
  MLIRContext context(MLIRContext::Threading::DISABLED);
  EXPECT_FALSE(findTopologyPatternLayout(graph, target, context, 100000,
                                         {PatternType::Chain6}));
  const auto layout =
      findTopologyPatternLayout(graph, target, context, 100000,
                                {PatternType::Chain6, PatternType::QubitPair});
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout->size(), graph.getNumNodes());
  EXPECT_EQ(std::set<size_t>(layout->begin(), layout->end()).size(),
            graph.getNumNodes());
  for (const auto& [a, b] : getEdges(graph)) {
    EXPECT_TRUE(target.areAdjacent((*layout)[a], (*layout)[b]));
  }
}
