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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {
namespace {
using Adjacency = SmallVector<SmallVector<size_t>, 0>;

struct JoinStep {
  size_t vertex;
  size_t degree;
  SmallVector<size_t> previous;
};

// Views into immutable adjacency lists replace intermediate join tables.
struct JoinFrame {
  ArrayRef<size_t> candidates;
  size_t next = 0;
};
} // namespace

// Each edge is an equality join with an already assigned neighbor. Choosing
// the most constrained vertex first avoids materializing pattern relations.
static SmallVector<JoinStep> planJoins(const Graph& graph,
                                       ArrayRef<size_t> nodes) {
  DenseMap<size_t, size_t> positions;
  for (const auto [position, node] : llvm::enumerate(nodes)) {
    positions[node] = position;
  }
  Adjacency adjacency(nodes.size());
  for (size_t vertex = 0; vertex < nodes.size(); ++vertex) {
    for (const auto neighbor : graph.getNeighbours(nodes[vertex])) {
      adjacency[vertex].push_back(positions.at(neighbor));
    }
    llvm::sort(adjacency[vertex]);
  }
  SmallVector<JoinStep> steps;
  llvm::BitVector selected(nodes.size());
  SmallVector<size_t> bound(nodes.size());
  for (size_t depth = 0; depth < nodes.size(); ++depth) {
    size_t best = nodes.size();
    std::pair<size_t, size_t> bestScore;
    for (size_t vertex = 0; vertex < nodes.size(); ++vertex) {
      const auto score = std::pair{bound[vertex], adjacency[vertex].size()};
      if (!selected[vertex] && (best == nodes.size() || score > bestScore)) {
        best = vertex;
        bestScore = score;
      }
    }
    JoinStep step{
        .vertex = best,
        .degree = adjacency[best].size(),
        .previous = {},
    };
    for (const auto neighbor : adjacency[best]) {
      if (selected[neighbor]) {
        step.previous.push_back(neighbor);
      }
      ++bound[neighbor];
    }
    selected.set(best);
    steps.push_back(std::move(step));
  }
  return steps;
}

static std::optional<SmallVector<size_t>>
searchLayouts(ArrayRef<JoinStep> steps, const Adjacency& adjacency,
              ArrayRef<size_t> sites, bool allToAll, size_t begin, size_t end,
              size_t maxCandidates,
              function_ref<bool(ArrayRef<size_t>)> visit) {
  SmallVector<size_t> layout(steps.size());
  llvm::BitVector used(sites.size());
  SmallVector<JoinFrame> frames(steps.size());
  frames[0] = {.candidates = sites.slice(begin, end - begin), .next = 0};
  size_t depth = 0;
  while (true) {
    auto& frame = frames[depth];
    const auto& step = steps[depth];
    if (frame.next == frame.candidates.size()) {
      if (depth == 0) {
        return std::nullopt;
      }
      used.reset(layout[steps[--depth].vertex]);
      continue;
    }
    if (maxCandidates == 0) {
      return std::nullopt;
    }
    --maxCandidates;
    const auto candidate = frame.candidates[frame.next++];
    if (used[candidate] ||
        (!allToAll && (adjacency[candidate].size() < step.degree ||
                       llvm::any_of(step.previous, [&](size_t neighbor) {
                         return !std::binary_search(
                             adjacency[candidate].begin(),
                             adjacency[candidate].end(), layout[neighbor]);
                       })))) {
      continue;
    }
    layout[step.vertex] = candidate;
    if (depth + 1 == steps.size()) {
      if (!visit || visit(layout)) {
        return layout;
      }
      continue;
    }
    used.set(candidate);
    const auto& next = steps[++depth];
    ArrayRef<size_t> candidates = sites;
    if (!allToAll) {
      for (const auto neighbor : next.previous) {
        const auto& adjacent = adjacency[layout[neighbor]];
        if (adjacent.size() < candidates.size()) {
          candidates = adjacent;
        }
      }
    }
    frames[depth] = {.candidates = candidates, .next = 0};
  }
}

static std::optional<SmallVector<size_t>>
findTopologyLayoutImpl(const Graph& graph, const CompilerTarget& target,
                       MLIRContext& context, size_t maxCandidates,
                       function_ref<bool(ArrayRef<size_t>)> visit = nullptr) {
  if (graph.getNumNodes() > target.numSites()) {
    return std::nullopt;
  }
  auto nodes = graph.getNodes();
  llvm::sort(nodes);
  if (nodes.empty()) {
    if (visit) {
      visit({});
    }
    return SmallVector<size_t>{};
  }
  const auto steps = planJoins(graph, nodes);
  const bool allToAll =
      target.connectivityKind() == CompilerTarget::Connectivity::Kind::AllToAll;
  Adjacency adjacency;
  if (!allToAll) {
    adjacency.resize(target.numSites());
    size_t maximumDegree = 0;
    for (size_t vertex = 0; vertex < target.numSites(); ++vertex) {
      target.forEachNeighbour(vertex, [&](size_t neighbor) {
        adjacency[vertex].push_back(neighbor);
      });
      llvm::sort(adjacency[vertex]);
      maximumDegree = std::max(maximumDegree, adjacency[vertex].size());
    }
    if (steps.front().degree > maximumDegree) {
      return std::nullopt;
    }
  }
  const auto sites = llvm::to_vector(llvm::seq<size_t>(0, target.numSites()));
  // Scheduling dominates small searches; a single vertex has no join work.
  // Fixed ranges keep bounded-search results independent of thread count.
  const auto tasks = !visit && sites.size() >= 64 && steps.size() > 1
                         ? std::min(sites.size(), size_t{32})
                         : size_t{1};
  SmallVector<std::optional<SmallVector<size_t>>> results(tasks);
  parallelFor(&context, 0, tasks, [&](size_t task) {
    const auto begin =
        sites.size() / tasks * task + std::min(sites.size() % tasks, task);
    const auto end =
        begin + sites.size() / tasks + (task < sites.size() % tasks ? 1 : 0);
    const auto budget =
        maxCandidates / tasks + (task < maxCandidates % tasks ? 1 : 0);
    results[task] = searchLayouts(steps, adjacency, sites, allToAll, begin, end,
                                  budget, visit);
  });
  for (auto& result : results) {
    if (result) {
      return std::move(result);
    }
  }
  return std::nullopt;
}

std::optional<SmallVector<size_t>>
findTopologyLayout(const Graph& graph, const CompilerTarget& target,
                   MLIRContext& context, size_t maxCandidates) {
  return findTopologyLayoutImpl(graph, target, context, maxCandidates);
}

namespace {
using PatternGroup = CompilerTarget::TopologyPatterns::Group;
using PatternType = CompilerTarget::TopologyPatterns::Type;
constexpr size_t UNASSIGNED = std::numeric_limits<size_t>::max();

struct PatternSearchResult {
  std::optional<SmallVector<size_t>> layout;
  size_t probes = 0;
};

} // namespace

namespace {
using PatternIndex = CompilerTarget::TopologyPatterns::Index;

struct PatternStep {
  // A singleton step assigns an isolated vertex, or an all-to-all vertex.
  const PatternIndex* index = nullptr;
  SmallVector<size_t, 8> vertices;
  SmallVector<size_t, 8> bound;
  SmallVector<size_t, 8> added;
};

// Explicit DFS frames bound call-stack use even for long paths or many idle
// qubits. A frame holds a view, not a materialized intermediate join.
struct PatternFrame {
  ArrayRef<size_t> rows;
  size_t next = 0;
  size_t end = 0;
  bool indexed = false;
};

class PatternSearch {
public:
  PatternSearch(ArrayRef<PatternStep> steps, ArrayRef<size_t> degrees,
                ArrayRef<size_t> targetDegrees, size_t begin, size_t end,
                function_ref<bool(ArrayRef<size_t>)> visit = nullptr)
      : visit_(visit), steps_(steps), degrees_(degrees),
        targetDegrees_(targetDegrees), layout_(degrees.size(), UNASSIGNED),
        used_(targetDegrees.size()), frames_(steps.size()) {
    frames_[0] = {.rows = {}, .next = begin, .end = end};
  }

  PatternSearchResult advance(size_t budget) {
    PatternSearchResult result;
    assert(!done_);
    while (true) {
      auto& frame = frames_[depth_];
      const auto& step = steps_[depth_];
      if (frame.next == frame.end) {
        if (depth_ == 0) {
          done_ = true;
          return result;
        }
        undo(steps_[--depth_]);
        continue;
      }
      if (result.probes == budget) {
        return result;
      }
      ++result.probes;
      const auto row = frame.indexed ? frame.rows[frame.next++] : frame.next++;
      const auto tuple =
          step.index != nullptr ? (*step.index)[row] : ArrayRef<size_t>(row);
      if (llvm::any_of(step.bound,
                       [&](size_t column) {
                         return layout_[step.vertices[column]] != tuple[column];
                       }) ||
          llvm::any_of(step.added, [&](size_t column) {
            return used_[tuple[column]] || degrees_[step.vertices[column]] >
                                               targetDegrees_[tuple[column]];
          })) {
        continue;
      }
      for (const auto column : step.added) {
        layout_[step.vertices[column]] = tuple[column];
        used_.set(tuple[column]);
      }
      if (depth_ + 1 == steps_.size()) {
        if (!visit_ || visit_(layout_)) {
          result.layout = layout_;
          done_ = true;
          return result;
        }
        undo(step);
        continue;
      }
      ++depth_;
      frames_[depth_] = candidates(steps_[depth_]);
    }
  }

  [[nodiscard]] bool done() const { return done_; }

private:
  [[nodiscard]] PatternFrame candidates(const PatternStep& step) const {
    PatternFrame frame{
        .rows = {},
        .end = step.index != nullptr ? step.index->size() : used_.size(),
    };
    for (const auto column : step.bound) {
      auto rows =
          step.index->candidates(column, layout_[step.vertices[column]]);
      if (!frame.indexed || rows.size() < frame.end) {
        frame.rows = rows;
        frame.end = rows.size();
        frame.indexed = true;
      }
    }
    return frame;
  }

  void undo(const PatternStep& step) {
    for (const auto column : step.added) {
      auto& vertex = layout_[step.vertices[column]];
      used_.reset(vertex);
      vertex = UNASSIGNED;
    }
  }

  function_ref<bool(ArrayRef<size_t>)> visit_;
  ArrayRef<PatternStep> steps_;
  ArrayRef<size_t> degrees_;
  ArrayRef<size_t> targetDegrees_;
  SmallVector<size_t> layout_;
  llvm::BitVector used_;
  SmallVector<PatternFrame> frames_;
  size_t depth_ = 0;
  bool done_ = false;
};
} // namespace

static std::optional<SmallVector<size_t>>
searchPatternLayouts(ArrayRef<PatternStep> steps, ArrayRef<size_t> degrees,
                     ArrayRef<size_t> targetDegrees, MLIRContext& context,
                     size_t budget,
                     function_ref<bool(ArrayRef<size_t>)> visit) {
  const auto seeds = steps.front().index != nullptr
                         ? steps.front().index->size()
                         : targetDegrees.size();
  if (seeds == 0 || budget == 0) {
    return std::nullopt;
  }
  if (visit) {
    PatternSearch search(steps, degrees, targetDegrees, 0, seeds, visit);
    return search.advance(budget).layout;
  }
  // Fixed logical ranges make budget allocation independent of worker count.
  const auto tasks = std::min(seeds, size_t{64});
  SmallVector<std::unique_ptr<PatternSearch>> searches;
  SmallVector<size_t> active;
  const auto makeTask = [&](size_t task) {
    const auto begin = seeds / tasks * task + std::min(seeds % tasks, task);
    const auto end = begin + seeds / tasks + (task < seeds % tasks ? 1 : 0);
    return std::make_unique<PatternSearch>(steps, degrees, targetDegrees, begin,
                                           end);
  };
  // Avoid worker dispatch for a short successful search. Incomplete ranges
  // resume below; completed ranges leave their remaining quota available.
  const auto prefixBudget = std::min(budget, size_t{256});
  size_t probes = 0;
  size_t nextTask = 0;
  while (nextTask < tasks && probes < prefixBudget) {
    searches.push_back(makeTask(nextTask));
    auto partial = searches.back()->advance(prefixBudget - probes);
    probes += partial.probes;
    ++nextTask;
    if (partial.layout) {
      return std::move(partial.layout);
    }
    if (!searches.back()->done()) {
      active.push_back(nextTask - 1);
      break;
    }
  }
  if (probes == budget) {
    return std::nullopt;
  }
  for (size_t task = nextTask; task < tasks; ++task) {
    searches.push_back(makeTask(task));
    active.push_back(task);
  }
  size_t quantum = 1024;
  while (!active.empty() && probes < budget) {
    const auto waveBudget = std::min(budget - probes, active.size() * quantum);
    SmallVector<PatternSearchResult> partials(active.size());
    parallelFor(&context, 0, active.size(), [&](size_t position) {
      const auto allowance = waveBudget / active.size() +
                             (position < waveBudget % active.size() ? 1 : 0);
      if (allowance != 0) {
        partials[position] = searches[active[position]]->advance(allowance);
      }
    });
    for (auto& partial : partials) {
      probes += partial.probes;
      if (partial.layout) {
        return std::move(partial.layout);
      }
    }
    llvm::erase_if(active, [&](size_t task) { return searches[task]->done(); });
    // Larger later slices amortize barriers. Unused quota passes to the next
    // wave, and the first successful logical range wins deterministically.
    quantum = std::min(quantum * 4, size_t{16384});
  }
  return std::nullopt;
}

static std::optional<SmallVector<size_t>> findTopologyPatternLayoutImpl(
    const Graph& graph, const CompilerTarget& target, MLIRContext& context,
    size_t maxCandidates, ArrayRef<PatternType> selected,
    function_ref<bool(ArrayRef<size_t>)> visit = nullptr) {
  if (graph.getNumNodes() > target.numSites()) {
    return std::nullopt;
  }
  auto nodes = graph.getNodes();
  llvm::sort(nodes);
  if (nodes.empty()) {
    if (visit) {
      visit({});
    }
    return SmallVector<size_t>{};
  }
  if (maxCandidates == 0) {
    return std::nullopt;
  }
  DenseMap<size_t, size_t> positions;
  SmallVector<size_t> degrees;
  for (const auto [position, node] : llvm::enumerate(nodes)) {
    positions[node] = position;
    degrees.push_back(graph.getDegree(node));
  }
  const bool allToAll =
      target.connectivityKind() == CompilerTarget::Connectivity::Kind::AllToAll;
  SmallVector<size_t> targetDegrees(target.numSites());
  for (size_t vertex = 0; vertex < target.numSites(); ++vertex) {
    if (allToAll) {
      targetDegrees[vertex] = target.numSites() - 1;
    } else {
      target.forEachNeighbour(vertex, [&](size_t) { ++targetDegrees[vertex]; });
    }
  }
  if (*llvm::max_element(degrees) > *llvm::max_element(targetDegrees)) {
    return std::nullopt;
  }

  SmallVector<PatternStep> remaining;
  const bool hasEdges =
      llvm::any_of(degrees, [](size_t degree) { return degree != 0; });
  if (hasEdges && !allToAll) {
    SmallVector<PatternType> types(selected);
    llvm::sort(types);
    types.erase(std::ranges::unique(types).begin(), types.end());
    // Necessary degree bounds avoid preparing rectangles for rings and other
    // patterns that cannot occur in the query.
    constexpr std::array<size_t, 10> arities{4, 2, 4, 4, 6, 6, 8, 6, 8, 12};
    constexpr std::array<size_t, 10> degreeTwo{4, 0, 1, 2, 4, 6, 8, 6, 8, 12};
    constexpr std::array<size_t, 10> degreeThree{0, 0, 1, 0, 0, 0, 0, 2, 4, 0};
    const size_t numDegreeTwo =
        llvm::count_if(degrees, [](size_t degree) { return degree >= 2; });
    const size_t numDegreeThree =
        llvm::count_if(degrees, [](size_t degree) { return degree >= 3; });
    SmallVector<const PatternGroup*> groups(types.size(), nullptr);
    const auto prepare = [&](size_t slot) {
      const auto type = static_cast<size_t>(types[slot]);
      assert(type < arities.size() && "Unknown topology pattern type");
      if (arities[type] <= nodes.size() && numDegreeTwo >= degreeTwo[type] &&
          numDegreeThree >= degreeThree[type]) {
        groups[slot] = target.topologyPattern(types[slot]);
      }
    };
    if (visit) {
      for (size_t slot = 0; slot < types.size(); ++slot) {
        prepare(slot);
      }
    } else {
      parallelFor(&context, 0, types.size(), prepare);
    }
    llvm::erase(groups, nullptr);
    auto slices = decomposeConnectivityGraph(graph, groups);
    if (failed(slices)) {
      return std::nullopt;
    }
    SmallVector<const PatternIndex*> indexes(groups.size());
    for (size_t slot = 0; slot < groups.size(); ++slot) {
      if (llvm::any_of(*slices, [&](const auto& slice) {
            return slice.type == groups[slot]->type();
          })) {
        indexes[slot] = target.topologyPatternIndex(groups[slot]->type());
      }
    }
    for (const auto [slot, group] : llvm::enumerate(groups)) {
      for (const auto& slice : *slices) {
        if (slice.type != group->type()) {
          continue;
        }
        PatternStep step{
            .index = indexes[slot],
            .vertices = {},
            .bound = {},
            .added = {},
        };
        for (const auto vertex : slice.vertices) {
          step.vertices.push_back(positions.at(vertex));
        }
        remaining.push_back(std::move(step));
      }
    }
  }
  SmallVector<PatternStep> steps;
  llvm::BitVector bound(nodes.size());
  while (!remaining.empty()) {
    const auto score = [&](const PatternStep& step) {
      const auto overlap = llvm::count_if(
          step.vertices, [&](size_t vertex) { return bound[vertex]; });
      return std::pair{overlap,
                       std::numeric_limits<size_t>::max() - step.index->size()};
    };
    auto* best =
        llvm::max_element(remaining, [&](const auto& a, const auto& b) {
          return score(a) < score(b);
        });
    auto step = std::move(*best);
    remaining.erase(best);
    for (const auto [column, vertex] : llvm::enumerate(step.vertices)) {
      (bound[vertex] ? step.bound : step.added).push_back(column);
      bound.set(vertex);
    }
    steps.push_back(std::move(step));
  }
  for (size_t vertex = 0; vertex < nodes.size(); ++vertex) {
    if (!bound[vertex]) {
      steps.push_back({.vertices = {vertex}, .bound = {}, .added = {0}});
    }
  }
  return searchPatternLayouts(steps, degrees, targetDegrees, context,
                              maxCandidates, visit);
}

std::optional<SmallVector<size_t>>
findTopologyPatternLayout(const Graph& graph, const CompilerTarget& target,
                          MLIRContext& context, size_t maxCandidates,
                          ArrayRef<PatternType> selected) {
  return findTopologyPatternLayoutImpl(graph, target, context, maxCandidates,
                                       selected);
}

static bool hasTwelveVertexCore(const Graph& graph) {
  DenseMap<size_t, size_t> degrees;
  SmallVector<size_t> pending;
  size_t remaining = graph.getNumNodes();
  if (remaining >= 12) {
    for (const auto vertex : graph.getNodes()) {
      degrees[vertex] = graph.getDegree(vertex);
      if (degrees[vertex] < 2) {
        pending.push_back(vertex);
      }
    }
    for (size_t next = 0; next < pending.size() && remaining >= 12; ++next) {
      --remaining;
      for (const auto neighbor : graph.getNeighbours(pending[next])) {
        auto& degree = degrees[neighbor];
        if (degree >= 2) {
          --degree;
          if (degree == 1) {
            pending.push_back(neighbor);
          }
        }
      }
    }
  }
  return remaining >= 12;
}

static constexpr std::array EXTENDED_PATTERNS{
    PatternType::FourCycle,   PatternType::QubitPair, PatternType::Star,
    PatternType::Chain4,      PatternType::Chain6,    PatternType::SixCycle,
    PatternType::EightCycle,  PatternType::Grid2x3,   PatternType::Grid2x4,
    PatternType::TwelveCycle,
};

std::optional<SmallVector<size_t>>
findTopologyAutomaticLayout(const Graph& graph, const CompilerTarget& target,
                            MLIRContext& context, size_t maxCandidates) {
  if (!hasTwelveVertexCore(graph)) {
    return findTopologyLayout(graph, target, context, maxCandidates);
  }
  return findTopologyPatternLayout(graph, target, context, maxCandidates,
                                   EXTENDED_PATTERNS);
}

void enumerateTopologyLayouts(const Graph& graph, const CompilerTarget& target,
                              MLIRContext& context, size_t maxCandidates,
                              function_ref<bool(ArrayRef<size_t>)> visit,
                              bool usePatterns) {
  if (maxCandidates == 0 && graph.getNumNodes() != 0) {
    return;
  }
  if (usePatterns && hasTwelveVertexCore(graph)) {
    const auto* cycle = target.topologyPattern(PatternType::TwelveCycle);
    if (cycle != nullptr && cycle->size() != 0 &&
        matchTopologyPattern(graph, *cycle)) {
      findTopologyPatternLayoutImpl(graph, target, context, maxCandidates,
                                    EXTENDED_PATTERNS, visit);
      return;
    }
  }
  findTopologyLayoutImpl(graph, target, context, maxCandidates, visit);
}
} // namespace mlir::qco
