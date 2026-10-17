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
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Placement.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {
using PatternType = CompilerTarget::TopologyPatterns::Type;
using Edges = std::set<std::pair<size_t, size_t>>;
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

namespace {
class PlacementAnalysisTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<QCODialect, qtensor::QTensorDialect,
                        arith::ArithDialect, cf::ControlFlowDialect,
                        func::FuncDialect, scf::SCFDialect>();
  }

  OwningOpRef<ModuleOp> parse(StringRef source) {
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    EXPECT_TRUE(moduleOp);
    if (moduleOp) {
      EXPECT_TRUE(succeeded(verify(*moduleOp)));
      EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    }
    return moduleOp;
  }
};
} // namespace

TEST_F(PlacementAnalysisTest, ScalarWireIdentityAndRepeatedInteractions) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%a: !qco.qubit, %b: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) {
      %idle = qco.alloc : !qco.qubit
      %c = qco.alloc : !qco.qubit
      %a1 = qco.h %a : !qco.qubit
      %a2, %b1 = qco.ctrl(%a1) targets(%target = %b) {
        %out = qco.x %target : !qco.qubit
        qco.yield %out : !qco.qubit
      } : ({!qco.qubit}, {!qco.qubit})
      %b2, %a3 = qco.swap %b1, %a2 : !qco.qubit, !qco.qubit
      %a4, %bit = qco.measure %a3 : !qco.qubit
      %a5 = qco.reset %a4 : !qco.qubit
      %a6, %c1 = qco.swap %a5, %c : !qco.qubit, !qco.qubit
      %b3, %c2, %idle1 = qco.barrier %b2, %c1, %idle
          : !qco.qubit, !qco.qubit, !qco.qubit
      qco.sink %c2 : !qco.qubit
      qco.sink %idle1 : !qco.qubit
      return %a6, %b3 : !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  moduleOp->print(beforeStream);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 4);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {0, 3}}));
  EXPECT_EQ(connectivity->graph.getDegree(2), 0);
  EXPECT_EQ(connectivity->graph.getDegree(0), 2);
  EXPECT_EQ(connectivity->qubits[0].root, func.getArgument(0));
  EXPECT_FALSE(connectivity->qubits[0].slot);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  moduleOp->print(afterStream);
  EXPECT_EQ(before, after);
}

TEST_F(PlacementAnalysisTest, TensorSlotsAndFromElementsKeepWireIdentity) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<3x!qco.qubit>) -> tensor<3x!qco.qubit> {
      %i0 = arith.constant 0 : index
      %i2 = arith.constant 2 : index
      %t1, %a = qtensor.extract %q[%i2] : tensor<3x!qco.qubit>
      %t2, %b = qtensor.extract %t1[%i0] : tensor<3x!qco.qubit>
      %c = qco.alloc : !qco.qubit
      %a1, %c1 = qco.swap %a, %c : !qco.qubit, !qco.qubit
      %t3 = qtensor.insert %a1 into %t2[%i2] : tensor<3x!qco.qubit>
      %t4, %a2 = qtensor.extract %t3[%i2] : tensor<3x!qco.qubit>
      %b1, %a3 = qco.swap %b, %a2 : !qco.qubit, !qco.qubit
      %t5 = qtensor.insert %a3 into %t4[%i2] : tensor<3x!qco.qubit>
      %t6 = qtensor.insert %b1 into %t5[%i0] : tensor<3x!qco.qubit>
      %other = qtensor.alloc(%i2) : tensor<2x!qco.qubit>
      qtensor.dealloc %other : tensor<2x!qco.qubit>
      %packed = qtensor.from_elements %c1 : tensor<1x!qco.qubit>
      %p1, %c2 = qtensor.extract %packed[%i0] : tensor<1x!qco.qubit>
      %c3 = qco.h %c2 : !qco.qubit
      %p2 = qtensor.insert %c3 into %p1[%i0] : tensor<1x!qco.qubit>
      qtensor.dealloc %p2 : tensor<1x!qco.qubit>
      return %t6 : tensor<3x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 6);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 2}, {2, 3}}));
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(connectivity->qubits[i].root, func.getArgument(0));
    EXPECT_EQ(connectivity->qubits[i].slot, i);
  }
  EXPECT_EQ(connectivity->graph.getDegree(1), 0);
}

TEST_F(PlacementAnalysisTest, NestedControlFlowUnionsInteractions) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%condition: i1, %n: index, %a: !qco.qubit,
                      %b: !qco.qubit, %c: !qco.qubit)
        -> (!qco.qubit, !qco.qubit, !qco.qubit) {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %x, %y, %z = scf.for %i = %zero to %n step %one
          iter_args(%fa = %a, %fb = %b, %fc = %c)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %u, %v, %w = qco.if %condition args(%qa = %fa, %qb = %fb, %qc = %fc)
            -> (!qco.qubit, !qco.qubit, !qco.qubit) {
          %sa, %sb = qco.swap %qa, %qb : !qco.qubit, !qco.qubit
          qco.yield %sa, %sb, %qc : !qco.qubit, !qco.qubit, !qco.qubit
        } else args(%qa = %fa, %qb = %fb, %qc = %fc) {
          %flag, %wa, %wb, %wc = scf.while
              (%state = %n, %aa = %qa, %ab = %qb, %ac = %qc)
              : (index, !qco.qubit, !qco.qubit, !qco.qubit)
              -> (i1, !qco.qubit, !qco.qubit, !qco.qubit) {
            %ha = qco.h %aa : !qco.qubit
            scf.condition(%condition) %condition, %ha, %ab, %ac
                : i1, !qco.qubit, !qco.qubit, !qco.qubit
          } do {
          ^bb0(%flag_arg: i1, %ba: !qco.qubit, %bb: !qco.qubit, %bc: !qco.qubit):
            %sb, %sc = qco.swap %bb, %bc : !qco.qubit, !qco.qubit
            scf.yield %n, %ba, %sb, %sc
                : index, !qco.qubit, !qco.qubit, !qco.qubit
          }
          qco.yield %wa, %wb, %wc : !qco.qubit, !qco.qubit, !qco.qubit
        }
        scf.yield %u, %v, %w : !qco.qubit, !qco.qubit, !qco.qubit
      }
      %r, %s, %t = qco.index_switch %n
          -> (!qco.qubit, !qco.qubit, !qco.qubit)
      case 0 args(%qa = %x, %qb = %y, %qc = %z) {
        %sa, %sc = qco.swap %qa, %qc : !qco.qubit, !qco.qubit
        qco.yield %sa, %qb, %sc : !qco.qubit, !qco.qubit, !qco.qubit
      }
      default args(%qa = %x, %qb = %y, %qc = %z) {
        qco.yield %qa, %qb, %qc : !qco.qubit, !qco.qubit, !qco.qubit
      }
      return %r, %s, %t : !qco.qubit, !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 3);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {1, 2}, {0, 2}}));
  EXPECT_EQ(connectivity->qubits[0].root, func.getArgument(2));
  auto loop = *func.getOps<scf::ForOp>().begin();
  auto branch = *loop.getBody()->getOps<IfOp>().begin();
  connectivity = buildConnectivityGraph(branch.getThenRegion());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}}));
  EXPECT_EQ(connectivity->qubits[0].root,
            branch.getThenRegion().getArgument(0));
}

TEST_F(PlacementAnalysisTest, TensorSlotsAcrossControlFlow) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<3x!qco.qubit>, %n: index)
        -> tensor<3x!qco.qubit> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %result = scf.for %i = %zero to %n step %one
          iter_args(%qs = %q) -> tensor<3x!qco.qubit> {
        %t1, %a = qtensor.extract %qs[%zero] : tensor<3x!qco.qubit>
        %t2, %b = qtensor.extract %t1[%two] : tensor<3x!qco.qubit>
        %x, %y = qco.swap %a, %b : !qco.qubit, !qco.qubit
        %t3 = qtensor.insert %x into %t2[%zero] : tensor<3x!qco.qubit>
        %t4 = qtensor.insert %y into %t3[%two] : tensor<3x!qco.qubit>
        scf.yield %t4 : tensor<3x!qco.qubit>
      }
      %t5, %c = qtensor.extract %result[%one] : tensor<3x!qco.qubit>
      %t6, %d = qtensor.extract %t5[%zero] : tensor<3x!qco.qubit>
      %u, %v = qco.swap %c, %d : !qco.qubit, !qco.qubit
      %t7 = qtensor.insert %u into %t6[%one] : tensor<3x!qco.qubit>
      %t8 = qtensor.insert %v into %t7[%zero] : tensor<3x!qco.qubit>
      return %t8 : tensor<3x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 3);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {0, 2}}));
  EXPECT_EQ(connectivity->qubits[2].slot, 2);
}

TEST_F(PlacementAnalysisTest, RegionInterfaceTracksProducedAndReorderedQubits) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit() {
      %x, %y = scf.execute_region -> (!qco.qubit, !qco.qubit) {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %u, %v = qco.swap %a, %b : !qco.qubit, !qco.qubit
        scf.yield %v, %u : !qco.qubit, !qco.qubit
      }
      %c = qco.alloc : !qco.qubit
      %z, %w = qco.swap %x, %c : !qco.qubit, !qco.qubit
      qco.sink %y : !qco.qubit
      qco.sink %z : !qco.qubit
      qco.sink %w : !qco.qubit
      return
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto connectivity = buildConnectivityGraph(func.getBody());
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_EQ(connectivity->qubits.size(), 3);
  EXPECT_EQ(getEdges(connectivity->graph), (Edges{{0, 1}, {1, 2}}));
}

TEST_F(PlacementAnalysisTest, RejectTensorPermutationAtLoopBackedge) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<2x!qco.qubit>, %n: index)
        -> tensor<2x!qco.qubit> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %result = scf.for %i = %zero to %n step %one
          iter_args(%qs = %q) -> tensor<2x!qco.qubit> {
        %t1, %a = qtensor.extract %qs[%zero] : tensor<2x!qco.qubit>
        %t2, %b = qtensor.extract %t1[%one] : tensor<2x!qco.qubit>
        %t3 = qtensor.insert %a into %t2[%one] : tensor<2x!qco.qubit>
        %t4 = qtensor.insert %b into %t3[%zero] : tensor<2x!qco.qubit>
        scf.yield %t4 : tensor<2x!qco.qubit>
      }
      return %result : tensor<2x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& message) {
    diagnostic = message.str();
    return success();
  });
  EXPECT_TRUE(failed(buildConnectivityGraph(func.getBody())));
  EXPECT_NE(diagnostic.find("backedges"), std::string::npos);
}

TEST_F(PlacementAnalysisTest, CachesAndInvalidatesConnectivity) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%a: !qco.qubit, %b: !qco.qubit)
        -> (!qco.qubit, !qco.qubit) {
      %x, %y = qco.swap %a, %b : !qco.qubit, !qco.qubit
      return %x, %y : !qco.qubit, !qco.qubit
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  ModuleAnalysisManager manager(moduleOp.get(), nullptr);
  auto analyses = AnalysisManager(manager).nest(func);
  EXPECT_FALSE(analyses.getCachedAnalysis<RegionConnectivityAnalysis>());
  const auto& initial = analyses.getAnalysis<RegionConnectivityAnalysis>();
  ASSERT_TRUE(succeeded(initial.getConnectivity()));
  EXPECT_EQ(getEdges(initial.getConnectivity()->graph), (Edges{{0, 1}}));
  EXPECT_EQ(&initial, &analyses.getAnalysis<RegionConnectivityAnalysis>());
  AnalysisManager::PreservedAnalyses preserved;
  preserved.preserve<RegionConnectivityAnalysis>();
  analyses.invalidate(preserved);
  ASSERT_TRUE(analyses.getCachedAnalysis<RegionConnectivityAnalysis>());
  EXPECT_EQ(&initial, &analyses.getAnalysis<RegionConnectivityAnalysis>());

  auto swap = *func.getOps<SWAPOp>().begin();
  swap->replaceAllUsesWith(swap->getOperands());
  swap.erase();
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
  analyses.invalidate(AnalysisManager::PreservedAnalyses{});
  EXPECT_FALSE(analyses.getCachedAnalysis<RegionConnectivityAnalysis>());
  const auto& updated = analyses.getAnalysis<RegionConnectivityAnalysis>();
  ASSERT_TRUE(succeeded(updated.getConnectivity()));
  EXPECT_TRUE(getEdges(updated.getConnectivity()->graph).empty());
}

TEST_F(PlacementAnalysisTest, RejectTensorPermutationAtRegionBoundary) {
  auto moduleOp = parse(R"mlir(
    func.func @circuit(%q: tensor<2x!qco.qubit>, %condition: i1)
        -> tensor<2x!qco.qubit> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %result = qco.if %condition args(%qs = %q) -> tensor<2x!qco.qubit> {
        %t1, %a = qtensor.extract %qs[%zero] : tensor<2x!qco.qubit>
        %t2, %b = qtensor.extract %t1[%one] : tensor<2x!qco.qubit>
        %t3 = qtensor.insert %a into %t2[%one] : tensor<2x!qco.qubit>
        %t4 = qtensor.insert %b into %t3[%zero] : tensor<2x!qco.qubit>
        qco.yield %t4 : tensor<2x!qco.qubit>
      } else args(%qs = %q) {
        qco.yield %qs : tensor<2x!qco.qubit>
      }
      return %result : tensor<2x!qco.qubit>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& message) {
    diagnostic = message.str();
    return success();
  });
  EXPECT_TRUE(failed(buildConnectivityGraph(func.getBody())));
  EXPECT_NE(diagnostic.find("tensor-slot identities at region joins"),
            std::string::npos);
  auto branch = *func.getOps<IfOp>().begin();
  EXPECT_TRUE(failed(RegionConnectivityAnalysis(branch).getConnectivity()));
  auto connectivity = buildConnectivityGraph(branch->getRegion(0));
  ASSERT_TRUE(succeeded(connectivity));
  EXPECT_TRUE(getEdges(connectivity->graph).empty());
}

TEST_F(PlacementAnalysisTest, RejectUnsupportedConnectivity) {
  for (const auto* source : {
           R"mlir(func.func @dynamic_size(%n: index) {
             %q = qtensor.alloc(%n) : tensor<?x!qco.qubit>
             qtensor.dealloc %q : tensor<?x!qco.qubit>
             return
           })mlir",
           R"mlir(func.func @dynamic_index(%q: tensor<2x!qco.qubit>, %i: index)
               -> tensor<2x!qco.qubit> {
             %t, %a = qtensor.extract %q[%i] : tensor<2x!qco.qubit>
             %out = qtensor.insert %a into %t[%i] : tensor<2x!qco.qubit>
             return %out : tensor<2x!qco.qubit>
           })mlir",
           R"mlir(func.func private @external(!qco.qubit) -> !qco.qubit
           func.func @call(%q: !qco.qubit) -> !qco.qubit {
             %out = func.call @external(%q) : (!qco.qubit) -> !qco.qubit
             return %out : !qco.qubit
           })mlir",
           R"mlir(func.func private @external(!qco.qubit) -> !qco.qubit
               attributes {mqt.unitary}
           func.func @call(%q: !qco.qubit) -> !qco.qubit {
             %out = qco.call @external(%q) : (!qco.qubit) -> !qco.qubit
             return %out : !qco.qubit
           })mlir",
           R"mlir(func.func @three_qubits(%a: !qco.qubit, %b: !qco.qubit,
                                         %c: !qco.qubit)
               -> (!qco.qubit, !qco.qubit, !qco.qubit) {
             %x, %y, %z = qco.ctrl(%a) targets(%qb = %b, %qc = %c) {
               %u, %v = qco.swap %qb, %qc
                   : !qco.qubit, !qco.qubit
               qco.yield %u, %v : !qco.qubit, !qco.qubit
             } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
             return %x, %y, %z : !qco.qubit, !qco.qubit, !qco.qubit
           })mlir",
           R"mlir(func.func @multiple_blocks() {
             cf.br ^exit
           ^exit:
             return
           })mlir",
       }) {
    auto moduleOp = parse(source);
    ASSERT_TRUE(moduleOp);
    auto func =
        *llvm::find_if(moduleOp->getOps<func::FuncOp>(),
                       [](func::FuncOp op) { return !op.isExternal(); });
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic&) { return success(); });
    EXPECT_TRUE(failed(RegionConnectivityAnalysis(func).getConnectivity()));
  }
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
  EXPECT_TRUE(succeeded(decomposeConnectivityGraph(graph, {})));
  EXPECT_FALSE(matchTopologyPattern(Graph({}), patterns[1]));
  for (const auto& [a, b] : Edges{{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}}) {
    graph.addEdge(a, b);
    graph.addEdge(b, a);
  }
  EXPECT_TRUE(failed(decomposeConnectivityGraph(graph, patterns.take_front())));
  auto slices = decomposeConnectivityGraph(graph, patterns);
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
