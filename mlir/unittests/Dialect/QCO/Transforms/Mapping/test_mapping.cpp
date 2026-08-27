/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;
using mlir::mqt::getEntryPoint;

static SmallVector<Value> getQubitValues(ValueRange values) {
  return to_vector(llvm::make_filter_range(
      values, [](Value value) { return isa<QubitType>(value.getType()); }));
}

/// Return true, if the operations within a region fulfill the given coupling
/// constraints.
static bool isExecutable(Region& body,
                         DenseMap<Value, CompilerTarget::SiteId>& m,
                         const CompilerTarget& target) {
  for (Operation& op : body.getOps()) {
    if (auto staticOp = dyn_cast<StaticOp>(op)) {
      m.try_emplace(staticOp.getQubit(), staticOp.getIndex());
      continue;
    }

    if (auto unitaryOp = dyn_cast<UnitaryOpInterface>(op)) {
      if (!isa<BarrierOp>(op) && unitaryOp.getNumQubits() > 1) {
        assert(unitaryOp.getNumQubits() <= 2 && "expected two-qubit decomp.");

        const auto siteA = m.at(unitaryOp.getInputQubit(0));
        const auto siteB = m.at(unitaryOp.getInputQubit(1));
        const auto vertexA = target.vertexForSite(siteA);
        const auto vertexB = target.vertexForSite(siteB);
        if (!vertexA || !vertexB || !target.areAdjacent(*vertexA, *vertexB)) {
          llvm::dbgs() << "The two-qubit gate (" << siteA << ", " << siteB
                       << ") is not executable: \n";
          unitaryOp->dump();
          return false;
        }
      }

      for (auto [pred, succ] : llvm::zip_equal(unitaryOp.getInputQubits(),
                                               unitaryOp.getOutputQubits())) {
        const auto hw = m.at(pred);
        m.try_emplace(succ, hw);
      }

      continue;
    }

    if (auto resetOp = dyn_cast<ResetOp>(op)) {
      const auto hw = m.at(resetOp.getQubitIn());
      m.try_emplace(resetOp.getQubitOut(), hw);
      continue;
    }

    if (auto measOp = dyn_cast<MeasureOp>(op)) {
      const auto hw = m.at(measOp.getQubitIn());
      m.try_emplace(measOp.getQubitOut(), hw);
      continue;
    }

    if (!isa<scf::ForOp, scf::WhileOp, qco::IfOp, qco::IndexSwitchOp>(op)) {
      continue;
    }

    for (Region& region : op.getRegions()) {
      ValueRange initArgs =
          TypeSwitch<Operation*, ValueRange>(&op)
              .Case<qco::IfOp>([&](qco::IfOp ifOp) { return ifOp.getQubits(); })
              .Case<qco::IndexSwitchOp>([&](qco::IndexSwitchOp switchOp) {
                return switchOp.getTargets();
              })
              .Case<scf::WhileOp>(
                  [&](scf::WhileOp whileOp) { return whileOp.getInits(); })
              .Case<scf::ForOp>(
                  [&](scf::ForOp forOp) { return forOp.getInits(); })
              .Default([](Operation*) -> ValueRange { return {}; });

      const auto initialHardwareOrder = to_vector(llvm::map_range(
          getQubitValues(initArgs), [&](auto v) { return m.at(v); }));

      const auto qubitArgs = getQubitValues(region.getArguments());

      DenseMap<Value, CompilerTarget::SiteId> localM;
      for (auto [arg, hw] : llvm::zip_equal(qubitArgs, initialHardwareOrder)) {
        localM.try_emplace(arg, hw);
      }

      if (!isExecutable(region, localM, target)) {
        return false;
      }

      Operation* terminator = region.front().getTerminator();
      ValueRange finalOrderArgs =
          TypeSwitch<Operation*, ValueRange>(region.getParentOp())
              .Case<qco::IfOp, qco::IndexSwitchOp>([&](auto) {
                return cast<qco::YieldOp>(terminator).getTargets();
              })
              .Case<scf::WhileOp>([&](auto) {
                // Choose between "before" and "after" terminator.
                return region.getRegionNumber() == 0
                           ? cast<scf::ConditionOp>(terminator).getArgs()
                           : cast<scf::YieldOp>(terminator).getResults();
              })
              .Case<scf::ForOp>([&](scf::ForOp) {
                return cast<scf::YieldOp>(terminator).getResults();
              })
              .Default([](Operation*) -> ValueRange { return {}; });

      const auto finalOrder =
          to_vector(llvm::map_range(getQubitValues(finalOrderArgs),
                                    [&](auto v) { return localM.at(v); }));

      if (finalOrder != initialHardwareOrder) {
        llvm::dbgs()
            << "The hardware indices of the yielded terminator qubit values "
               "must be in the same order as parent's op input qubit values!\n";
        for (const auto hw : initialHardwareOrder) {
          llvm::dbgs() << hw << ' ';
        }
        llvm::dbgs() << "\n";
        for (const auto hw : finalOrder) {
          llvm::dbgs() << hw << ' ';
        }
        llvm::dbgs() << "\n";
        return false;
      }
    }

    for (OpResult res : op.getResults()) {
      if (!isa<QubitType>(res.getType())) {
        continue;
      }
      Value init =
          TypeSwitch<Operation*, Value>(&op)
              .Case<scf::WhileOp>([&](scf::WhileOp whileOp) {
                return whileOp.getInits()[res.getResultNumber()];
              })
              .Case<scf::ForOp>([&](scf::ForOp forOp) {
                return forOp.getTiedLoopInit(res)->get();
              })
              .Case<qco::IfOp>(
                  [&](qco::IfOp ifOp) { return ifOp.getTiedQubit(res)->get(); })
              .Case<qco::IndexSwitchOp>([&](qco::IndexSwitchOp switchOp) {
                return switchOp.getTiedTarget(res)->get();
              });

      const auto hw = m.at(init);
      m.try_emplace(res, hw);
    }
  }

  return true;
}

/// Return true, if the entry point fulfills the given coupling constraints.
static bool isExecutable(func::FuncOp entry, const CompilerTarget& target) {
  DenseMap<Value, CompilerTarget::SiteId> m;
  return isExecutable(entry.getFunctionBody(), m, target);
}

/// Return a nxn square-grid compiler target.
static CompilerTarget getSquareGridTarget(const size_t n) {
  const auto numTarget = n * n;

  std::vector<CompilerTarget::Coupling> couplings;
  couplings.reserve(n * n);

  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const auto i = (r * n) + c;
      if (c + 1 < n) {
        couplings.emplace_back(i, i + 1);
      }
      if (r + 1 < n) {
        couplings.emplace_back(i, i + n);
      }
    }
  }

  return llvm::cantFail(
      CompilerTarget::create(numTarget, std::move(couplings)));
}

/// Creates an N-qubit GHZ state, where N = `qubits.size()` using
/// straight-line programming.
static void flatGHZ(QCOProgramBuilder& builder, SmallVector<Value>& qubits) {
  qubits[0] = builder.h(qubits[0]);
  for (size_t i = 1; i < qubits.size(); ++i) {
    std::tie(qubits[0], qubits[i]) = builder.cx(qubits[0], qubits[i]);
  }
}

/// Creates an N-qubit GHZ state, where N = `qubits.size()` using an scf.for
/// operation.
static void loopGHZ(QCOProgramBuilder& builder, Value& tensor,
                    const int64_t size) {
  Value q0;
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  q0 = builder.h(q0);
  tensor = builder.qtensorInsert(q0, tensor, 0);

  tensor = builder
               .scfFor(1, size, 1, {tensor},
                       [&builder](Value iv, ValueRange args) {
                         SmallVector argQs{args[0]}; // ... is a tensor.

                         Value ctrl;
                         Value targ;

                         std::tie(argQs[0], ctrl) =
                             builder.qtensorExtract(argQs[0], 0);
                         std::tie(argQs[0], targ) =
                             builder.qtensorExtract(argQs[0], iv);

                         std::tie(ctrl, targ) = builder.cx(ctrl, targ);

                         argQs[0] = builder.qtensorInsert(ctrl, argQs[0], 0);
                         argQs[0] = builder.qtensorInsert(targ, argQs[0], iv);

                         return SmallVector{argQs};
                       })
               .front();
}

/// Creates an N-qubit CX/CZ circuit.
static void cxcz(QCOProgramBuilder& builder, SmallVector<Value>& qubits) {
  for (size_t i = 0; i + 1 < qubits.size(); ++i) {
    std::tie(qubits[i], qubits[i + 1]) = builder.cx(qubits[i], qubits[i + 1]);
  }
  for (size_t i = 0; i + 2 < qubits.size(); ++i) {
    std::tie(qubits[i], qubits[i + 2]) = builder.cz(qubits[i], qubits[i + 2]);
  }
}

namespace {

class MappingPassFixture : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<mqt::MQTDialect, QCODialect, qtensor::QTensorDialect,
                    scf::SCFDialect, arith::ArithDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static LogicalResult runPass(ModuleOp m, const CompilerTarget& target,
                               const MappingPassOptions& options) {
    PassManager pm(m->getContext());
    pm.addPass(createMappingPass(target, options));
    if (failed(pm.run(m))) {
      return failure();
    }

    RewritePatternSet patterns(m.getContext());
    SinkOp::getCanonicalizationPatterns(patterns, m.getContext());
    return applyPatternsGreedily(m, std::move(patterns));
  }

  std::unique_ptr<MLIRContext> context;
};

class MappingPassTest : public MappingPassFixture,
                        public testing::WithParamInterface<CompilerTarget> {};

}; // namespace

TEST_F(MappingPassFixture, MapTopologyOnlyWithEmptyOperationSet) {
  constexpr int64_t size = 3;

  const auto target = llvm::cantFail(CompilerTarget::create(
      3, std::vector<CompilerTarget::Coupling>{{0, 1}, {1, 2}},
      std::vector<CompilerTarget::Operation>{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  for (int64_t i = 0; i < size; ++i) {
    qubits[i] = builder.allocQubit();
  }

  qubits[0] = builder.x(qubits[0]);
  std::tie(qubits[0], qubits[1]) = builder.rxx(0.25, qubits[0], qubits[1]);
  std::tie(qubits[1], qubits[2]) = builder.rzx(0.5, qubits[1], qubits[2]);
  std::tie(qubits[0], qubits[2]) = builder.cx(qubits[0], qubits[2]);

  for (int64_t i = 0; i < qubits.size(); ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
    builder.sink(qubits[i]);
  }

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  size_t numSwaps = 0;
  m->walk([&](SWAPOp) { ++numSwaps; });
  EXPECT_GT(numSwaps, 0);
}

TEST_F(MappingPassFixture, PreserveNoncontiguousTargetSiteIds) {
  constexpr int64_t size = 3;

  std::vector<CompilerTarget::Site> sites;
  sites.emplace_back(llvm::cantFail(CompilerTarget::Site::create(7)));
  sites.emplace_back(llvm::cantFail(CompilerTarget::Site::create(19)));
  sites.emplace_back(llvm::cantFail(CompilerTarget::Site::create(42)));

  const auto target = llvm::cantFail(CompilerTarget::create(
      std::move(sites),
      std::vector<CompilerTarget::Coupling>{{7, 19}, {19, 42}},
      std::vector<CompilerTarget::Operation>{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  for (int64_t i = 0; i < size; ++i) {
    qubits[i] = builder.allocQubit();
  }

  std::tie(qubits[0], qubits[1]) = builder.cx(qubits[0], qubits[1]);
  std::tie(qubits[1], qubits[2]) = builder.cz(qubits[1], qubits[2]);
  std::tie(qubits[0], qubits[2]) = builder.cx(qubits[0], qubits[2]);
  for (int64_t i = 0; i < qubits.size(); ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
    builder.sink(qubits[i]);
  }

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  const DenseSet<CompilerTarget::SiteId> expectedSites{7, 19, 42};
  size_t numStatics = 0;
  m->walk([&](StaticOp op) {
    ++numStatics;
    EXPECT_TRUE(expectedSites.contains(op.getIndex()));
  });
  EXPECT_EQ(numStatics, 3);
}

TEST_F(MappingPassFixture, MapNoncontiguousTargetWithUnusedSites) {
  std::vector<CompilerTarget::Site> sites;
  sites.emplace_back(llvm::cantFail(CompilerTarget::Site::create(7)));
  sites.emplace_back(llvm::cantFail(CompilerTarget::Site::create(19)));
  sites.emplace_back(llvm::cantFail(CompilerTarget::Site::create(42)));
  const auto target = llvm::cantFail(CompilerTarget::create(std::move(sites)));

  QCOProgramBuilder builder(context.get());
  builder.initialize({builder.getI1Type()});
  auto qubit = builder.h(builder.allocQubit());
  Value bit;
  std::tie(qubit, bit) = builder.measure(qubit);
  builder.sink(qubit);
  auto module = builder.finalize(bit);

  PassManager pm(module->getContext());
  pm.addPass(createMappingPass(target, MappingPassOptions{.ntrials = 1}));
  ASSERT_TRUE(pm.run(module.get()).succeeded());
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(isExecutable(getEntryPoint(module.get()), target));

  const DenseSet<CompilerTarget::SiteId> expectedSites{7, 19, 42};
  size_t numAllocations = 0;
  size_t numStatics = 0;
  size_t numSinks = 0;
  module->walk([&](AllocOp) { ++numAllocations; });
  module->walk([&](StaticOp op) {
    ++numStatics;
    EXPECT_TRUE(expectedSites.contains(op.getIndex()));
  });
  module->walk([&](SinkOp) { ++numSinks; });
  EXPECT_EQ(numAllocations, 0);
  EXPECT_EQ(numStatics, 3);
  EXPECT_EQ(numSinks, 3);
}

TEST_F(MappingPassFixture, KeepWorkspaceSparseOnLargeTarget) {
  constexpr size_t numTargetQubits = 64;
  std::vector<CompilerTarget::Coupling> couplings;
  couplings.reserve(numTargetQubits - 1);
  for (size_t site = 1; site < numTargetQubits; ++site) {
    couplings.emplace_back(0, static_cast<int64_t>(site));
  }

  const auto target = llvm::cantFail(
      CompilerTarget::create(numTargetQubits, std::move(couplings)));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(2, builder.getI1Type()));

  SmallVector<Value> bits(2);
  Value q0 = builder.allocQubit();
  Value q1 = builder.allocQubit();
  std::tie(q0, q1) = builder.cx(q0, q1);
  std::tie(q0, bits[0]) = builder.measure(q0);
  std::tie(q1, bits[1]) = builder.measure(q1);
  builder.sink(q0);
  builder.sink(q1);

  auto m = builder.finalize(bits);
  ASSERT_TRUE(runPass(m.get(), target,
                      MappingPassOptions{.niterations = 1, .ntrials = 1})
                  .succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  size_t numStatics = 0;
  size_t numSinks = 0;
  m->walk([&](StaticOp) { ++numStatics; });
  m->walk([&](SinkOp) { ++numSinks; });
  EXPECT_GE(numStatics, 2);
  EXPECT_LE(numStatics, 3);
  EXPECT_LT(numStatics, numTargetQubits);
  EXPECT_EQ(numSinks, numStatics);
}

TEST_P(MappingPassTest, FailNoEntryPoint) {
  const auto& target = GetParam();

  OwningOpRef m = ModuleOp::create(UnknownLoc::get(context.get()));
  auto res = runPass(m.get(), target, MappingPassOptions{});
  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, MapScalarAllocation) {
  const auto& target = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize({builder.getI1Type()});

  Value q0;
  Value c0;
  q0 = builder.allocQubit();
  q0 = builder.h(q0);
  std::tie(q0, c0) = builder.measure(q0);
  builder.sink(q0);

  auto m = builder.finalize(c0);
  auto res = runPass(m.get(), target, MappingPassOptions{});

  ASSERT_TRUE(res.succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  size_t numAllocations = 0;
  size_t numStatics = 0;
  m->walk([&](AllocOp) { ++numAllocations; });
  m->walk([&](StaticOp) { ++numStatics; });
  EXPECT_EQ(numAllocations, 0);
  EXPECT_EQ(numStatics, 1);
}

TEST_P(MappingPassTest, MapMixedScalarAndTensorAllocations) {
  const auto& target = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value scalar = builder.allocQubit();
  Value tensor = builder.qtensorAlloc(2);
  Value tensorQubit0;
  Value tensorQubit1;
  std::tie(tensor, tensorQubit0) = builder.qtensorExtract(tensor, 0);
  std::tie(tensor, tensorQubit1) = builder.qtensorExtract(tensor, 1);

  scalar = builder.h(scalar);
  std::tie(scalar, tensorQubit0) = builder.cx(scalar, tensorQubit0);
  std::tie(tensorQubit0, tensorQubit1) =
      builder.rzx(0.5, tensorQubit0, tensorQubit1);

  builder.sink(scalar);
  tensor = builder.qtensorInsert(tensorQubit0, tensor, 0);
  tensor = builder.qtensorInsert(tensorQubit1, tensor, 1);
  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  size_t numScalarAllocations = 0;
  size_t numTensorAllocations = 0;
  m->walk([&](AllocOp) { ++numScalarAllocations; });
  m->walk([&](qtensor::AllocOp) { ++numTensorAllocations; });
  EXPECT_EQ(numScalarAllocations, 0);
  EXPECT_EQ(numTensorAllocations, 0);
}

TEST_P(MappingPassTest, MapProgramAfterQubitReuse) {
  const auto& target = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize({builder.getI1Type(), builder.getI1Type()});

  Value q0 = builder.allocQubit();
  q0 = builder.h(q0);
  Value bit0;
  std::tie(q0, bit0) = builder.measure(q0);
  builder.sink(q0);

  Value q1 = builder.allocQubit();
  q1 = builder.x(q1);
  Value bit1;
  std::tie(q1, bit1) = builder.measure(q1);
  builder.sink(q1);

  auto m = builder.finalize({bit0, bit1});
  PassManager pm(context.get());
  pm.addPass(createReuseQubits());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createMappingPass(target, MappingPassOptions{.ntrials = 1}));
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(pm.run(m.get()).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  size_t numStatics = 0;
  size_t numResets = 0;
  m->walk([&](StaticOp) { ++numStatics; });
  m->walk([&](ResetOp) { ++numResets; });
  EXPECT_EQ(numStatics, 1);
  EXPECT_EQ(numResets, 1);
}

TEST_P(MappingPassTest, FailNestedScalarAllocation) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %condition = arith.constant true
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.if %condition args(%arg0 = %q0) -> (!qco.qubit) {
          %nested = qco.alloc : !qco.qubit
          qco.sink %nested : !qco.qubit
          qco.yield %arg0 : !qco.qubit
        } else args(%arg0 = %q0) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(succeeded(verify(*m)));

  std::string diagnostics;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
    diagnostics += diagnostic.str();
    return success();
  });
  EXPECT_TRUE(failed(runPass(m.get(), target, MappingPassOptions{})));
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains(
              "target mapping requires dynamic qubit allocations in the entry "
              "function body"))
      << diagnostics;
}

TEST_P(MappingPassTest, FailNestedTensorAllocation) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %condition = arith.constant true
        %c1 = arith.constant 1 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.if %condition args(%arg0 = %q0) -> (!qco.qubit) {
          %nested = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
          qtensor.dealloc %nested : tensor<1x!qco.qubit>
          qco.yield %arg0 : !qco.qubit
        } else args(%arg0 = %q0) {
          qco.yield %arg0 : !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(succeeded(verify(*m)));

  std::string diagnostics;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
    diagnostics += diagnostic.str();
    return success();
  });
  EXPECT_TRUE(failed(runPass(m.get(), target, MappingPassOptions{})));
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains(
              "target mapping requires dynamic qubit allocations in the entry "
              "function body"))
      << diagnostics;
}

TEST_P(MappingPassTest, FailNestedHigherArityUnitary) {
  const auto& target = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();
  SmallVector<Value> qubits{builder.allocQubit(), builder.allocQubit(),
                            builder.allocQubit()};
  qubits = llvm::to_vector(builder.qcoIf(
      true, qubits,
      [&](ValueRange args) {
        auto [controls, targetQubit] = builder.mcx({args[0], args[1]}, args[2]);
        return SmallVector<Value>{controls[0], controls[1], targetQubit};
      },
      [](ValueRange args) { return llvm::to_vector(args); }));
  for (auto qubit : qubits) {
    builder.sink(qubit);
  }

  auto m = builder.finalize();
  std::string diagnostics;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
    diagnostics += diagnostic.str();
    return success();
  });
  EXPECT_TRUE(failed(runPass(m.get(), target, MappingPassOptions{})));
  EXPECT_TRUE(
      StringRef(diagnostics)
          .contains("decompose it to one- and two-qubit operations first"))
      << diagnostics;

  size_t numAllocations = 0;
  size_t numStatics = 0;
  m->walk([&](AllocOp) { ++numAllocations; });
  m->walk([&](StaticOp) { ++numStatics; });
  EXPECT_EQ(numAllocations, 3);
  EXPECT_EQ(numStatics, 0);
}

TEST_P(MappingPassTest, FailNoExtractAfterInsert) {
  const auto& target = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize({builder.getI1Type()});

  Value tensor0 = builder.qtensorAlloc(1);

  Value q0;
  Value c0;
  std::tie(tensor0, q0) = builder.qtensorExtract(tensor0, 0);
  q0 = builder.h(q0);
  tensor0 = builder.qtensorInsert(q0, tensor0, 0);

  std::tie(tensor0, q0) = builder.qtensorExtract(tensor0, 0);
  q0 = builder.x(q0);
  std::tie(q0, c0) = builder.measure(q0);
  tensor0 = builder.qtensorInsert(q0, tensor0, 0);

  builder.qtensorDealloc(tensor0);

  auto m = builder.finalize(c0);
  auto res = runPass(m.get(), target, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, FailTooManyQubitsForArch) {
  const auto& target = GetParam();
  const auto size = static_cast<int64_t>(target.numQubits()) + 1;

  SmallVector<Value> bits(size);
  SmallVector<Value> qubits(size);

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(size);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
    qubits[i] = builder.h(qubits[i]);
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  auto res = runPass(m.get(), target, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, MapFlatGHZ) {
  const auto& target = GetParam();
  const int64_t size = 3;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(3, builder.getI1Type()));

  auto tensor = builder.qtensorAlloc(3);
  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  flatGHZ(builder, qubits);

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapLoopBasedGHZByUnrolling) {
  const auto& target = GetParam();
  const auto size = static_cast<int64_t>(target.numQubits());

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addNestedPass<func::FuncOp>(createQuantumLoopUnroll());
  populateQCOCleanupPipeline(pm);
  pm.addPass(createMappingPass(target, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(size);

  loopGHZ(builder, tensor, size);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  ASSERT_TRUE(pm.run(m.get()).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapGroverLike) {
  const auto& target = GetParam();
  const int64_t size = 5;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(4);
  Value flagTensor = builder.qtensorAlloc(1);

  std::tie(tensor, qubits[0]) = builder.qtensorExtract(tensor, 0);
  std::tie(tensor, qubits[1]) = builder.qtensorExtract(tensor, 1);
  std::tie(tensor, qubits[2]) = builder.qtensorExtract(tensor, 2);
  std::tie(tensor, qubits[3]) = builder.qtensorExtract(tensor, 3);
  std::tie(flagTensor, qubits[4]) = builder.qtensorExtract(flagTensor, 0);

  qubits[0] = builder.h(qubits[0]);
  qubits[1] = builder.h(qubits[1]);
  qubits[2] = builder.h(qubits[2]);
  qubits[3] = builder.h(qubits[3]);
  qubits[4] = builder.x(qubits[4]);

  qubits =
      builder.scfFor(1, 3, 1, qubits, [&builder](Value, ValueRange iterArgs) {
        Value iterQ0 = iterArgs[0];
        Value iterQ1 = iterArgs[1];
        Value iterQ2 = iterArgs[2];
        Value iterQ3 = iterArgs[3];
        Value iterFlag = iterArgs[4];

        std::tie(iterQ0, iterQ2) = builder.cx(iterQ0, iterQ2);
        std::tie(iterQ2, iterQ3) = builder.cx(iterQ2, iterQ3);
        std::tie(iterQ3, iterQ0) = builder.cx(iterQ3, iterQ0);
        std::tie(iterQ0, iterFlag) = builder.cx(iterQ0, iterFlag);

        return SmallVector{iterQ0, iterQ1, iterQ2, iterQ3, iterFlag};
      });
  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  tensor = builder.qtensorInsert(qubits[0], tensor, 0);
  tensor = builder.qtensorInsert(qubits[1], tensor, 1);
  tensor = builder.qtensorInsert(qubits[2], tensor, 2);
  tensor = builder.qtensorInsert(qubits[3], tensor, 3);
  flagTensor = builder.qtensorInsert(qubits[4], flagTensor, 0);

  builder.qtensorDealloc(tensor);
  builder.qtensorDealloc(flagTensor);

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapParallelLoops) {
  const auto& target = GetParam();
  constexpr int64_t size = 6;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(size);
  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
    qubits[i] = builder.h(qubits[i]);
  }

  auto upForResults =
      builder.scfFor(1, 3, 1, {qubits[0], qubits[1], qubits[2]},
                     [&builder](Value, ValueRange iterArgs) {
                       Value iterQ0 = iterArgs[0];
                       Value iterQ1 = iterArgs[1];
                       Value iterQ2 = iterArgs[2];

                       std::tie(iterQ0, iterQ1) = builder.cx(iterQ0, iterQ1);
                       iterQ0 = builder.h(iterQ0);
                       std::tie(iterQ0, iterQ1) = builder.cz(iterQ0, iterQ1);
                       std::tie(iterQ1, iterQ2) = builder.cz(iterQ1, iterQ2);
                       std::tie(iterQ0, iterQ2) = builder.cx(iterQ0, iterQ2);

                       return SmallVector{iterQ0, iterQ1, iterQ2};
                     });

  qubits[0] = upForResults[0];
  qubits[1] = upForResults[1];
  qubits[2] = upForResults[2];

  auto downForResults =
      builder.scfFor(1, 3, 1, {qubits[3], qubits[4], qubits[5]},
                     [&builder](Value, ValueRange iterArgs) {
                       Value iterQ0 = iterArgs[0];
                       Value iterQ1 = iterArgs[1];
                       Value iterQ2 = iterArgs[2];

                       std::tie(iterQ0, iterQ1) = builder.cx(iterQ0, iterQ1);
                       iterQ0 = builder.h(iterQ0);
                       std::tie(iterQ1, iterQ2) = builder.cz(iterQ1, iterQ2);
                       std::tie(iterQ0, iterQ1) = builder.cz(iterQ0, iterQ1);
                       std::tie(iterQ0, iterQ2) = builder.cx(iterQ0, iterQ2);

                       return SmallVector{iterQ0, iterQ1, iterQ2};
                     });

  qubits[3] = downForResults[0];
  qubits[4] = downForResults[1];
  qubits[5] = downForResults[2];

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapParallelLoopsWithClassicalDependencies) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.alloc : !qco.qubit
        %q2 = qco.alloc : !qco.qubit
        %q3 = qco.alloc : !qco.qubit
        %q4 = qco.alloc : !qco.qubit
        %q5 = qco.alloc : !qco.qubit
        %q6 = qco.alloc : !qco.qubit
        %q7 = qco.alloc : !qco.qubit
        %a0, %a1, %s1 = scf.for %i = %c0 to %c1 step %c1
            iter_args(%x = %q0, %y = %q1, %s = %c1)
            -> (!qco.qubit, !qco.qubit, index) {
          %nx, %ny = qco.swap %x, %y
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          scf.yield %nx, %ny, %s : !qco.qubit, !qco.qubit, index
        }
        %b0, %b1, %s2 = scf.for %i = %c0 to %s1 step %c1
            iter_args(%x = %q2, %y = %q3, %s = %s1)
            -> (!qco.qubit, !qco.qubit, index) {
          %nx, %ny = qco.swap %x, %y
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          scf.yield %nx, %ny, %s : !qco.qubit, !qco.qubit, index
        }
        %d0, %d1, %s3 = scf.for %i = %c0 to %s2 step %c1
            iter_args(%x = %q4, %y = %q5, %s = %s2)
            -> (!qco.qubit, !qco.qubit, index) {
          %nx, %ny = qco.swap %x, %y
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          scf.yield %nx, %ny, %s : !qco.qubit, !qco.qubit, index
        }
        %e0, %e1, %s4 = scf.for %i = %c0 to %s3 step %c1
            iter_args(%x = %q6, %y = %q7, %s = %s3)
            -> (!qco.qubit, !qco.qubit, index) {
          %nx, %ny = qco.swap %x, %y
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          scf.yield %nx, %ny, %s : !qco.qubit, !qco.qubit, index
        }
        qco.sink %a0 : !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b0 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        qco.sink %d0 : !qco.qubit
        qco.sink %d1 : !qco.qubit
        qco.sink %e0 : !qco.qubit
        qco.sink %e1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(succeeded(verify(*m)));
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  EXPECT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapForWithClassicalIterArg) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() -> i64 attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %state = arith.constant 0 : i64
        %one = arith.constant 1 : i64
        %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %tensor1, %q0 = qtensor.extract %tensor0[%c0] : tensor<2x!qco.qubit>
        %tensor2, %q1 = qtensor.extract %tensor1[%c1] : tensor<2x!qco.qubit>
        %next_state, %next_q0, %next_q1 =
            scf.for %iv = %c0 to %c2 step %c1
                iter_args(%iter_state = %state, %iter_q0 = %q0,
                          %iter_q1 = %q1)
                -> (i64, !qco.qubit, !qco.qubit) {
          %updated_state = arith.addi %iter_state, %one : i64
          %updated_q0, %updated_q1 =
              qco.swap %iter_q0, %iter_q1
                  : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          scf.yield %updated_state, %updated_q0, %updated_q1
              : i64, !qco.qubit, !qco.qubit
        }
        %tensor3 = qtensor.insert %next_q0 into %tensor2[%c0]
            : tensor<2x!qco.qubit>
        %tensor4 = qtensor.insert %next_q1 into %tensor3[%c1]
            : tensor<2x!qco.qubit>
        qtensor.dealloc %tensor4 : tensor<2x!qco.qubit>
        return %next_state : i64
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(verify(*m).succeeded());
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  EXPECT_TRUE(verify(*m).succeeded());
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapTypeChangingWhileWithClassicalState) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() -> i64 attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %false = arith.constant false
        %state = arith.constant 0 : i32
        %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %tensor1, %q0 = qtensor.extract %tensor0[%c0] : tensor<2x!qco.qubit>
        %tensor2, %q1 = qtensor.extract %tensor1[%c1] : tensor<2x!qco.qubit>
        %next_state, %next_q0, %next_q1 =
            scf.while (%iter_state = %state, %iter_q0 = %q0,
                       %iter_q1 = %q1)
                : (i32, !qco.qubit, !qco.qubit)
                  -> (i64, !qco.qubit, !qco.qubit) {
          %extended_state = arith.extsi %iter_state : i32 to i64
          %updated_q0, %updated_q1 =
              qco.swap %iter_q0, %iter_q1
                  : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          scf.condition(%false) %extended_state, %updated_q0, %updated_q1
              : i64, !qco.qubit, !qco.qubit
        } do {
        ^bb0(%after_state: i64, %after_q0: !qco.qubit,
             %after_q1: !qco.qubit):
          %truncated_state = arith.trunci %after_state : i64 to i32
          scf.yield %truncated_state, %after_q0, %after_q1
              : i32, !qco.qubit, !qco.qubit
        }
        %tensor3 = qtensor.insert %next_q0 into %tensor2[%c0]
            : tensor<2x!qco.qubit>
        %tensor4 = qtensor.insert %next_q1 into %tensor3[%c1]
            : tensor<2x!qco.qubit>
        qtensor.dealloc %tensor4 : tensor<2x!qco.qubit>
        return %next_state : i64
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(verify(*m).succeeded());

  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  EXPECT_TRUE(verify(*m).succeeded());
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapIfWithClassicalResult) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main() -> i64 attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %tensor1, %q0 = qtensor.extract %tensor0[%c0]
            : tensor<2x!qco.qubit>
        %tensor2, %q1 = qtensor.extract %tensor1[%c1]
            : tensor<2x!qco.qubit>
        %q2 = qco.h %q0 : !qco.qubit -> !qco.qubit
        %q3, %condition = qco.measure %q2 : !qco.qubit
        %state, %q4, %q5 = qco.if %condition
            args(%arg0 = %q3, %arg1 = %q1)
            -> (i64, !qco.qubit, !qco.qubit) {
          %next0, %next1 = qco.swap %arg0, %arg1
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          %then = arith.constant 1 : i64
          qco.yield %then, %next0, %next1
              : i64, !qco.qubit, !qco.qubit
        } else args(%arg0 = %q3, %arg1 = %q1) {
          %else = arith.constant 2 : i64
          qco.yield %else, %arg0, %arg1
              : i64, !qco.qubit, !qco.qubit
        }
        %tensor3 = qtensor.insert %q4 into %tensor2[%c0]
            : tensor<2x!qco.qubit>
        %tensor4 = qtensor.insert %q5 into %tensor3[%c1]
            : tensor<2x!qco.qubit>
        qtensor.dealloc %tensor4 : tensor<2x!qco.qubit>
        return %state : i64
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(succeeded(verify(*m)));

  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  IfOp ifOp;
  m->walk([&](IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getClassicalResults().size(), 1);
  EXPECT_TRUE(ifOp.getClassicalResults().front().getType().isInteger(64));
  for (YieldOp yield : {ifOp.thenYield(), ifOp.elseYield()}) {
    ASSERT_EQ(yield.getNumOperands(), ifOp.getNumResults());
    EXPECT_TRUE(yield.getOperand(0).getType().isInteger(64));
  }
}

TEST_P(MappingPassTest, MapIndexSwitchWithClassicalResult) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%selector: index) -> i64
          attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %tensor1, %q0 = qtensor.extract %tensor0[%c0]
            : tensor<2x!qco.qubit>
        %tensor2, %q1 = qtensor.extract %tensor1[%c1]
            : tensor<2x!qco.qubit>
        %state, %q2, %q3 = qco.index_switch %selector
            -> (i64, !qco.qubit, !qco.qubit)
        case 0 args(%arg0 = %q0, %arg1 = %q1) {
          %next0, %next1 = qco.swap %arg0, %arg1
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          %case = arith.constant 1 : i64
          qco.yield %case, %next0, %next1
              : i64, !qco.qubit, !qco.qubit
        }
        case 1 args(%arg0 = %q0, %arg1 = %q1) {
          %next0, %next1 = qco.swap %arg0, %arg1
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          %case = arith.constant 2 : i64
          qco.yield %case, %next0, %next1
              : i64, !qco.qubit, !qco.qubit
        }
        default args(%arg0 = %q0, %arg1 = %q1) {
          %default = arith.constant 3 : i64
          qco.yield %default, %arg0, %arg1
              : i64, !qco.qubit, !qco.qubit
        }
        %tensor3 = qtensor.insert %q2 into %tensor2[%c0]
            : tensor<2x!qco.qubit>
        %tensor4 = qtensor.insert %q3 into %tensor3[%c1]
            : tensor<2x!qco.qubit>
        qtensor.dealloc %tensor4 : tensor<2x!qco.qubit>
        return %state : i64
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(succeeded(verify(*m)));

  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));

  IndexSwitchOp switchOp;
  m->walk([&](IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);
  ASSERT_EQ(switchOp.getClassicalResults().size(), 1);
  EXPECT_TRUE(switchOp.getClassicalResults().front().getType().isInteger(64));
  for (Region* region : switchOp.getRegions()) {
    auto yield = cast<YieldOp>(region->front().getTerminator());
    ASSERT_EQ(yield.getNumOperands(), switchOp.getNumResults());
    EXPECT_TRUE(yield.getOperand(0).getType().isInteger(64));
  }
}

TEST_P(MappingPassTest, MapIndexSwitchRegions) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%selector: index)
          attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %tensor0 = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
        %tensor1, %q0 = qtensor.extract %tensor0[%c0]
            : tensor<3x!qco.qubit>
        %tensor2, %q1 = qtensor.extract %tensor1[%c1]
            : tensor<3x!qco.qubit>
        %tensor3, %q2 = qtensor.extract %tensor2[%c2]
            : tensor<3x!qco.qubit>
        %q3, %q4, %q5 = qco.index_switch %selector
            -> (!qco.qubit, !qco.qubit, !qco.qubit)
        case 0 args(%arg0 = %q0, %arg1 = %q1, %arg2 = %q2) {
          %next0, %next1 = qco.swap %arg0, %arg1
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          qco.yield %next0, %next1, %arg2
              : !qco.qubit, !qco.qubit, !qco.qubit
        }
        case 1 args(%arg0 = %q0, %arg1 = %q1, %arg2 = %q2) {
          %next1, %next2 = qco.swap %arg1, %arg2
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          qco.yield %arg0, %next1, %next2
              : !qco.qubit, !qco.qubit, !qco.qubit
        }
        default args(%arg0 = %q0, %arg1 = %q1, %arg2 = %q2) {
          %next0, %next2 = qco.swap %arg0, %arg2
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          qco.yield %next0, %arg1, %next2
              : !qco.qubit, !qco.qubit, !qco.qubit
        }
        %tensor4 = qtensor.insert %q3 into %tensor3[%c0]
            : tensor<3x!qco.qubit>
        %tensor5 = qtensor.insert %q4 into %tensor4[%c1]
            : tensor<3x!qco.qubit>
        %tensor6 = qtensor.insert %q5 into %tensor5[%c2]
            : tensor<3x!qco.qubit>
        qtensor.dealloc %tensor6 : tensor<3x!qco.qubit>
        return
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(succeeded(verify(*m)));

  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));

  size_t numSwaps = 0;
  m->walk([&](SWAPOp) { ++numSwaps; });
  EXPECT_GT(numSwaps, 3);
}

TEST_P(MappingPassTest, MapNestedOperationOnceWhileIndependentWiresAdvance) {
  const auto& target = GetParam();
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @main(%selector: index)
          attributes {mqt.entry_point} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %tensor0 = qtensor.alloc(%c4) : tensor<4x!qco.qubit>
        %tensor1, %q0 = qtensor.extract %tensor0[%c0]
            : tensor<4x!qco.qubit>
        %tensor2, %q1 = qtensor.extract %tensor1[%c1]
            : tensor<4x!qco.qubit>
        %tensor3, %q2 = qtensor.extract %tensor2[%c2]
            : tensor<4x!qco.qubit>
        %tensor4, %q3 = qtensor.extract %tensor3[%c3]
            : tensor<4x!qco.qubit>
        %q4, %q5 = qco.index_switch %selector
            -> (!qco.qubit, !qco.qubit)
        case 0 args(%arg0 = %q0, %arg1 = %q1) {
          %next0, %next1 = qco.swap %arg0, %arg1
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          qco.yield %next0, %next1 : !qco.qubit, !qco.qubit
        }
        default args(%arg0 = %q0, %arg1 = %q1) {
          qco.yield %arg0, %arg1 : !qco.qubit, !qco.qubit
        }
        %q6, %q7 = qco.barrier %q2, %q3
            : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        %q8, %q9 = qco.barrier %q6, %q7
            : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        %tensor5 = qtensor.insert %q4 into %tensor4[%c0]
            : tensor<4x!qco.qubit>
        %tensor6 = qtensor.insert %q5 into %tensor5[%c1]
            : tensor<4x!qco.qubit>
        %tensor7 = qtensor.insert %q8 into %tensor6[%c2]
            : tensor<4x!qco.qubit>
        %tensor8 = qtensor.insert %q9 into %tensor7[%c3]
            : tensor<4x!qco.qubit>
        qtensor.dealloc %tensor8 : tensor<4x!qco.qubit>
        return
      }
    }
  )mlir";

  auto m = parseSourceString<ModuleOp>(source, context.get());
  ASSERT_TRUE(m);
  ASSERT_TRUE(succeeded(verify(*m)));

  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  EXPECT_TRUE(succeeded(verify(*m)));

  size_t numIndexSwitches = 0;
  m->walk([&](IndexSwitchOp) { ++numIndexSwitches; });
  EXPECT_EQ(numIndexSwitches, 1);
}

TEST_P(MappingPassTest, MapSABRECircuit) {
  const auto& target = GetParam();
  constexpr int64_t size = 6;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(6, builder.getI1Type()));

  Value tensorUp = builder.qtensorAlloc(4);
  Value tensorDown = builder.qtensorAlloc(2);

  std::tie(tensorUp, qubits[0]) = builder.qtensorExtract(tensorUp, 0);
  std::tie(tensorUp, qubits[1]) = builder.qtensorExtract(tensorUp, 1);
  std::tie(tensorUp, qubits[2]) = builder.qtensorExtract(tensorUp, 2);
  std::tie(tensorUp, qubits[3]) = builder.qtensorExtract(tensorUp, 3);
  std::tie(tensorDown, qubits[4]) = builder.qtensorExtract(tensorDown, 0);
  std::tie(tensorDown, qubits[5]) = builder.qtensorExtract(tensorDown, 1);

  qubits[0] = builder.h(qubits[0]);
  qubits[1] = builder.h(qubits[1]);
  qubits[4] = builder.h(qubits[4]);

  qubits[0] = builder.z(qubits[0]);
  std::tie(qubits[1], qubits[2]) = builder.cx(qubits[1], qubits[2]);
  std::tie(qubits[4], qubits[5]) = builder.cx(qubits[4], qubits[5]);

  std::tie(qubits[0], qubits[1]) = builder.cx(qubits[0], qubits[1]);

  qubits[0] = builder.h(qubits[0]);
  qubits[1] = builder.y(qubits[1]);
  std::tie(qubits[0], qubits[1]) = builder.cx(qubits[0], qubits[1]);

  std::tie(qubits[2], qubits[3]) = builder.cx(qubits[2], qubits[3]);

  qubits[2] = builder.h(qubits[2]);
  qubits[3] = builder.h(qubits[3]);

  std::tie(qubits[1], qubits[2]) = builder.cx(qubits[1], qubits[2]);
  std::tie(qubits[3], qubits[5]) = builder.cx(qubits[3], qubits[5]);

  qubits[3] = builder.z(qubits[3]);

  std::tie(qubits[3], qubits[4]) = builder.cx(qubits[3], qubits[4]);

  std::tie(qubits[3], qubits[0]) = builder.cx(qubits[3], qubits[0]);

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  tensorUp = builder.qtensorInsert(qubits[0], tensorUp, 0);
  tensorUp = builder.qtensorInsert(qubits[1], tensorUp, 1);
  tensorUp = builder.qtensorInsert(qubits[2], tensorUp, 2);
  tensorUp = builder.qtensorInsert(qubits[3], tensorUp, 3);
  tensorDown = builder.qtensorInsert(qubits[4], tensorDown, 0);
  tensorDown = builder.qtensorInsert(qubits[5], tensorDown, 1);

  builder.qtensorDealloc(tensorUp);
  builder.qtensorDealloc(tensorDown);

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapBranchingGHZ) {
  const auto& target = GetParam();
  constexpr int64_t size = 7;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(size);
  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  qubits[0] = builder.h(qubits[0]);
  std::tie(qubits[0], bits[0]) = builder.measure(qubits[0]);

  qubits = builder.qcoIf(
      bits[0], qubits,
      [&](ValueRange args) {
        SmallVector<Value> argQs(args);
        flatGHZ(builder, argQs);
        return argQs;
      },
      [&](ValueRange args) {
        SmallVector<Value> argQs(llvm::reverse(args));
        flatGHZ(builder, argQs);
        return argQs;
      });

  flatGHZ(builder, qubits);

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapDoUntil) {
  const auto& target = GetParam();
  const auto size = 4;

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(size);
  SmallVector<Value> qubits(size);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  qubits = builder.scfWhile(
      qubits,
      [&](ValueRange args) {
        SmallVector<Value> beforeArgs(args);
        SmallVector<Value> beforeBits(args);

        flatGHZ(builder, beforeArgs);

        beforeArgs = builder.barrier(beforeArgs);

        for (int64_t i = 0; i < size; ++i) {
          std::tie(beforeArgs[i], beforeBits[i]) =
              builder.measure(beforeArgs[i]);
        }

        for (int64_t i = 0; i < size - 1; ++i) {
          beforeBits[i + 1] =
              arith::AndIOp::create(builder, beforeBits[i], beforeBits[i + 1])
                  .getResult();
        }

        builder.scfCondition(beforeBits[size - 1], beforeArgs);
        return beforeArgs;
      },
      [&](ValueRange args) {
        SmallVector<Value> afterArgs(args);
        flatGHZ(builder, afterArgs);
        return afterArgs;
      });

  flatGHZ(builder, qubits);

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapNestedForSwitch) {
  const auto& target = GetParam();
  const auto size = 9;

  std::mt19937 gen(42);

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(size);
  SmallVector<Value> qubits(size);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  qubits = builder.scfFor(0, 1000, 1, qubits, [&](Value, ValueRange initArgs) {
    SmallVector<Value> args(initArgs);

    std::tie(args[0], args[3]) = builder.cx(args[0], args[3]);
    std::tie(args[0], args[6]) = builder.cx(args[0], args[6]);

    SmallVector<Value> t(ArrayRef<Value>(args).slice(0, 3));
    SmallVector<Value> m(ArrayRef<Value>(args).slice(3, 3));
    SmallVector<Value> b(ArrayRef<Value>(args).slice(6, 3));

    flatGHZ(builder, t);
    flatGHZ(builder, m);
    flatGHZ(builder, b);

    args.clear();
    args.append(t.begin(), t.end());
    args.append(m.begin(), m.end());
    args.append(b.begin(), b.end());

    args = builder.barrier(args);

    auto cnt = arith::ConstantIntOp::create(builder, builder.getI64Type(), 0)
                   .getResult();
    for (auto& arg : args) {
      Value bit;
      std::tie(arg, bit) = builder.measure(arg);
      cnt = arith::AddUIExtendedOp::create(
                builder, cnt,
                arith::ExtUIOp::create(builder, builder.getI64Type(), bit)
                    .getResult())
                .getSum();
    }
    auto index =
        arith::IndexCastOp::create(builder, builder.getIndexType(), cnt);

    const auto cases = to_vector(llvm::seq<int64_t>(1, size));

    SmallVector<std::function<SmallVector<Value>(ValueRange)>> bodies(
        cases.size());
    for (auto& body : bodies) {
      body = [&](ValueRange initArgs) {
        std::uniform_int_distribution<size_t> distrib(1UL, initArgs.size());
        const auto n = distrib(gen);
        SmallVector<Value> caseArgs(initArgs);
        for (size_t j = 1; j < n; ++j) {
          std::tie(caseArgs[0], caseArgs[j]) =
              builder.cx(caseArgs[0], caseArgs[j]);
        }
        return caseArgs;
      };
    }

    const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies(
        bodies.begin(), bodies.end());
    const auto defaultCase = [](auto initArgs) {
      return llvm::to_vector(initArgs);
    };

    return llvm::to_vector(
        builder.qcoIndexSwitch(index, args, cases, caseBodies, defaultCase));
  });

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

TEST_P(MappingPassTest, MapIndexSwitchUsesVotedLayout) {
  const auto target = llvm::cantFail(CompilerTarget::create(
      3, std::vector<CompilerTarget::Coupling>{{0, 1}, {1, 2}}));

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(3);
  SmallVector<Value> qubits(3);
  for (int64_t i = 0; i < 3; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  const auto routeTriangle = [&](ValueRange initArgs) {
    SmallVector<Value> args(initArgs);
    std::tie(args[0], args[1]) = builder.cx(args[0], args[1]);
    std::tie(args[1], args[2]) = builder.cx(args[1], args[2]);
    std::tie(args[0], args[2]) = builder.cx(args[0], args[2]);
    return args;
  };
  const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies(
      3, routeTriangle);
  qubits = llvm::to_vector(builder.qcoIndexSwitch(
      0, qubits, SmallVector<int64_t>{0, 1, 2}, caseBodies,
      [](ValueRange args) { return llvm::to_vector(args); }));

  for (int64_t i = 0; i < 3; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }
  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());

  size_t numSwaps = 0;
  m->walk([&](SWAPOp) { ++numSwaps; });
  // The three routed cases agree on the voted exit layout; only the default
  // case must be restored to it. Restoring every case to the parent needs 12.
  EXPECT_EQ(numSwaps, 6UL);
}

TEST_P(MappingPassTest, MapPaddedCXCZGrid) {
  const auto& target = GetParam();
  const auto size = (target.numQubits() + 1) / 2;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  for (size_t i = 0; i < size; ++i) {
    qubits[i] = builder.allocQubit();
  }
  cxcz(builder, qubits);
  for (int64_t i = 0; i < qubits.size(); ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
    builder.sink(qubits[i]);
  }

  auto m = builder.finalize(bits);
  ASSERT_TRUE(
      runPass(m.get(), target, MappingPassOptions{.ntrials = 1}).succeeded());
  ASSERT_TRUE(succeeded(verify(*m)));
  EXPECT_TRUE(isExecutable(getEntryPoint(m.get()), target));
}

INSTANTIATE_TEST_SUITE_P(ThreeByThreeSquareGrid, MappingPassTest,
                         testing::Values(getSquareGridTarget(3)));
INSTANTIATE_TEST_SUITE_P(FourByFourSquareGrid, MappingPassTest,
                         testing::Values(getSquareGridTarget(4)));
INSTANTIATE_TEST_SUITE_P(TenByTenSquareGrid, MappingPassTest,
                         testing::Values(getSquareGridTarget(10)));
