/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

namespace {
struct Device {
  size_t nqubits{};
  DenseSet<std::pair<size_t, size_t>> couplingSet;
};
} // namespace

/// Return true, if the operations within a region fulfill the given coupling
/// constraints.
static bool
isExecutable(Region& body, DenseMap<Value, size_t>& m,
             const DenseSet<std::pair<size_t, size_t>>& couplingSet) {
  for (Operation& op : body.getOps()) {
    if (auto staticOp = dyn_cast<StaticOp>(op)) {
      m.try_emplace(staticOp.getQubit(), staticOp.getIndex());
      continue;
    }

    if (auto unitaryOp = dyn_cast<UnitaryOpInterface>(op)) {
      if (!isa<BarrierOp>(op) && unitaryOp.getNumQubits() > 1) {
        assert(unitaryOp.getNumQubits() <= 2 && "expected two-qubit decomp.");

        const auto hwA = m.at(unitaryOp.getInputQubit(0));
        const auto hwB = m.at(unitaryOp.getInputQubit(1));
        if (!couplingSet.contains(std::make_pair(hwA, hwB))) {
          llvm::dbgs() << "The two-qubit gate (" << hwA << ", " << hwB
                       << ") is not executable: \n";
          unitaryOp->dump();
          return false;
        }
      }

      for (const auto [pred, succ] : llvm::zip_equal(
               unitaryOp.getInputQubits(), unitaryOp.getOutputQubits())) {
        m.try_emplace(succ, m.at(pred));
      }

      continue;
    }

    if (auto resetOp = dyn_cast<ResetOp>(op)) {
      m.try_emplace(resetOp.getQubitOut(), m.at(resetOp.getQubitIn()));
      continue;
    }

    if (auto measOp = dyn_cast<MeasureOp>(op)) {
      m.try_emplace(measOp.getQubitOut(), m.at(measOp.getQubitIn()));
      continue;
    }

    if (!isa<scf::ForOp, scf::WhileOp, qco::IfOp>(op)) {
      continue;
    }

    for (Region& region : op.getRegions()) {
      const ValueRange initArgs =
          TypeSwitch<Operation*, ValueRange>(region.getParentOp())
              .Case<qco::IfOp>([&](qco::IfOp ifOp) { return ifOp.getQubits(); })
              .Case<scf::WhileOp>(
                  [&](scf::WhileOp whileOp) { return whileOp.getInits(); })
              .Case<scf::ForOp>(
                  [&](scf::ForOp forOp) { return forOp.getInits(); })
              .Default([](Operation*) -> ValueRange { return {}; });

      const auto initialHardwareOrder =
          to_vector(llvm::map_range(initArgs, [&](auto v) { return m.at(v); }));

      const auto qubitArgs =
          llvm::make_filter_range(region.getArguments(), [](auto& arg) {
            return isa<QubitType>(arg.getType());
          });

      DenseMap<Value, size_t> localM;
      for (const auto [i, arg] : llvm::enumerate(qubitArgs)) {
        localM.try_emplace(arg, initialHardwareOrder[i]);
      }

      if (!isExecutable(region, localM, couplingSet)) {
        return false;
      }

      Operation* terminator = region.front().getTerminator();
      const ValueRange finalOrderArgs =
          TypeSwitch<Operation*, ValueRange>(region.getParentOp())
              .Case<qco::IfOp>([&](qco::IfOp) {
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

      const auto finalOrder = to_vector(llvm::map_range(
          finalOrderArgs, [&](auto v) { return localM.at(v); }));

      if (finalOrder != initialHardwareOrder) {
        llvm::dbgs()
            << "The hardware indices of the yielded terminator qubit values "
               "must be in the same order as parent's op input qubit values!\n";
        return false;
      }
    }

    for (OpResult res : op.getResults()) {
      const Value init = TypeSwitch<Operation*, Value>(&op)
                             .Case<scf::WhileOp>([&](scf::WhileOp whileOp) {
                               return whileOp.getInits()[res.getResultNumber()];
                             })
                             .Case<scf::ForOp>([&](scf::ForOp forOp) {
                               return forOp.getTiedLoopInit(res)->get();
                             })
                             .Case<qco::IfOp>([&](qco::IfOp ifOp) {
                               return ifOp.getTiedQubit(res)->get();
                             });
      m.try_emplace(res, m.at(init));
    }
  }

  return true;
}

/// Return true, if the entry point fulfills the given coupling constraints.
static bool
isExecutable(func::FuncOp entry,
             const DenseSet<std::pair<size_t, size_t>>& couplingSet) {
  DenseMap<Value, size_t> m;
  return isExecutable(entry.getFunctionBody(), m, couplingSet);
}

/// Return a 9x9 square-grid coupling set.
static Device getNineQubitSquareGrid() {
  return {.nqubits = 9,
          .couplingSet = {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1},
                          {1, 2}, {2, 1}, {2, 5}, {5, 2}, {3, 6}, {6, 3},
                          {3, 4}, {4, 3}, {4, 7}, {7, 4}, {4, 5}, {5, 4},
                          {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}}};
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

namespace {

class MappingPassTest : public testing::Test,
                        public testing::WithParamInterface<Device> {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static LogicalResult
  runPass(ModuleOp m, const DenseSet<std::pair<size_t, size_t>>& couplingSet,
          const MappingPassOptions& options) {
    PassManager pm(m->getContext());
    pm.addPass(createMappingPass(couplingSet, options));
    return pm.run(m);
  }

  std::unique_ptr<MLIRContext> context;
};

}; // namespace

TEST_P(MappingPassTest, FailNoEntryPoint) {
  const auto& device = GetParam();

  OwningOpRef m = ModuleOp::create(UnknownLoc::get(context.get()));
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});
  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, FailNoQubitAllocations) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize({builder.getI1Type()});

  Value q0;
  Value c0;
  q0 = builder.allocQubit();
  q0 = builder.h(q0);
  std::tie(q0, c0) = builder.measure(q0);
  builder.sink(q0);

  auto m = builder.finalize(c0);
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, FailNoExtractAfterInsert) {
  const auto& device = GetParam();

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
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, FailTooManyQubitsForArch) {
  const auto& device = GetParam();
  const auto size = static_cast<int64_t>(device.nqubits) + 1;

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
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, MapFlatGHZ) {
  const auto& device = GetParam();
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
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, MapLoopBasedGHZByUnrolling) {
  const auto& device = GetParam();
  const auto size = static_cast<int64_t>(device.nqubits);

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addNestedPass<func::FuncOp>(createQuantumLoopUnroll());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createMappingPass(device.couplingSet, MappingPassOptions{}));

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
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, MapGroverLike) {
  const auto& device = GetParam();
  const int64_t size = 5;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device.couplingSet, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(5, builder.getI1Type()));

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
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, MapParallelLoops) {
  const auto& device = GetParam();
  constexpr int64_t size = 6;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device.couplingSet, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(size);
  for (int64_t i = 0; i < size; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
    qubits[i] = builder.h(qubits[i]);
  }

  const auto upForResults =
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

  const auto downForResults =
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
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, MapSABRECircuit) {
  const auto& device = GetParam();
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
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, MapBranchingGHZ) {
  const auto& device = GetParam();
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
  auto res =
      runPass(m.get(), device.couplingSet, MappingPassOptions{.ntrials = 1});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, MapDoUntil) {
  const auto& device = GetParam();
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
      [&](ValueRange args) { return args; });

  flatGHZ(builder, qubits);

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res =
      runPass(m.get(), device.couplingSet, MappingPassOptions{.ntrials = 1});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

INSTANTIATE_TEST_SUITE_P(NineQubitSquareGrid, MappingPassTest,
                         testing::Values(getNineQubitSquareGrid()));
