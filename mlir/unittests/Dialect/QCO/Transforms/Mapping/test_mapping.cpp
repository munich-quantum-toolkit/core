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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
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
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

struct Device {
  size_t nqubits{};
  DenseSet<std::pair<size_t, size_t>> couplingSet;
};

/// Return true, if the operations within a region fulfill the given coupling
/// constraints.
static bool
isExecutable(Region& body, DenseMap<Value, size_t>& m,
             const DenseSet<std::pair<size_t, size_t>>& couplingSet) {
  for (Operation& rop : body.getOps()) {
    bool executable = true;
    TypeSwitch<Operation*>(&rop)
        .Case<StaticOp>(
            [&](StaticOp op) { m.try_emplace(op.getQubit(), op.getIndex()); })
        .Case<BarrierOp>([&](BarrierOp op) {
          for (const auto [pred, succ] :
               llvm::zip_equal(op.getInputQubits(), op.getOutputQubits())) {
            const auto hw = m.at(pred);
            m.try_emplace(succ, hw);
          }
        })
        .Case<UnitaryOpInterface>([&](UnitaryOpInterface& op) {
          assert(op.getNumQubits() <= 2 && "expected two-qubit decomp.");

          if (op.getNumQubits() > 1) {
            const auto hwA = m.at(op.getInputQubit(0));
            const auto hwB = m.at(op.getInputQubit(1));
            if (!couplingSet.contains(std::make_pair(hwA, hwB))) {
              llvm::dbgs() << "(" << hwA << ", " << hwB << ") "
                           << "not executable: \n";
              op->dump();
              executable = false;
            }
          }

          for (const auto [pred, succ] :
               llvm::zip_equal(op.getInputQubits(), op.getOutputQubits())) {
            const auto hw = m.at(pred);
            m.try_emplace(succ, hw);
          }
        })
        .Case<scf::ForOp>([&](scf::ForOp op) {
          DenseMap<Value, size_t> loopM;
          for (const auto [init, arg] :
               llvm::zip_equal(op.getInits(), op.getRegionIterArgs())) {
            const auto hw = m.at(init);
            loopM.try_emplace(arg, hw);
          }

          for (OpOperand& operand : op.getInitsMutable()) {
            const auto pred = operand.get();
            const auto succ = op.getTiedLoopResult(&operand);
            const auto hw = m.at(pred);
            m.try_emplace(succ, hw);
          }

          if (!isExecutable(op.getRegion(), loopM, couplingSet)) {
            executable = false;
            return;
          }

          for (const auto& [arg, yielded] :
               llvm::zip_equal(op.getRegionIterArgs(), op.getYieldedValues())) {
            if (loopM.at(arg) != loopM.at(yielded)) {
              llvm::dbgs() << "for loop layout not restored!\n";
              executable = false;
              return;
            }
          }
        })
        .Case<scf::YieldOp>([&](scf::YieldOp op) {
          assert(isa<scf::ForOp>(op->getParentOp()));
          auto forOp = cast<scf::ForOp>(op->getParentOp());
        })
        .Case<ResetOp, MeasureOp>([&](auto op) {
          const auto pred = op.getQubitIn();
          const auto succ = op.getQubitOut();
          const auto hw = m.at(pred);
          m.try_emplace(succ, hw);
        });

    if (!executable) {
      return false;
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

TEST_P(MappingPassTest, NoEntryPoint) {
  const auto& device = GetParam();

  OwningOpRef m = ModuleOp::create(UnknownLoc::get(context.get()));
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});
  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, NoQubitAllocations) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value q0 = builder.allocQubit();
  q0 = builder.h(q0);
  builder.sink(q0);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, NoExtractAfterInsert) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor0 = builder.qtensorAlloc(1);

  Value q0;
  std::tie(tensor0, q0) = builder.qtensorExtract(tensor0, 0);
  q0 = builder.h(q0);
  tensor0 = builder.qtensorInsert(q0, tensor0, 0);

  std::tie(tensor0, q0) = builder.qtensorExtract(tensor0, 0);
  q0 = builder.x(q0);
  tensor0 = builder.qtensorInsert(q0, tensor0, 0);

  builder.qtensorDealloc(tensor0);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, TooManyQubitsForArch) {
  const auto& device = GetParam();
  const auto n = static_cast<int64_t>(device.nqubits) + 1;

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(n);
  SmallVector<Value> qubits(n);
  for (int64_t i = 0; i < n; ++i) {
    Value qi;
    std::tie(tensor, qi) = builder.qtensorExtract(tensor, i);
    qi = builder.h(qi);
    qubits[i] = qi;
  }

  for (int64_t i = 0; i < n; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});

  ASSERT_TRUE(res.failed());
}

TEST_P(MappingPassTest, GHZ) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(3);

  Value q0;
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);

  Value q1;
  std::tie(tensor, q1) = builder.qtensorExtract(tensor, 1);

  Value q2;
  std::tie(tensor, q2) = builder.qtensorExtract(tensor, 2);

  q0 = builder.h(q0);
  std::tie(q0, q1) = builder.cx(q0, q1);
  std::tie(q0, q2) = builder.cx(q0, q2);

  tensor = builder.qtensorInsert(q0, tensor, 0);
  tensor = builder.qtensorInsert(q1, tensor, 1);
  tensor = builder.qtensorInsert(q2, tensor, 2);
  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, GHZUnrolled) {
  const auto& device = GetParam();
  const auto n = static_cast<int64_t>(device.nqubits);

  PassManager pm(context.get());
  pm.addNestedPass<func::FuncOp>(createQuantumLoopUnroll());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createMappingPass(device.couplingSet, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(n);
  Value q0;
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  q0 = builder.h(q0);
  tensor = builder.qtensorInsert(q0, tensor, 0);
  tensor = builder.scfFor(
      1, n, 1, {tensor}, [&builder](Value iv, ValueRange iterArgs) {
        Value loopTensor = iterArgs[0];
        Value ctrl;
        Value targ;

        std::tie(loopTensor, ctrl) = builder.qtensorExtract(loopTensor, 0);
        std::tie(loopTensor, targ) = builder.qtensorExtract(loopTensor, iv);

        std::tie(ctrl, targ) = builder.cx(ctrl, targ);

        loopTensor = builder.qtensorInsert(ctrl, loopTensor, 0);
        loopTensor = builder.qtensorInsert(targ, loopTensor, iv);

        return SmallVector{loopTensor};
      })[0];
  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, GroverLike) {
  const auto& device = GetParam();

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device.couplingSet, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(4);
  Value flagTensor = builder.qtensorAlloc(1);
  Value q0;
  Value q1;
  Value q2;
  Value q3;
  Value flag;

  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  std::tie(tensor, q1) = builder.qtensorExtract(tensor, 1);
  std::tie(tensor, q2) = builder.qtensorExtract(tensor, 2);
  std::tie(tensor, q3) = builder.qtensorExtract(tensor, 3);
  std::tie(flagTensor, flag) = builder.qtensorExtract(flagTensor, 0);

  q0 = builder.h(q0);
  q1 = builder.h(q1);
  q2 = builder.h(q2);
  q3 = builder.h(q3);
  flag = builder.x(flag);

  const auto forResults = builder.scfFor(
      1, 3, 1, {q0, q1, q2, q3, flag}, [&builder](Value, ValueRange iterArgs) {
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

  q0 = forResults[0];
  q1 = forResults[1];
  q2 = forResults[2];
  q3 = forResults[3];
  flag = forResults[4];

  const auto barrierResults = builder.barrier({q0, q1, q2, q3, flag});
  q0 = barrierResults[0];
  q1 = barrierResults[1];
  q2 = barrierResults[2];
  q3 = barrierResults[3];
  flag = barrierResults[4];

  Value c0;
  Value c1;
  Value c2;
  Value c3;
  Value c4;

  std::tie(q0, c0) = builder.measure(q0);
  std::tie(q1, c1) = builder.measure(q1);
  std::tie(q2, c2) = builder.measure(q2);
  std::tie(q3, c3) = builder.measure(q3);
  std::tie(flag, c4) = builder.measure(flag);

  tensor = builder.qtensorInsert(q0, tensor, 1);
  tensor = builder.qtensorInsert(q1, tensor, 2);
  tensor = builder.qtensorInsert(q2, tensor, 3);
  tensor = builder.qtensorInsert(q3, tensor, 4);
  flagTensor = builder.qtensorInsert(flag, flagTensor, 0);

  builder.qtensorDealloc(tensor);
  builder.qtensorDealloc(flagTensor);

  auto m = builder.finalize();
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, ParallelLoops) {
  constexpr int64_t nqubits = 6;
  const auto& device = GetParam();

  PassManager pm(context.get());
  pm.addPass(
      createMappingPass(device.couplingSet, MappingPassOptions{.ntrials = 1, .niterations=1}));

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensor = builder.qtensorAlloc(nqubits);
  SmallVector<Value> creg(nqubits);
  SmallVector<Value> qreg(nqubits);

  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qreg[i]) = builder.qtensorExtract(tensor, i);
    qreg[i] = builder.h(qreg[i]);
  }

  const auto upForResults =
      builder.scfFor(1, 3, 1, {qreg[0], qreg[1], qreg[2]},
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

  qreg[0] = upForResults[0];
  qreg[1] = upForResults[1];
  qreg[2] = upForResults[2];

  const auto downForResults =
      builder.scfFor(1, 3, 1, {qreg[3], qreg[4], qreg[5]},
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

  qreg[3] = downForResults[0];
  qreg[4] = downForResults[1];
  qreg[5] = downForResults[2];

  qreg = builder.barrier(qreg);

  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(qreg[i], creg[i]) = builder.measure(qreg[i]);
    qreg[i] = builder.h(qreg[i]);
  }

  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qreg[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize();
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

TEST_P(MappingPassTest, Sabre) {
  const auto& device = GetParam();

  QCOProgramBuilder builder(context.get());
  builder.initialize();

  Value tensorUp = builder.qtensorAlloc(4);
  Value tensorDown = builder.qtensorAlloc(2);

  Value q0;
  std::tie(tensorUp, q0) = builder.qtensorExtract(tensorUp, 0);

  Value q1;
  std::tie(tensorUp, q1) = builder.qtensorExtract(tensorUp, 1);

  Value q2;
  std::tie(tensorUp, q2) = builder.qtensorExtract(tensorUp, 2);

  Value q3;
  std::tie(tensorUp, q3) = builder.qtensorExtract(tensorUp, 3);

  Value q4;
  std::tie(tensorDown, q4) = builder.qtensorExtract(tensorDown, 0);

  Value q5;
  std::tie(tensorDown, q5) = builder.qtensorExtract(tensorDown, 1);

  q0 = builder.h(q0);
  q1 = builder.h(q1);
  q4 = builder.h(q4);

  q0 = builder.z(q0);
  std::tie(q1, q2) = builder.cx(q1, q2);
  std::tie(q4, q5) = builder.cx(q4, q5);

  std::tie(q0, q1) = builder.cx(q0, q1);

  q0 = builder.h(q0);
  q1 = builder.y(q1);
  std::tie(q0, q1) = builder.cx(q0, q1);

  std::tie(q2, q3) = builder.cx(q2, q3);

  q2 = builder.h(q2);
  q3 = builder.h(q3);

  std::tie(q1, q2) = builder.cx(q1, q2);
  std::tie(q3, q5) = builder.cx(q3, q5);

  q3 = builder.z(q3);

  std::tie(q3, q4) = builder.cx(q3, q4);

  std::tie(q3, q0) = builder.cx(q3, q0);

  ValueRange out = builder.barrier({q0, q1, q2, q3, q4, q5});
  q0 = out[0];
  q1 = out[1];
  q2 = out[2];
  q3 = out[3];
  q4 = out[4];
  q5 = out[5];

  Value c0;
  Value c1;
  Value c2;
  Value c3;
  Value c4;
  Value c5;

  std::tie(q0, c0) = builder.measure(q0);
  std::tie(q1, c1) = builder.measure(q1);
  std::tie(q2, c2) = builder.measure(q2);
  std::tie(q3, c3) = builder.measure(q3);
  std::tie(q4, c4) = builder.measure(q4);
  std::tie(q5, c5) = builder.measure(q5);

  tensorUp = builder.qtensorInsert(q0, tensorUp, 0);
  tensorUp = builder.qtensorInsert(q1, tensorUp, 1);
  tensorUp = builder.qtensorInsert(q2, tensorUp, 2);
  tensorUp = builder.qtensorInsert(q3, tensorUp, 3);
  tensorDown = builder.qtensorInsert(q4, tensorDown, 0);
  tensorDown = builder.qtensorInsert(q5, tensorDown, 1);
  builder.qtensorDealloc(tensorUp);
  builder.qtensorDealloc(tensorDown);

  auto m = builder.finalize();
  auto res = runPass(m.get(), device.couplingSet, MappingPassOptions{});
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device.couplingSet));
}

INSTANTIATE_TEST_SUITE_P(NineQubitSquareGrid, MappingPassTest,
                         testing::Values(getNineQubitSquareGrid()));
