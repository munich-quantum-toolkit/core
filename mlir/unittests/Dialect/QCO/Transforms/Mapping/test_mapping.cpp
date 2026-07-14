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
#include "mlir/Support/SuperconductingDevice.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
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
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

/// Return true, if the operations within a region fulfill the given coupling
/// constraints.
static bool isExecutable(Region& body, DenseMap<Value, size_t>& m,
                         const std::shared_ptr<SuperconductingDevice>& device) {
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
            if (!device->areAdjacent(hwA, hwB)) {
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
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
          DenseMap<Value, size_t> loopM;
          for (const auto [init, arg] :
               llvm::zip_equal(forOp.getInits(), forOp.getRegionIterArgs())) {
            const auto hw = m.at(init);
            loopM.try_emplace(arg, hw);
          }

          for (OpOperand& operand : forOp.getInitsMutable()) {
            const auto pred = operand.get();
            const auto succ = forOp.getTiedLoopResult(&operand);
            const auto hw = m.at(pred);
            m.try_emplace(succ, hw);
          }

          if (!isExecutable(forOp.getRegion(), loopM, device)) {
            executable = false;
            return;
          }

          for (const auto& [arg, yielded] : llvm::zip_equal(
                   forOp.getRegionIterArgs(), forOp.getYieldedValues())) {
            if (loopM.at(arg) != loopM.at(yielded)) {
              llvm::dbgs() << "scf::forOp: layout not restored!\n";
              executable = false;
              return;
            }
          }
        })
        .Case<qco::IfOp>([&](qco::IfOp ifOp) {
          std::array mappings{DenseMap<Value, size_t>{},
                              DenseMap<Value, size_t>{}};

          const std::array regions{&ifOp.getThenRegion(),
                                   &ifOp.getElseRegion()};

          for (size_t i = 0; i < 2; ++i) {
            for (const auto [init, arg] : llvm::zip_equal(
                     ifOp.getQubits(), regions[i]->getArguments())) {
              const auto hw = m.at(init);
              mappings[i].try_emplace(arg, hw);
            }
          }

          for (OpOperand& operand : ifOp.getQubitsMutable()) {
            const auto pred = operand.get();
            const auto succ = ifOp.getTiedResult(&operand);
            const auto hw = m.at(pred);
            m.try_emplace(succ, hw);
          }

          std::array<SmallVector<size_t>, 2> finalPermutation{};

          for (size_t i = 0; i < 2; ++i) {
            Region* body = regions[i];
            if (!isExecutable(*body, mappings[i], device)) {
              executable = false;
              return;
            }

            Block& block = body->getBlocks().front();
            auto yield = cast<qco::YieldOp>(block.getTerminator());
            for (const auto v : yield.getTargets()) {
              finalPermutation[i].emplace_back(mappings[i].at(v));
            }
          }

          if (finalPermutation[0] != finalPermutation[1]) {
            executable = false;
            return;
          }
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
static bool isExecutable(func::FuncOp entry,
                         const std::shared_ptr<SuperconductingDevice>& device) {
  DenseMap<Value, size_t> m;
  return isExecutable(entry.getFunctionBody(), m, device);
}

/// Return a 3x3 square-grid coupling set.
static std::shared_ptr<SuperconductingDevice> getNineQubitSquareGrid() {
  SmallVector<size_t> qubits(9);
  std::iota(qubits.begin(), qubits.end(), 0);

  DenseSet<std::pair<size_t, size_t>> coupling(
      {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1}, {1, 2}, {2, 1},
       {2, 5}, {5, 2}, {3, 6}, {6, 3}, {3, 4}, {4, 3}, {4, 7}, {7, 4},
       {4, 5}, {5, 4}, {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}});

  return std::make_shared<SuperconductingDevice>(qubits, coupling);
}

namespace {

class MappingPassTest : public testing::Test,
                        public testing::WithParamInterface<
                            std::shared_ptr<SuperconductingDevice>> {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  std::unique_ptr<MLIRContext> context;
};

}; // namespace

TEST_P(MappingPassTest, NoEntryPoint) {
  const auto& device = GetParam();

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

  OwningOpRef m = ModuleOp::create(UnknownLoc::get(context.get()));
  ASSERT_TRUE(pm.run(*m).failed());
}

TEST_P(MappingPassTest, NoQubitAllocations) {
  const auto& device = GetParam();

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize({builder.getI1Type()});

  Value q0;
  Value c0;
  q0 = builder.allocQubit();
  q0 = builder.h(q0);
  std::tie(q0, c0) = builder.measure(q0);
  builder.sink(q0);

  auto m = builder.finalize(c0);
  ASSERT_TRUE(pm.run(*m).failed());
}

TEST_P(MappingPassTest, NoExtractAfterInsert) {
  const auto& device = GetParam();

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

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
  ASSERT_TRUE(pm.run(*m).failed());
}

TEST_P(MappingPassTest, TooManyQubitsForArch) {
  const auto& device = GetParam();
  const auto size = static_cast<int64_t>(device->nqubits()) + 1;

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

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
  ASSERT_TRUE(pm.run(*m).failed());
}

TEST_P(MappingPassTest, GHZ) {
  const auto& device = GetParam();
  const int64_t size = 3;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(3, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(3);
  std::tie(tensor, qubits[0]) = builder.qtensorExtract(tensor, 0);
  std::tie(tensor, qubits[1]) = builder.qtensorExtract(tensor, 1);
  std::tie(tensor, qubits[2]) = builder.qtensorExtract(tensor, 2);

  qubits[0] = builder.h(qubits[0]);
  std::tie(qubits[0], qubits[1]) = builder.cx(qubits[0], qubits[1]);
  std::tie(qubits[0], qubits[2]) = builder.cx(qubits[0], qubits[2]);

  std::tie(qubits[0], bits[0]) = builder.measure(qubits[0]);
  std::tie(qubits[1], bits[1]) = builder.measure(qubits[1]);
  std::tie(qubits[2], bits[2]) = builder.measure(qubits[2]);

  tensor = builder.qtensorInsert(qubits[0], tensor, 0);
  tensor = builder.qtensorInsert(qubits[1], tensor, 1);
  tensor = builder.qtensorInsert(qubits[2], tensor, 2);

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  auto res = pm.run(*m);
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device));
}

TEST_P(MappingPassTest, GHZUnrolled) {
  const auto& device = GetParam();
  const auto size = static_cast<int64_t>(device->nqubits());

  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addNestedPass<func::FuncOp>(createQuantumLoopUnroll());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(size, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(size);
  Value q0;
  std::tie(tensor, q0) = builder.qtensorExtract(tensor, 0);
  q0 = builder.h(q0);
  tensor = builder.qtensorInsert(q0, tensor, 0);
  tensor = builder.scfFor(
      1, size, 1, {tensor}, [&builder](Value iv, ValueRange iterArgs) {
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

  for (int64_t i = 0; i < size; ++i) {
    Value q;
    std::tie(tensor, q) = builder.qtensorExtract(tensor, i);
    std::tie(q, bits[i]) = builder.measure(q);
    tensor = builder.qtensorInsert(q, tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device));
}

TEST_P(MappingPassTest, GroverLike) {
  const auto& device = GetParam();
  const int64_t size = 5;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

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
  EXPECT_TRUE(isExecutable(entry, device));
}

TEST_P(MappingPassTest, ParallelLoops) {
  const auto& device = GetParam();
  constexpr int64_t size = 6;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

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
    qubits[i] = builder.h(qubits[i]);
  }

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device));
}

TEST_P(MappingPassTest, Sabre) {
  const auto& device = GetParam();
  constexpr int64_t size = 6;

  SmallVector<Value> qubits(size);
  SmallVector<Value> bits(size);

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

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
  auto res = pm.run(m.get());
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device));
}

TEST_P(MappingPassTest, RandomOrderGHZ) {
  const auto& device = GetParam();
  constexpr int64_t size = 9;

  PassManager pm(context.get());
  pm.addPass(createMappingPass(device, MappingPassOptions{}));

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
        SmallVector<Value> values(args);
        values[0] = builder.h(values[0]);
        for (size_t i = 1; i < size; ++i) {
          std::tie(values[0], values[i]) = builder.cx(values[0], values[i]);
        }
        return values;
      },
      [&](ValueRange args) {
        SmallVector<Value> values(args);
        values[size - 1] = builder.h(values[size - 1]);
        for (size_t i = size - 1; i > 0; --i) {
          std::tie(values[8], values[i - 1]) =
              builder.cx(values[8], values[i - 1]);
        }
        return values;
      });

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < size; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  for (int64_t i = 0; i < size; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto m = builder.finalize(bits);
  auto res = pm.run(*m);
  auto entry = getEntryPoint(m.get());

  ASSERT_TRUE(res.succeeded());
  EXPECT_TRUE(isExecutable(entry, device));
}

INSTANTIATE_TEST_SUITE_P(NineQubitSquareGrid, MappingPassTest,
                         testing::Values(getNineQubitSquareGrid()));
