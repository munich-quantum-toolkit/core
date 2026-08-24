/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ExactUnitaryTest.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::qco;

enum class GateKind : uint8_t { CX, CZ, ECR, ISWAP };

class PauliTwirlingTest : public testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;

  PauliTwirlingTest() : builder(&context) {}

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    builder.initialize();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> buildGate(const GateKind kind) {
    auto first = builder.staticQubit(0);
    auto second = builder.staticQubit(1);
    switch (kind) {
    case GateKind::CX:
      std::tie(first, second) = builder.cx(first, second);
      break;
    case GateKind::CZ:
      std::tie(first, second) = builder.cz(first, second);
      break;
    case GateKind::ECR:
      std::tie(first, second) = builder.ecr(first, second);
      break;
    case GateKind::ISWAP:
      std::tie(first, second) = builder.iswap(first, second);
      break;
    }
    builder.sink(first);
    builder.sink(second);
    return builder.finalize();
  }

  static LogicalResult runPass(ModuleOp moduleOp, const uint64_t seed) {
    PassManager pm(moduleOp.getContext());
    pm.addPass(createPauliTwirl2QGates({.seed = seed}));
    return pm.run(moduleOp);
  }

  static size_t countTopLevelPaulis(ModuleOp moduleOp) {
    size_t count = 0;
    moduleOp.walk([&](Operation* op) {
      if (isa<IdOp, XOp, YOp, ZOp>(op) && !op->getParentOfType<CtrlOp>() &&
          !op->getParentOfType<InvOp>() && !op->getParentOfType<PowOp>()) {
        ++count;
      }
    });
    return count;
  }

  static std::string print(ModuleOp moduleOp) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    moduleOp.print(stream);
    return result;
  }
};

class AllPauliTwirlRowsTest
    : public PauliTwirlingTest,
      public testing::WithParamInterface<std::tuple<GateKind, uint64_t>> {};

TEST_P(AllPauliTwirlRowsTest, PreservesExactUnitary) {
  const auto [gate, seed] = GetParam();
  auto module = buildGate(gate);
  OwningOpRef<ModuleOp> original = cast<ModuleOp>(module->clone());

  ASSERT_TRUE(succeeded(runPass(*module, seed)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(countTopLevelPaulis(*module), 4);
  ::mqt::test::expectFullUnitaryEqual(*original, *module, 2);
}

TEST_F(PauliTwirlingTest, SameSeedProducesSameProgram) {
  auto source = buildGate(GateKind::CX);
  OwningOpRef<ModuleOp> first = cast<ModuleOp>(source->clone());
  OwningOpRef<ModuleOp> second = cast<ModuleOp>(source->clone());

  ASSERT_TRUE(succeeded(runPass(*first, 12345)));
  ASSERT_TRUE(succeeded(runPass(*second, 12345)));
  EXPECT_EQ(print(*first), print(*second));
}

TEST_F(PauliTwirlingTest, LeavesUnsupportedModifiedGatesAndPhasesUnchanged) {
  auto q0 = builder.staticQubit(0);
  auto q1 = builder.staticQubit(1);
  auto q2 = builder.staticQubit(2);
  auto [controls, target] = builder.mcx({q0, q1}, q2);
  builder.sink(controls[0]);
  builder.sink(controls[1]);
  builder.sink(target);

  auto q3 = builder.staticQubit(3);
  auto q4 = builder.staticQubit(4);
  auto [swap0, swap1] = builder.swap(q3, q4);
  builder.sink(swap0);
  builder.sink(swap1);

  auto q5 = builder.staticQubit(5);
  auto q6 = builder.staticQubit(6);
  const auto [emptyControl, emptyTarget] =
      builder.ctrl(q5, q6, [](const Value target) { return target; });
  builder.sink(emptyControl);
  builder.sink(emptyTarget);

  auto q7 = builder.staticQubit(7);
  auto q8 = builder.staticQubit(8);
  const auto [hControl, hTarget] = builder.ctrl(
      q7, q8, [&](const Value target) { return builder.h(target); });
  builder.sink(hControl);
  builder.sink(hTarget);

  auto q9 = builder.staticQubit(9);
  auto q10 = builder.staticQubit(10);
  const auto inverse = builder.inv({q9, q10}, [&](ValueRange qubits) {
    auto [control, nestedTarget] = builder.cx(qubits[0], qubits[1]);
    return SmallVector<Value>{control, nestedTarget};
  });
  builder.sink(inverse[0]);
  builder.sink(inverse[1]);

  builder.gphase(0.25);
  builder.gphase(0.5);

  auto module = builder.finalize();
  const auto original = print(*module);

  ASSERT_TRUE(succeeded(runPass(*module, 42)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(countTopLevelPaulis(*module), 0);
  EXPECT_EQ(print(*module), original);
}

constexpr std::array<uint64_t, 16> SEEDS_FOR_EACH_TWIRL = {
    6, 16, 10, 11, 12, 18, 5, 4, 1, 8, 62, 3, 2, 94, 0, 13};

INSTANTIATE_TEST_SUITE_P(
    SupportedGates, AllPauliTwirlRowsTest,
    testing::Combine(testing::Values(GateKind::CX, GateKind::CZ, GateKind::ECR,
                                     GateKind::ISWAP),
                     testing::ValuesIn(SEEDS_FOR_EACH_TWIRL)));

} // namespace
