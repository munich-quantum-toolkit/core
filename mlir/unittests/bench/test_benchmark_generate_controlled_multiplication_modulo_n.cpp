/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.h"
#include "bench/ControlledMultiplicationModuloN.hpp"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>

namespace mqt::bench {

using namespace mlir;

static void expectIndexConstant(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(constant);
  EXPECT_EQ(constant.value(), expected);
}

static void expectFloatConstant(Value value, double expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constant);
  auto attribute = dyn_cast<FloatAttr>(constant.getValue());
  ASSERT_TRUE(attribute);
  EXPECT_DOUBLE_EQ(attribute.getValueAsDouble(), expected);
}

static void expectStaticLoop(scf::ForOp loop, int64_t lower, int64_t upper) {
  expectIndexConstant(loop.getLowerBound(), lower);
  expectIndexConstant(loop.getUpperBound(), upper);
  expectIndexConstant(loop.getStep(), 1);
}

static void expectIntegerConstant(Value value, const llvm::APInt& expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constant);
  auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
  ASSERT_TRUE(attribute);
  ASSERT_EQ(attribute.getValue().getBitWidth(), expected.getBitWidth());
  EXPECT_EQ(attribute.getValue(), expected);
}

[[nodiscard]] static Value integerConstant(ModuleOp moduleOp,
                                           const llvm::APInt& expected) {
  Value result;
  moduleOp.walk([&](arith::ConstantOp constant) {
    auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
    if (attribute &&
        attribute.getValue().getBitWidth() == expected.getBitWidth() &&
        attribute.getValue() == expected &&
        isa<IntegerType>(constant.getType())) {
      result = constant.getResult();
    }
  });
  EXPECT_TRUE(result);
  return result;
}

static void expectRegisterLoad(Value value, Value expectedRegister,
                               Value expectedIndex) {
  auto load = value.getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(load);
  EXPECT_EQ(load.getMemref(), expectedRegister);
  ASSERT_EQ(load.getIndices().size(), 1U);
  EXPECT_EQ(load.getIndices().front(), expectedIndex);
}

static void expectQFTStage(scf::ForOp loop, Value expectedRegister,
                           int64_t width, bool inverse) {
  expectStaticLoop(loop, 0, width);

  qc::HOp hadamard;
  scf::ForOp phases;
  for (Operation& operation : loop.getBody()->without_terminator()) {
    if (auto candidate = dyn_cast<qc::HOp>(operation)) {
      ASSERT_FALSE(hadamard);
      hadamard = candidate;
    } else if (auto candidate = dyn_cast<scf::ForOp>(operation)) {
      ASSERT_FALSE(phases);
      phases = candidate;
    }
  }
  ASSERT_TRUE(hadamard);
  ASSERT_TRUE(phases);
  EXPECT_EQ(phases->isBeforeInBlock(hadamard), inverse);

  auto hadamardLoad = hadamard.getQubit(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(hadamardLoad);
  EXPECT_EQ(hadamardLoad.getMemref(), expectedRegister);

  SmallVector<qc::CtrlOp> rotations;
  phases.walk([&](qc::CtrlOp rotation) { rotations.push_back(rotation); });
  ASSERT_EQ(rotations.size(), 1U);
  auto control =
      rotations.front().getControl(0).getDefiningOp<memref::LoadOp>();
  auto target = rotations.front().getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(control);
  ASSERT_TRUE(target);
  EXPECT_EQ(control.getMemref(), expectedRegister);
  EXPECT_EQ(target.getMemref(), expectedRegister);
}

static void expectPhaseLayer(scf::ForOp loop, Value accumulator, Value addend,
                             llvm::ArrayRef<Value> expectedControls,
                             bool inverse, int64_t width) {
  expectStaticLoop(loop, 0, width);
  ASSERT_EQ(loop.getInitArgs().size(), 2U);
  expectFloatConstant(loop.getInitArgs().front(), 0.);
  EXPECT_EQ(loop.getInitArgs()[1], addend);

  SmallVector<qc::POp> phases;
  SmallVector<qc::CtrlOp> modifiers;
  loop.walk([&](qc::POp phase) { phases.push_back(phase); });
  loop.walk([&](qc::CtrlOp modifier) { modifiers.push_back(modifier); });
  ASSERT_EQ(phases.size(), 1U);
  ASSERT_EQ(modifiers.size(), expectedControls.empty() ? 0U : 1U);

  Value angle = phases.front().getTheta();
  if (inverse) {
    auto negated = angle.getDefiningOp<arith::MulFOp>();
    ASSERT_TRUE(negated);
    expectFloatConstant(negated.getRhs(), -1.);
    angle = negated.getLhs();
  }

  auto selectAngle = angle.getDefiningOp<scf::IfOp>();
  ASSERT_TRUE(selectAngle);
  ASSERT_EQ(selectAngle.getNumResults(), 1U);
  auto thenYield = dyn_cast<scf::YieldOp>(
      selectAngle.getThenRegion().front().getTerminator());
  auto elseYield = dyn_cast<scf::YieldOp>(
      selectAngle.getElseRegion().front().getTerminator());
  ASSERT_TRUE(thenYield);
  ASSERT_TRUE(elseYield);
  ASSERT_EQ(thenYield.getNumOperands(), 1U);
  ASSERT_EQ(elseYield.getNumOperands(), 1U);

  auto sum = thenYield.getOperand(0).getDefiningOp<arith::AddFOp>();
  ASSERT_TRUE(sum);
  auto decayed = elseYield.getOperand(0).getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(decayed);
  EXPECT_EQ(sum.getLhs(), decayed.getResult());
  expectFloatConstant(sum.getRhs(), std::numbers::pi);
  EXPECT_EQ(decayed.getLhs(), loop.getRegionIterArg(0));
  expectFloatConstant(decayed.getRhs(), 0.5);

  auto hasBit = selectAngle.getCondition().getDefiningOp<arith::CmpIOp>();
  ASSERT_TRUE(hasBit);
  EXPECT_EQ(hasBit.getPredicate(), arith::CmpIPredicate::ne);
  auto masked = hasBit.getLhs().getDefiningOp<arith::AndIOp>();
  ASSERT_TRUE(masked);
  EXPECT_EQ(masked.getLhs(), loop.getRegionIterArg(1));
  expectIntegerConstant(
      hasBit.getRhs(),
      llvm::APInt(cast<IntegerType>(addend.getType()).getWidth(), 0));
  expectIntegerConstant(
      masked.getRhs(),
      llvm::APInt(cast<IntegerType>(addend.getType()).getWidth(), 1));

  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  ASSERT_TRUE(yield);
  ASSERT_EQ(yield.getNumOperands(), 2U);
  EXPECT_EQ(yield.getOperand(0), angle);
  auto next = yield.getOperand(1).getDefiningOp<arith::ShRUIOp>();
  ASSERT_TRUE(next);
  EXPECT_EQ(next.getLhs(), loop.getRegionIterArg(1));
  expectIntegerConstant(
      next.getRhs(),
      llvm::APInt(cast<IntegerType>(addend.getType()).getWidth(), 1));

  if (expectedControls.empty()) {
    expectRegisterLoad(phases.front().getQubit(0), accumulator,
                       loop.getInductionVar());
    return;
  }

  auto modifier = modifiers.front();
  ASSERT_EQ(modifier.getNumControls(), expectedControls.size());
  ASSERT_EQ(modifier.getNumTargets(), 1U);
  for (size_t index = 0; index < expectedControls.size(); ++index) {
    EXPECT_EQ(modifier.getControl(index), expectedControls[index]);
  }
  expectRegisterLoad(modifier.getTarget(0), accumulator,
                     loop.getInductionVar());
}

[[nodiscard]] static qc::CtrlOp expectControlledX(Operation* operation) {
  auto modifier = dyn_cast<qc::CtrlOp>(operation);
  EXPECT_TRUE(modifier);
  if (!modifier) {
    return {};
  }
  EXPECT_EQ(modifier.getNumControls(), 1U);
  EXPECT_EQ(modifier.getNumTargets(), 1U);
  size_t xGates = 0;
  modifier.walk([&](qc::XOp /*unused*/) { ++xGates; });
  EXPECT_EQ(xGates, 1U);
  return modifier;
}

TEST(GenerateProgramTest, EmitsExactControlledMultiplicationModuloNSchedule) {
  constexpr int64_t bits = 3;
  constexpr int64_t width = bits + 1;
  auto program = generate(
      ControlledMultiplicationModuloN{{.multiplier = "011", .modulus = "101"}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(test::countOps<qc::AllocOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<memref::AllocOp>(moduleOp), 2U);
  EXPECT_EQ(test::countOps<cbit::AllocOp>(moduleOp), 1U);
  EXPECT_EQ(test::countOps<qc::SWAPOp>(moduleOp), 0U);
  EXPECT_EQ(test::countOps<qc::RZOp>(moduleOp), 0U);
  EXPECT_EQ(test::countOps<qc::MeasureOp>(moduleOp), 3U);

  cbit::AllocOp resultAllocation;
  moduleOp.walk(
      [&](cbit::AllocOp allocation) { resultAllocation = allocation; });
  ASSERT_TRUE(resultAllocation);
  EXPECT_EQ(resultAllocation.getResult().getType().getWidth(), 2 * bits + 2);

  scf::ForOp multiplication;
  moduleOp.walk([&](scf::ForOp loop) {
    if (loop.getInitArgs().size() == 1U &&
        isa<IntegerType>(loop.getInitArgs().front().getType())) {
      multiplication = loop;
    }
  });
  ASSERT_TRUE(multiplication);
  expectStaticLoop(multiplication, 0, bits);

  auto multiplier = integerConstant(moduleOp, llvm::APInt(width, 3));
  auto modulus = integerConstant(moduleOp, llvm::APInt(width, 5));
  ASSERT_EQ(multiplication.getInitArgs().size(), 1U);
  EXPECT_EQ(multiplication.getInitArgs().front(), multiplier);
  auto currentAddend = multiplication.getRegionIterArg(0);

  SmallVector<Operation*> schedule;
  for (Operation& operation : multiplication.getBody()->without_terminator()) {
    if (isa<scf::ForOp, qc::CtrlOp, qc::XOp>(operation)) {
      schedule.push_back(&operation);
    }
  }
  ASSERT_EQ(schedule.size(), 13U);

  auto firstAdd = dyn_cast<scf::ForOp>(schedule[0]);
  ASSERT_TRUE(firstAdd);
  qc::CtrlOp firstModifier;
  firstAdd.walk([&](qc::CtrlOp modifier) { firstModifier = modifier; });
  ASSERT_TRUE(firstModifier);
  ASSERT_EQ(firstModifier.getNumControls(), 2U);
  auto control = firstModifier.getControl(0);
  auto multiplicandBit = firstModifier.getControl(1);
  auto multiplicandLoad = multiplicandBit.getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(multiplicandLoad);
  EXPECT_EQ(multiplicandLoad.getIndices().front(),
            multiplication.getInductionVar());
  auto multiplicand = multiplicandLoad.getMemref();
  auto accumulatorLoad =
      firstModifier.getTarget(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(accumulatorLoad);
  auto accumulator = accumulatorLoad.getMemref();

  const SmallVector<Value, 2> controls{control, multiplicandBit};
  expectPhaseLayer(firstAdd, accumulator, currentAddend, controls, false,
                   width);
  expectPhaseLayer(cast<scf::ForOp>(schedule[1]), accumulator, modulus, {},
                   true, width);

  auto firstInverse = cast<scf::ForOp>(schedule[2]);
  expectQFTStage(firstInverse, accumulator, width, true);

  auto firstCarry = expectControlledX(schedule[3]);
  ASSERT_TRUE(firstCarry);
  auto work = firstCarry.getTarget(0);
  auto overflow = firstCarry.getControl(0);
  auto overflowLoad = overflow.getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(overflowLoad);
  EXPECT_EQ(overflowLoad.getMemref(), accumulator);
  expectIndexConstant(overflowLoad.getIndices().front(), width - 1);

  auto firstForward = cast<scf::ForOp>(schedule[4]);
  expectQFTStage(firstForward, accumulator, width, false);
  expectPhaseLayer(cast<scf::ForOp>(schedule[5]), accumulator, modulus, {work},
                   false, width);
  expectPhaseLayer(cast<scf::ForOp>(schedule[6]), accumulator, currentAddend,
                   controls, true, width);

  auto secondInverse = cast<scf::ForOp>(schedule[7]);
  expectQFTStage(secondInverse, accumulator, width, true);

  auto firstOverflowX = cast<qc::XOp>(schedule[8]);
  ASSERT_TRUE(firstOverflowX);
  expectRegisterLoad(firstOverflowX.getQubit(0), accumulator,
                     overflowLoad.getIndices().front());
  auto secondCarry = expectControlledX(schedule[9]);
  ASSERT_TRUE(secondCarry);
  EXPECT_EQ(secondCarry.getTarget(0), work);
  expectRegisterLoad(secondCarry.getControl(0), accumulator,
                     overflowLoad.getIndices().front());
  auto secondOverflowX = cast<qc::XOp>(schedule[10]);
  ASSERT_TRUE(secondOverflowX);
  expectRegisterLoad(secondOverflowX.getQubit(0), accumulator,
                     overflowLoad.getIndices().front());

  auto secondForward = cast<scf::ForOp>(schedule[11]);
  expectQFTStage(secondForward, accumulator, width, false);
  expectPhaseLayer(cast<scf::ForOp>(schedule[12]), accumulator, currentAddend,
                   controls, false, width);

  auto multiplicationYield =
      dyn_cast<scf::YieldOp>(multiplication.getBody()->getTerminator());
  ASSERT_TRUE(multiplicationYield);
  ASSERT_EQ(multiplicationYield.getNumOperands(), 1U);
  auto remainder =
      multiplicationYield.getOperand(0).getDefiningOp<arith::RemUIOp>();
  ASSERT_TRUE(remainder);
  EXPECT_EQ(remainder.getRhs(), modulus);
  auto doubled = remainder.getLhs().getDefiningOp<arith::ShLIOp>();
  ASSERT_TRUE(doubled);
  EXPECT_EQ(doubled.getLhs(), currentAddend);
  expectIntegerConstant(doubled.getRhs(), llvm::APInt(width, 1));

  SmallVector<scf::ForOp> outerQFTs;
  moduleOp.walk([&](scf::ForOp loop) {
    if (loop == multiplication || loop->getParentOfType<scf::ForOp>()) {
      return;
    }
    SmallVector<qc::CtrlOp> rotations;
    loop.walk([&](qc::CtrlOp rotation) { rotations.push_back(rotation); });
    if (!rotations.empty()) {
      outerQFTs.push_back(loop);
    }
  });
  ASSERT_EQ(outerQFTs.size(), 2U);
  EXPECT_TRUE(outerQFTs.front()->isBeforeInBlock(multiplication));
  EXPECT_TRUE(multiplication->isBeforeInBlock(outerQFTs.back()));
  expectQFTStage(outerQFTs.front(), accumulator, width, false);
  expectQFTStage(outerQFTs.back(), accumulator, width, true);

  qc::HOp controlPreparation;
  scf::ForOp multiplicandPreparation;
  moduleOp.walk([&](qc::HOp h) {
    auto load = h.getQubit(0).getDefiningOp<memref::LoadOp>();
    if (!load && !h->getParentOfType<scf::ForOp>()) {
      controlPreparation = h;
    } else if (load && load.getMemref() == multiplicand) {
      multiplicandPreparation = h->getParentOfType<scf::ForOp>();
    }
  });
  ASSERT_TRUE(controlPreparation);
  ASSERT_TRUE(multiplicandPreparation);
  EXPECT_EQ(controlPreparation.getQubit(0), control);
  expectStaticLoop(multiplicandPreparation, 0, bits);
  qc::HOp multiplicandH;
  multiplicandPreparation.walk([&](qc::HOp h) { multiplicandH = h; });
  ASSERT_TRUE(multiplicandH);
  expectRegisterLoad(multiplicandH.getQubit(0), multiplicand,
                     multiplicandPreparation.getInductionVar());

  qc::MeasureOp accumulatorMeasurement;
  qc::MeasureOp multiplicandMeasurement;
  qc::MeasureOp controlMeasurement;
  moduleOp.walk([&](qc::MeasureOp measurement) {
    auto load = measurement.getQubit().getDefiningOp<memref::LoadOp>();
    if (!load) {
      controlMeasurement = measurement;
    } else if (load.getMemref() == accumulator) {
      accumulatorMeasurement = measurement;
    } else if (load.getMemref() == multiplicand) {
      multiplicandMeasurement = measurement;
    }
  });
  ASSERT_TRUE(accumulatorMeasurement);
  ASSERT_TRUE(multiplicandMeasurement);
  ASSERT_TRUE(controlMeasurement);

  auto accumulatorMeasurementLoop =
      accumulatorMeasurement->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(accumulatorMeasurementLoop);
  expectStaticLoop(accumulatorMeasurementLoop, 0, width);
  expectRegisterLoad(accumulatorMeasurement.getQubit(), accumulator,
                     accumulatorMeasurementLoop.getInductionVar());
  auto accumulatorStore =
      dyn_cast<cbit::StoreOp>(*accumulatorMeasurement.getResult().user_begin());
  ASSERT_TRUE(accumulatorStore);
  EXPECT_EQ(accumulatorStore.getReg(), resultAllocation.getResult());
  EXPECT_EQ(accumulatorStore.getIndex(),
            accumulatorMeasurementLoop.getInductionVar());

  auto multiplicandMeasurementLoop =
      multiplicandMeasurement->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(multiplicandMeasurementLoop);
  expectStaticLoop(multiplicandMeasurementLoop, 0, bits);
  expectRegisterLoad(multiplicandMeasurement.getQubit(), multiplicand,
                     multiplicandMeasurementLoop.getInductionVar());
  auto multiplicandStore = dyn_cast<cbit::StoreOp>(
      *multiplicandMeasurement.getResult().user_begin());
  ASSERT_TRUE(multiplicandStore);
  EXPECT_EQ(multiplicandStore.getReg(), resultAllocation.getResult());
  auto multiplicandResultIndex =
      multiplicandStore.getIndex().getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(multiplicandResultIndex);
  EXPECT_EQ(multiplicandResultIndex.getLhs(),
            multiplicandMeasurementLoop.getInductionVar());
  expectIndexConstant(multiplicandResultIndex.getRhs(), width);

  EXPECT_EQ(controlMeasurement.getQubit(), control);
  auto controlStore =
      dyn_cast<cbit::StoreOp>(*controlMeasurement.getResult().user_begin());
  ASSERT_TRUE(controlStore);
  EXPECT_EQ(controlStore.getReg(), resultAllocation.getResult());
  expectIndexConstant(controlStore.getIndex(), 2 * bits + 1);
}

TEST(GenerateProgramTest, KeepsLargestControlledMultiplicationStructured) {
  constexpr size_t bits = ControlledMultiplicationModuloNOptions::MAX_BITS;
  const auto multiplier = std::string(bits - 1U, '0') + "1";
  const auto modulus = "1" + std::string(bits - 1U, '0');
  auto program = generate(ControlledMultiplicationModuloN{
      {.multiplier = multiplier, .modulus = modulus}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  scf::ForOp multiplication;
  moduleOp.walk([&](scf::ForOp loop) {
    if (loop.getInitArgs().size() == 1U &&
        isa<IntegerType>(loop.getInitArgs().front().getType())) {
      multiplication = loop;
    }
  });
  ASSERT_TRUE(multiplication);
  EXPECT_EQ(cast<IntegerType>(multiplication.getInitArgs().front().getType())
                .getWidth(),
            bits + 1U);
  EXPECT_LT(test::countOperations(moduleOp), 250U);
}

TEST(GenerateProgramTest,
     SamplesControlledMultiplicationModuloNAgainstReference) {
  test::expectSamplingMatchesReference(
      ControlledMultiplicationModuloN{{.multiplier = "011", .modulus = "101"}});
  test::expectSamplingMatchesReference(
      ControlledMultiplicationModuloN{{.multiplier = "010", .modulus = "100"}});
}

} // namespace mqt::bench
