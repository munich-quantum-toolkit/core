/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QPE.hpp"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <utility>
#include <variant>

namespace mqt::bench {

using namespace mlir;

template <class Op> [[nodiscard]] static size_t countOps(ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&](Op) { ++count; });
  return count;
}

template <class Benchmark>
static void expectValidQCAndJeff(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  EXPECT_TRUE(program->isValid());
  auto compiled = runDefaultPipeline(CompilerInput{std::move(*program)},
                                     ProgramFormat::Jeff);
  ASSERT_TRUE(compiled);
  EXPECT_TRUE(std::holds_alternative<JeffProgram>(*compiled));
}

TEST(GenerateProgramTest, GeneratesEveryBenchmarkMethodAsQCAndJeff) {
  expectValidQCAndJeff(BV{{.hiddenBitstring = "101"}});
  expectValidQCAndJeff(
      BV{{.hiddenBitstring = "101", .method = BVMethod::Dynamic}});
  expectValidQCAndJeff(GHZ{{.qubits = 3}});
  expectValidQCAndJeff(Grover{{.markedBitstring = "101"}});
  expectValidQCAndJeff(Multiplexer{{.qubits = 3}});
  expectValidQCAndJeff(QFT{{.qubits = 3, .periodExponent = 1}});
  expectValidQCAndJeff(QFT{
      {.qubits = 3, .periodExponent = 1, .method = QFTMethod::Semiclassical}});
  expectValidQCAndJeff(QPE{{.precision = 3, .phase = Phase(3, 8)}});
  expectValidQCAndJeff(QPE{
      {.precision = 3, .phase = Phase(3, 8), .method = QPEMethod::Iterative}});
}

TEST(GenerateProgramTest, OmitsAllocationAdjacentResets) {
  EXPECT_EQ(countOps<qc::ResetOp>(generate(GHZ{{.qubits = 3}})->module()), 0U);
  EXPECT_EQ(countOps<qc::ResetOp>(
                generate(Grover{{.markedBitstring = "101"}})->module()),
            0U);
  EXPECT_EQ(
      countOps<qc::ResetOp>(generate(Multiplexer{{.qubits = 3}})->module()),
      0U);
  EXPECT_EQ(
      countOps<qc::ResetOp>(generate(BV{{.hiddenBitstring = "101"}})->module()),
      0U);
  EXPECT_EQ(countOps<qc::ResetOp>(
                generate(QFT{{.qubits = 3, .periodExponent = 1}})->module()),
            0U);
  EXPECT_EQ(
      countOps<qc::ResetOp>(
          generate(QPE{{.precision = 3, .phase = Phase(3, 8)}})->module()),
      0U);

  EXPECT_GT(countOps<qc::ResetOp>(generate(BV{{.hiddenBitstring = "101",
                                               .method = BVMethod::Dynamic}})
                                      ->module()),
            0U);
  EXPECT_GT(
      countOps<qc::ResetOp>(generate(QFT{{.qubits = 3,
                                          .periodExponent = 1,
                                          .method = QFTMethod::Semiclassical}})
                                ->module()),
      0U);
  EXPECT_GT(
      countOps<qc::ResetOp>(generate(QPE{{.precision = 3,
                                          .phase = Phase(3, 8),
                                          .method = QPEMethod::Iterative}})
                                ->module()),
      0U);
}

TEST(GenerateProgramTest, EmitsStructuredBVWithMethodSpecificResources) {
  const BV staticBenchmark{{.hiddenBitstring = "101"}};
  const BV dynamicBenchmark{
      {.hiddenBitstring = "101", .method = BVMethod::Dynamic}};
  auto staticProgram = generate(staticBenchmark);
  auto dynamicProgram = generate(dynamicBenchmark);
  ASSERT_TRUE(staticProgram);
  ASSERT_TRUE(dynamicProgram);

  EXPECT_EQ(countOps<qc::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(countOps<memref::AllocOp>(staticProgram->module()), 1U);
  EXPECT_EQ(countOps<qc::AllocOp>(dynamicProgram->module()), 2U);
  EXPECT_EQ(countOps<memref::AllocOp>(dynamicProgram->module()), 0U);
  EXPECT_EQ(countOps<tensor::ExtractOp>(staticProgram->module()), 1U);
  EXPECT_EQ(countOps<tensor::ExtractOp>(dynamicProgram->module()), 1U);

  const auto checkIndexing = [](ModuleOp moduleOp) {
    tensor::ExtractOp secret;
    moduleOp.walk([&](tensor::ExtractOp op) { secret = op; });
    ASSERT_TRUE(secret);
    auto loop = secret->getParentOfType<scf::ForOp>();
    ASSERT_TRUE(loop);
    EXPECT_EQ(secret.getIndices().front(), loop.getInductionVar());

    qc::MeasureOp measure;
    moduleOp.walk([&](qc::MeasureOp op) { measure = op; });
    ASSERT_TRUE(measure);
    auto measurementLoop = measure->getParentOfType<scf::ForOp>();
    ASSERT_TRUE(measurementLoop);
    auto store = dyn_cast<cbit::StoreOp>(*measure.getResult().user_begin());
    ASSERT_TRUE(store);
    EXPECT_EQ(store.getIndex(), measurementLoop.getInductionVar());
  };
  checkIndexing(staticProgram->module());
  checkIndexing(dynamicProgram->module());
}

[[nodiscard]] static DenseElementsAttr angleTable(ModuleOp moduleOp) {
  DenseElementsAttr result;
  moduleOp.walk([&](arith::ConstantOp op) {
    if (const auto table = dyn_cast<DenseElementsAttr>(op.getValue())) {
      EXPECT_FALSE(result);
      result = table;
    }
  });
  return result;
}

TEST(GenerateProgramTest, EmitsConfiguredGHZWithoutEagerRegisterLoads) {
  const GHZ benchmark(
      {.qubits = 64, .topology = GHZTopology::Star, .basis = GHZBasis::X});
  auto program = generate(benchmark);
  ASSERT_TRUE(program);

  EXPECT_LT(countOps<memref::LoadOp>(program->module()), 10U);
}

TEST(GenerateProgramTest, EmitsDirectGroverOracleWithBigEndianMarkedState) {
  const Grover benchmark({.markedBitstring = "01", .iterations = 2});
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(countOps<qc::AllocOp>(moduleOp), 0U);
  EXPECT_EQ(countOps<qc::CtrlOp>(moduleOp), 2U);

  scf::ForOp iterations;
  moduleOp.walk([&](qc::CtrlOp op) {
    if (!iterations) {
      iterations = op->getParentOfType<scf::ForOp>();
    }
  });
  ASSERT_TRUE(iterations);

  size_t markedStateFlips = 0;
  iterations.walk([&](qc::XOp op) {
    if (op->getParentOp() != iterations.getOperation()) {
      return;
    }
    ++markedStateFlips;
    auto load = op.getQubit(0).getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(load);
    auto index =
        load.getIndices().front().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(index);
    EXPECT_EQ(index.value(), 1);
  });
  EXPECT_EQ(markedStateFlips, 2U);
}

TEST(GenerateProgramTest, EmitsStructuredQuantumMultiplexer) {
  auto program = generate(Multiplexer{{.qubits = 3}});
  ASSERT_TRUE(program);
  auto moduleOp = program->module();

  EXPECT_EQ(countOps<scf::ForOp>(moduleOp), 4U);
  EXPECT_EQ(countOps<scf::IfOp>(moduleOp), 2U);
  EXPECT_EQ(countOps<qc::CtrlOp>(moduleOp), 1U);
  EXPECT_EQ(countOps<qc::RYOp>(moduleOp), 1U);
  EXPECT_EQ(countOps<qc::ResetOp>(moduleOp), 0U);
  EXPECT_EQ(countOps<tensor::ExtractOp>(moduleOp), 0U);

  scf::ForOp stateLoop;
  moduleOp.walk([&](scf::ForOp loop) {
    if (!loop.getInitArgs().empty()) {
      stateLoop = loop;
    }
  });
  ASSERT_TRUE(stateLoop);
  ASSERT_EQ(stateLoop.getInitArgs().size(), 1U);
  auto lower =
      stateLoop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto upper =
      stateLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = stateLoop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(lower);
  ASSERT_TRUE(upper);
  ASSERT_TRUE(step);
  EXPECT_EQ(lower.value(), 0);
  EXPECT_EQ(upper.value(), 4);
  EXPECT_EQ(step.value(), 1);
  auto firstAngle =
      stateLoop.getInitArgs().front().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(firstAngle);
  EXPECT_DOUBLE_EQ(cast<FloatAttr>(firstAngle.getValue()).getValueAsDouble(),
                   0.);

  auto yield = cast<scf::YieldOp>(stateLoop.getBody()->getTerminator());
  ASSERT_EQ(yield.getResults().size(), 1U);
  auto advance = yield.getResults().front().getDefiningOp<arith::AddFOp>();
  ASSERT_TRUE(advance);
  EXPECT_EQ(advance.getLhs(), stateLoop.getRegionIterArg(0));
  auto increment = advance.getRhs().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(increment);
  EXPECT_DOUBLE_EQ(cast<FloatAttr>(increment.getValue()).getValueAsDouble(),
                   std::numbers::pi / 4.);

  qc::CtrlOp controlledRotation;
  stateLoop.walk([&](qc::CtrlOp op) { controlledRotation = op; });
  ASSERT_TRUE(controlledRotation);
  EXPECT_EQ(controlledRotation.getNumControls(), 2U);
  ASSERT_EQ(controlledRotation.getNumBodyUnitaries(), 1U);

  SmallVector<scf::ForOp> flipLoops;
  for (auto loop : stateLoop.getBody()->getOps<scf::ForOp>()) {
    flipLoops.push_back(loop);
  }
  ASSERT_EQ(flipLoops.size(), 2U);
  ASSERT_EQ(flipLoops.front()->getBlock(), controlledRotation->getBlock());
  ASSERT_EQ(flipLoops.back()->getBlock(), controlledRotation->getBlock());
  EXPECT_TRUE(flipLoops.front()->isBeforeInBlock(controlledRotation));
  EXPECT_TRUE(controlledRotation->isBeforeInBlock(flipLoops.back()));
  for (auto loop : flipLoops) {
    auto bitLower =
        loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto bitUpper =
        loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto bitStep = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(bitLower);
    ASSERT_TRUE(bitUpper);
    ASSERT_TRUE(bitStep);
    EXPECT_EQ(bitLower.value(), 0);
    EXPECT_EQ(bitUpper.value(), 2);
    EXPECT_EQ(bitStep.value(), 1);
  }

  Value controlsStorage;
  for (size_t index = 0; index < controlledRotation.getNumControls(); ++index) {
    auto controlLoad =
        controlledRotation.getControl(index).getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(controlLoad);
    if (!controlsStorage) {
      controlsStorage = controlLoad.getMemref();
    } else {
      EXPECT_EQ(controlLoad.getMemref(), controlsStorage);
    }
    auto controlIndex = controlLoad.getIndices()
                            .front()
                            .getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(controlIndex);
    EXPECT_EQ(controlIndex.value(), static_cast<int64_t>(index));
  }
  ASSERT_TRUE(controlsStorage);
  EXPECT_TRUE(controlledRotation.getTarget(0).getDefiningOp<qc::AllocOp>());

  auto rotation =
      dyn_cast<qc::RYOp>(controlledRotation.getBodyUnitary(0).getOperation());
  ASSERT_TRUE(rotation);
  EXPECT_EQ(rotation.getTheta(), stateLoop.getRegionIterArg(0));

  size_t checkedConditionals = 0;
  stateLoop.walk([&](scf::IfOp conditional) {
    auto bitLoop = conditional->getParentOfType<scf::ForOp>();
    ASSERT_TRUE(bitLoop);
    EXPECT_EQ(bitLoop->getParentOfType<scf::ForOp>(), stateLoop);
    auto comparison = conditional.getCondition().getDefiningOp<arith::CmpIOp>();
    ASSERT_TRUE(comparison);
    EXPECT_EQ(comparison.getPredicate(), arith::CmpIPredicate::eq);
    auto bit = comparison.getLhs().getDefiningOp<arith::AndIOp>();
    ASSERT_TRUE(bit);
    auto zero = comparison.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    auto mask = bit.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(zero);
    ASSERT_TRUE(mask);
    EXPECT_EQ(zero.value(), 0);
    EXPECT_EQ(mask.value(), 1);
    auto shifted = bit.getLhs().getDefiningOp<arith::ShRSIOp>();
    ASSERT_TRUE(shifted);
    EXPECT_EQ(shifted.getLhs(), stateLoop.getInductionVar());
    EXPECT_EQ(shifted.getRhs(), bitLoop.getInductionVar());

    qc::XOp flip;
    conditional.walk([&](qc::XOp op) { flip = op; });
    ASSERT_TRUE(flip);
    auto load = flip.getQubit(0).getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(load);
    EXPECT_EQ(load.getMemref(), controlsStorage);
    EXPECT_EQ(load.getIndices().front(), bitLoop.getInductionVar());
    ++checkedConditionals;
  });
  EXPECT_EQ(checkedConditionals, 2U);

  size_t targetMeasurements = 0;
  size_t controlMeasurements = 0;
  Value resultRegister;
  moduleOp.walk([&](qc::MeasureOp measure) {
    ASSERT_FALSE(measure.getResult().use_empty());
    auto store = dyn_cast<cbit::StoreOp>(*measure.getResult().user_begin());
    ASSERT_TRUE(store);
    if (!resultRegister) {
      resultRegister = store.getReg();
    } else {
      EXPECT_EQ(store.getReg(), resultRegister);
    }
    if (auto loop = measure->getParentOfType<scf::ForOp>()) {
      auto measurementLower =
          loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
      auto measurementUpper =
          loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
      auto measurementStep =
          loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
      ASSERT_TRUE(measurementLower);
      ASSERT_TRUE(measurementUpper);
      ASSERT_TRUE(measurementStep);
      EXPECT_EQ(measurementLower.value(), 0);
      EXPECT_EQ(measurementUpper.value(), 2);
      EXPECT_EQ(measurementStep.value(), 1);

      auto controlLoad = measure.getQubit().getDefiningOp<memref::LoadOp>();
      ASSERT_TRUE(controlLoad);
      EXPECT_EQ(controlLoad.getMemref(), controlsStorage);
      EXPECT_EQ(controlLoad.getIndices().front(), loop.getInductionVar());

      auto resultIndex = store.getIndex().getDefiningOp<arith::AddIOp>();
      ASSERT_TRUE(resultIndex);
      EXPECT_EQ(resultIndex.getLhs(), loop.getInductionVar());
      auto offset =
          resultIndex.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      ASSERT_TRUE(offset);
      EXPECT_EQ(offset.value(), 1);
      ++controlMeasurements;
      return;
    }
    EXPECT_EQ(measure.getQubit(), controlledRotation.getTarget(0));
    auto resultIndex = store.getIndex().getDefiningOp<arith::ConstantIndexOp>();
    ASSERT_TRUE(resultIndex);
    EXPECT_EQ(resultIndex.value(), 0);
    ++targetMeasurements;
  });
  EXPECT_EQ(targetMeasurements, 1U);
  EXPECT_EQ(controlMeasurements, 1U);
}

TEST(GenerateProgramTest, KeepsLargeQuantumMultiplexerStructured) {
  auto program =
      generate(Multiplexer{{.qubits = MultiplexerOptions::MAX_QUBITS}});
  ASSERT_TRUE(program);
  scf::ForOp stateLoop;
  program->module().walk([&](scf::ForOp loop) {
    if (!loop.getInitArgs().empty()) {
      stateLoop = loop;
    }
  });
  ASSERT_TRUE(stateLoop);
  auto upper =
      stateLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(upper);
  EXPECT_EQ(upper.value(), int64_t{1} << (MultiplexerOptions::MAX_QUBITS - 1));

  size_t operations = 0;
  program->module().walk([&](Operation*) { ++operations; });
  EXPECT_LT(operations, 150U);

  auto compiled = runDefaultPipeline(CompilerInput{std::move(*program)},
                                     ProgramFormat::Jeff);
  ASSERT_TRUE(compiled);
  EXPECT_TRUE(std::holds_alternative<JeffProgram>(*compiled));
}

TEST(GenerateProgramTest, KeepsStandardQPEPowerAndResultOrderAligned) {
  const QPE benchmark({.precision = 2, .phase = Phase(1, 4)});
  EXPECT_DOUBLE_EQ(benchmark.probability("01"), 1.);

  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  auto table = angleTable(moduleOp);
  ASSERT_TRUE(table);
  const auto angles = llvm::to_vector(table.getValues<double>());
  ASSERT_EQ(angles.size(), 2U);
  EXPECT_NEAR(angles[0], std::numbers::pi / 2., 1e-15);
  EXPECT_NEAR(angles[1], std::numbers::pi, 1e-15);

  tensor::ExtractOp extract;
  moduleOp.walk([&](tensor::ExtractOp op) { extract = op; });
  ASSERT_TRUE(extract);
  auto powerLoop = extract->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(powerLoop);
  EXPECT_EQ(extract.getIndices().front(), powerLoop.getInductionVar());

  qc::CtrlOp controlledPower;
  powerLoop.walk([&](qc::CtrlOp op) { controlledPower = op; });
  ASSERT_TRUE(controlledPower);
  auto controlLoad =
      controlledPower.getControl(0).getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(controlLoad);
  auto controlIndex =
      controlLoad.getIndices().front().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(controlIndex);
  EXPECT_EQ(controlIndex.getRhs(), powerLoop.getInductionVar());

  qc::MeasureOp measure;
  moduleOp.walk([&](qc::MeasureOp op) { measure = op; });
  ASSERT_TRUE(measure);
  auto measurementLoop = measure->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(measurementLoop);
  auto measuredLoad = measure.getQubit().getDefiningOp<memref::LoadOp>();
  ASSERT_TRUE(measuredLoad);
  EXPECT_EQ(measuredLoad.getIndices().front(),
            measurementLoop.getInductionVar());

  ASSERT_FALSE(measure.getResult().use_empty());
  auto* user = *measure.getResult().getUsers().begin();
  auto store = dyn_cast<cbit::StoreOp>(user);
  ASSERT_TRUE(store);
  EXPECT_EQ(store.getIndex(), measurementLoop.getInductionVar());
  EXPECT_EQ(countOps<qc::SWAPOp>(moduleOp), 0U);
}

TEST(GenerateProgramTest, EmitsStandardQFTWithoutSwaps) {
  const QFT benchmark{{.qubits = 4, .periodExponent = 2}};
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  auto moduleOp = program->module();
  EXPECT_EQ(countOps<qc::SWAPOp>(moduleOp), 0U);

  qc::MeasureOp measure;
  moduleOp.walk([&](qc::MeasureOp op) { measure = op; });
  ASSERT_TRUE(measure);
  auto loop = measure->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(loop);
  auto store = dyn_cast<cbit::StoreOp>(*measure.getResult().user_begin());
  ASSERT_TRUE(store);
  auto index = store.getIndex().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(index);
  EXPECT_EQ(index.getRhs(), loop.getInductionVar());
}

TEST(GenerateProgramTest, KeepsLargeQPEFiniteAndStructured) {
  constexpr size_t precision = 1025;
  for (const auto method : {QPEMethod::Standard, QPEMethod::Iterative}) {
    SCOPED_TRACE(static_cast<int>(method));
    const QPE benchmark(
        {.precision = precision,
         .phase = Phase(std::numeric_limits<uint64_t>::max() - 1,
                        std::numeric_limits<uint64_t>::max()),
         .method = method});
    auto program = generate(benchmark);
    ASSERT_TRUE(program);
    auto moduleOp = program->module();

    auto table = angleTable(moduleOp);
    ASSERT_TRUE(table);
    EXPECT_EQ(table.getNumElements(), precision);
    for (const auto angle : table.getValues<double>()) {
      EXPECT_TRUE(std::isfinite(angle));
    }

    size_t operations = 0;
    moduleOp.walk([&](Operation*) { ++operations; });
    EXPECT_LT(operations, 150U);
    EXPECT_EQ(countOps<tensor::ExtractOp>(moduleOp), 1U);
  }
}

TEST(GenerateProgramTest, KeepsLargeQFTStructured) {
  for (const auto method : {QFTMethod::Standard, QFTMethod::Semiclassical}) {
    SCOPED_TRACE(static_cast<int>(method));
    auto program =
        generate(QFT{{.qubits = 1025, .periodExponent = 10, .method = method}});
    ASSERT_TRUE(program);
    size_t operations = 0;
    program->module().walk([&](Operation*) { ++operations; });
    EXPECT_LT(operations, 100U);
    program->module().walk([&](arith::ConstantOp op) {
      if (const auto value = dyn_cast<FloatAttr>(op.getValue())) {
        EXPECT_TRUE(std::isfinite(value.getValueAsDouble()));
      }
    });
  }
}

TEST(GenerateProgramTest, DoublesQPEPhaseModuloOneWithoutOverflow) {
  const QPE benchmark({.precision = 4,
                       .phase = Phase(uint64_t{1} << 63,
                                      std::numeric_limits<uint64_t>::max())});
  auto program = generate(benchmark);
  ASSERT_TRUE(program);
  const auto table = angleTable(program->module());
  ASSERT_TRUE(table);
  const auto angles = llvm::to_vector(table.getValues<double>());
  ASSERT_EQ(angles.size(), 4U);

  const auto denominator =
      static_cast<long double>(std::numeric_limits<uint64_t>::max());
  const auto turn = 2.L * std::numbers::pi_v<long double> / denominator;
  EXPECT_DOUBLE_EQ(angles[0], static_cast<double>((uint64_t{1} << 63) * turn));
  EXPECT_DOUBLE_EQ(angles[1], static_cast<double>(turn));
  EXPECT_DOUBLE_EQ(angles[2], static_cast<double>(2.L * turn));
  EXPECT_DOUBLE_EQ(angles[3], static_cast<double>(4.L * turn));
}

} // namespace mqt::bench
