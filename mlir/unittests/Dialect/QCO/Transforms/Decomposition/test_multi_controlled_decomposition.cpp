/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

/// DD for k=2…20 plus the first HP24 width (k=33): full matrix DD through k=8
/// (MCX/MCZ) or k=6 (MCP); basis-state DD for larger MCX/MCZ widths;
/// coherent-state DD at selected policy boundaries and representative larger
/// MCP widths.
static constexpr std::array<size_t, 20> K_DD_CONTROL_COUNTS = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 33};
static constexpr size_t K_MATRIX_DD_MAX_PAULI = 8;
static constexpr size_t K_MATRIX_DD_MAX_MCP = 6;
static constexpr std::array<size_t, 5> K_COHERENT_HP24_CONTROL_COUNTS = {
    10, 11, 21, 22, 23};
static constexpr std::array<size_t, 2> K_COHERENT_MCP_CONTROL_COUNTS = {7, 12};
/// Additional fully-lowered/CX smoke checks for k > 20 through the SP22 MCX
/// limit (k=32) and the first HP24 width (k=33). Selected widths also receive
/// coherent DD coverage above.
static constexpr std::array<size_t, 13> K_SMOKE_CONTROL_COUNTS = {
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33};

/// Expected elementary Ctrl@X counts after default `min-qubits=3` lowering.
/// Indexed by control count `k`; unused slots are zero.
/// For `5 ≤ k ≤ 32`, MCX uses SP22 MCP(π); each CRX expands to 2 Ctrl@X while
/// CP stays as Ctrl@P, so elementary CX is `4k² − 8k + 4`. k=33 is the first
/// HP24 width (pinned measured CX).
static constexpr std::array<size_t, 34> K_EXPECTED_MCX_CX = {
    0,    0,
    6,    // 2
    14,   // 3
    20,   // 4
    64,   // 5  SP22
    100,  // 6
    144,  // 7
    196,  // 8
    256,  // 9
    324,  // 10
    400,  // 11
    484,  // 12
    576,  // 13
    676,  // 14
    784,  // 15
    900,  // 16
    1024, // 17
    1156, // 18
    1296, // 19
    1444, // 20
    1600, // 21
    1764, // 22
    1936, // 23
    2116, // 24
    2304, // 25
    2500, // 26
    2704, // 27
    2916, // 28
    3136, // 29
    3364, // 30
    3600, // 31
    3844, // 32  SP22 max
    3872, // 33  HP24
};

/// Effective CX for MCP: elementary Ctrl@X plus ~2 CX per leftover
/// single-controlled P. For `k >= 5` this matches the SP22 LDD bound
/// `4k^2 - 4k + 2`.
[[nodiscard]] static constexpr size_t expectedMcpCx(size_t k) {
  if (k >= 5) {
    return (4 * k * k) - (4 * k) + 2;
  }
  constexpr std::array<size_t, 5> small = {0, 0, 6, 20, 42};
  return small[k];
}

namespace {

enum class ControlledPauli : uint8_t { X, Z };

class MultiControlledDecompositionTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context_ = std::make_unique<MLIRContext>();
    context_->appendDialectRegistry(registry);
    context_->loadAllAvailableDialects();
  }

public:
  [[nodiscard]] MLIRContext* context() const { return context_.get(); }

private:
  std::unique_ptr<MLIRContext> context_;
};

class McxDdTest : public MultiControlledDecompositionTest,
                  public testing::WithParamInterface<size_t> {};
class MczDdTest : public MultiControlledDecompositionTest,
                  public testing::WithParamInterface<size_t> {};
class McpDdTest : public MultiControlledDecompositionTest,
                  public testing::WithParamInterface<size_t> {};
class McxSmokeTest : public MultiControlledDecompositionTest,
                     public testing::WithParamInterface<size_t> {};
class MczSmokeTest : public MultiControlledDecompositionTest,
                     public testing::WithParamInterface<size_t> {};
class McpSmokeTest : public MultiControlledDecompositionTest,
                     public testing::WithParamInterface<size_t> {};

} // namespace

[[nodiscard]] static OwningOpRef<ModuleOp>
buildControlledPauliModule(MLIRContext* context, size_t numControls,
                           ControlledPauli pauli) {
  return QCOProgramBuilder::build(
      context, [numControls, pauli](QCOProgramBuilder& b) {
        SmallVector<Value> wires;
        wires.reserve(numControls + 1);
        for (size_t i = 0; i <= numControls; ++i) {
          wires.push_back(b.staticQubit(i));
        }
        auto controls = ValueRange(wires).drop_back();
        auto target = wires.back();
        if (pauli == ControlledPauli::X) {
          b.mcx(controls, target);
        } else {
          b.mcz(controls, target);
        }
        return SmallVector<Value>{};
      });
}

[[nodiscard]] static OwningOpRef<ModuleOp> buildMcxModule(MLIRContext* context,
                                                          size_t numControls) {
  return buildControlledPauliModule(context, numControls, ControlledPauli::X);
}

[[nodiscard]] static OwningOpRef<ModuleOp> buildMczModule(MLIRContext* context,
                                                          size_t numControls) {
  return buildControlledPauliModule(context, numControls, ControlledPauli::Z);
}

[[nodiscard]] static OwningOpRef<ModuleOp>
buildMcpModule(MLIRContext* context, size_t numControls, double theta) {
  return QCOProgramBuilder::build(
      context, [numControls, theta](QCOProgramBuilder& b) {
        SmallVector<Value> wires;
        wires.reserve(numControls + 1);
        for (size_t i = 0; i <= numControls; ++i) {
          wires.push_back(b.staticQubit(i));
        }
        b.mcp(theta, ValueRange(wires).drop_back(), wires.back());
        return SmallVector<Value>{};
      });
}

[[nodiscard]] static OwningOpRef<ModuleOp>
buildRCCXModule(MLIRContext* context) {
  return QCOProgramBuilder::build(context, [](QCOProgramBuilder& b) {
    std::ignore = b.rccx(b.staticQubit(0), b.staticQubit(1), b.staticQubit(2));
    return SmallVector<Value>{};
  });
}

[[nodiscard]] static size_t countStaticQubits(func::FuncOp funcOp) {
  size_t numQubits = 0;
  for (StaticOp staticOp : funcOp.getOps<StaticOp>()) {
    numQubits =
        std::max(numQubits, static_cast<size_t>(staticOp.getIndex()) + 1);
  }
  return numQubits;
}

static void expectFullyDecomposed(func::FuncOp funcOp) {
  funcOp.walk([](CtrlOp op) {
    EXPECT_EQ(op.getNumControls(), 1U);
    EXPECT_EQ(op.getNumTargets(), 1U);
  });
  funcOp.walk([](RCCXOp) { ADD_FAILURE() << "unexpected leftover rccx"; });
}

static void expectImplementsControlledPauli(func::FuncOp funcOp,
                                            size_t numControls,
                                            ControlledPauli pauli) {
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);

  const auto dd = std::make_unique<dd::Package>(numQubits);
  const auto decomposedDD = buildFunctionality(funcOp, *dd);
  ASSERT_TRUE(succeeded(decomposedDD));

  qc::QuantumComputation referenceQc(numQubits);
  qc::Controls controls;
  for (size_t i = 0; i < numControls; ++i) {
    controls.emplace(static_cast<qc::Qubit>(i));
  }
  const auto target = static_cast<qc::Qubit>(numControls);
  if (pauli == ControlledPauli::X) {
    referenceQc.mcx(controls, target);
  } else {
    referenceQc.mcz(controls, target);
  }

  const dd::MatrixDD referenceDD = dd::buildFunctionality(referenceQc, *dd);
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
  dd->decRef(referenceDD);
}

static void expectMatchesReferenceOnBasisStates(func::FuncOp funcOp,
                                                size_t numControls,
                                                ControlledPauli pauli) {
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);

  qc::QuantumComputation referenceQc(numQubits);
  qc::Controls controls;
  for (size_t i = 0; i < numControls; ++i) {
    controls.emplace(static_cast<qc::Qubit>(i));
  }
  const auto target = static_cast<qc::Qubit>(numControls);
  if (pauli == ControlledPauli::X) {
    referenceQc.mcx(controls, target);
  } else {
    referenceQc.mcz(controls, target);
  }

  std::vector<std::vector<bool>> basisStates;
  basisStates.emplace_back(numQubits, true);
  basisStates.back()[numControls] = false;
  basisStates.emplace_back(numQubits, true);
  for (const size_t inactiveControl :
       std::array<size_t, 3>{0U, numControls / 2U, numControls - 1U}) {
    basisStates.emplace_back(numQubits, true);
    basisStates.back()[inactiveControl] = false;
    basisStates.back()[numControls] = false;
  }

  const auto dd = std::make_unique<dd::Package>(numQubits);
  for (const auto& basisState : basisStates) {
    const auto decomposedOutput =
        simulate(funcOp, dd::makeBasisState(numQubits, basisState, *dd), *dd);
    ASSERT_TRUE(succeeded(decomposedOutput));
    dd->incRef(*decomposedOutput);
    const auto referenceOutput = dd::simulate(
        referenceQc, dd::makeBasisState(numQubits, basisState, *dd), *dd);
    EXPECT_EQ(decomposedOutput->p, referenceOutput.p);
    EXPECT_NEAR(dd::RealNumber::val(decomposedOutput->w.r),
                dd::RealNumber::val(referenceOutput.w.r), 1e-11);
    EXPECT_NEAR(dd::RealNumber::val(decomposedOutput->w.i),
                dd::RealNumber::val(referenceOutput.w.i), 1e-11);
    dd->decRef(*decomposedOutput);
    dd->decRef(referenceOutput);
  }
}

[[nodiscard]] static dd::VectorDD
makeCoherentControlInput(size_t numControls, bool targetOne, dd::Package& dd) {
  const auto numQubits = numControls + 1;
  const auto coherentControl = numControls / 2;
  qc::QuantumComputation preparationQc(numQubits);
  for (size_t control = 0; control < numControls; ++control) {
    if (control == coherentControl) {
      preparationQc.h(static_cast<qc::Qubit>(control));
    } else {
      preparationQc.x(static_cast<qc::Qubit>(control));
    }
  }
  if (targetOne) {
    preparationQc.x(static_cast<qc::Qubit>(numControls));
  }
  return dd::simulate(preparationQc, dd::makeZeroState(numQubits, dd), dd);
}

static void
expectMatchesReferenceOnCoherentState(func::FuncOp funcOp,
                                      const qc::QuantumComputation& referenceQc,
                                      size_t numControls, bool targetOne) {
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);

  const auto dd = std::make_unique<dd::Package>(numQubits);
  const auto decomposedOutput = simulate(
      funcOp, makeCoherentControlInput(numControls, targetOne, *dd), *dd);
  ASSERT_TRUE(succeeded(decomposedOutput));
  dd->incRef(*decomposedOutput);
  const auto referenceOutput = dd::simulate(
      referenceQc, makeCoherentControlInput(numControls, targetOne, *dd), *dd);

  EXPECT_EQ(decomposedOutput->p, referenceOutput.p);
  EXPECT_NEAR(dd::RealNumber::val(decomposedOutput->w.r),
              dd::RealNumber::val(referenceOutput.w.r), 1e-11);
  EXPECT_NEAR(dd::RealNumber::val(decomposedOutput->w.i),
              dd::RealNumber::val(referenceOutput.w.i), 1e-11);

  dd->decRef(*decomposedOutput);
  dd->decRef(referenceOutput);
}

static void expectMatchesControlledPauliOnCoherentState(func::FuncOp funcOp,
                                                        size_t numControls,
                                                        ControlledPauli pauli) {
  const auto numQubits = numControls + 1;
  qc::QuantumComputation referenceQc(numQubits);
  qc::Controls controls;
  for (size_t i = 0; i < numControls; ++i) {
    controls.emplace(static_cast<qc::Qubit>(i));
  }
  const auto target = static_cast<qc::Qubit>(numControls);
  if (pauli == ControlledPauli::X) {
    referenceQc.mcx(controls, target);
  } else {
    referenceQc.mcz(controls, target);
  }
  expectMatchesReferenceOnCoherentState(funcOp, referenceQc, numControls,
                                        pauli == ControlledPauli::Z);
}

static void expectMatchesMcpOnCoherentState(func::FuncOp funcOp,
                                            size_t numControls, double theta) {
  qc::QuantumComputation referenceQc(numControls + 1);
  qc::Controls controls;
  for (size_t i = 0; i < numControls; ++i) {
    controls.emplace(static_cast<qc::Qubit>(i));
  }
  referenceQc.mcp(theta, controls, static_cast<qc::Qubit>(numControls));
  expectMatchesReferenceOnCoherentState(funcOp, referenceQc, numControls, true);
}

static void expectImplementsMcp(func::FuncOp funcOp, size_t numControls,
                                double theta) {
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);

  const auto dd = std::make_unique<dd::Package>(numQubits);
  const auto decomposedDD = buildFunctionality(funcOp, *dd);
  ASSERT_TRUE(succeeded(decomposedDD));

  qc::QuantumComputation referenceQc(numQubits);
  qc::Controls controls;
  for (size_t i = 0; i < numControls; ++i) {
    controls.emplace(static_cast<qc::Qubit>(i));
  }
  referenceQc.mcp(theta, controls, static_cast<qc::Qubit>(numControls));

  const dd::MatrixDD referenceDD = dd::buildFunctionality(referenceQc, *dd);
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
  dd->decRef(referenceDD);
}

/// Count `CtrlOp`s whose control operand count is at least @p minControlCount.
[[nodiscard]] static size_t
countMultiControlledOps(ModuleOp moduleOp, size_t minControlCount = 2) {
  size_t count = 0;
  moduleOp.walk([&count, minControlCount](CtrlOp op) {
    if (op.getNumControls() >= minControlCount) {
      ++count;
    }
  });
  return count;
}

[[nodiscard]] static size_t countRCCXOps(ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&count](RCCXOp) { ++count; });
  return count;
}

[[nodiscard]] static bool ctrlBodyIsSingleX(CtrlOp op) {
  if (op.getNumControls() != 1) {
    return false;
  }
  size_t xCount = 0;
  op.getBody()->walk([&](XOp) { ++xCount; });
  return xCount == 1;
}

[[nodiscard]] static size_t countElementaryCxOps(ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&count](CtrlOp op) {
    if (ctrlBodyIsSingleX(op)) {
      ++count;
    }
  });
  return count;
}

[[nodiscard]] static bool ctrlBodyIsSingleP(CtrlOp op) {
  if (op.getNumControls() != 1) {
    return false;
  }
  size_t pCount = 0;
  size_t xCount = 0;
  op.getBody()->walk([&](POp) { ++pCount; });
  op.getBody()->walk([&](XOp) { ++xCount; });
  return xCount == 0 && pCount == 1;
}

[[nodiscard]] static size_t countEffectiveCxOps(ModuleOp moduleOp) {
  size_t singleP = 0;
  moduleOp.walk([&singleP](CtrlOp op) {
    if (ctrlBodyIsSingleP(op)) {
      ++singleP;
    }
  });
  return countElementaryCxOps(moduleOp) + (2 * singleP);
}

static void expectFullyLowered(ModuleOp moduleOp) {
  EXPECT_EQ(countMultiControlledOps(moduleOp, 2), 0U);
  EXPECT_EQ(countRCCXOps(moduleOp), 0U);
}

static LogicalResult runDecomposeMultiControlled(
    ModuleOp moduleOp, const DecomposeMultiControlledOptions& options = {}) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createDecomposeMultiControlled(options));
  return pm.run(moduleOp);
}

//===----------------------------------------------------------------------===//
// MCX / MCZ / MCP: DD + CX for k = 2..20
//===----------------------------------------------------------------------===//

TEST_P(McxDdTest, EquivalenceAndCxCount) {
  const size_t k = GetParam();
  auto moduleOp = buildMcxModule(context(), k);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  EXPECT_EQ(countElementaryCxOps(moduleOp.get()), K_EXPECTED_MCX_CX[k])
      << "k=" << k;

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  if (k <= K_MATRIX_DD_MAX_PAULI) {
    expectImplementsControlledPauli(funcOp, k, ControlledPauli::X);
  } else {
    expectMatchesReferenceOnBasisStates(funcOp, k, ControlledPauli::X);
  }
}

TEST_P(MczDdTest, EquivalenceAndCxCount) {
  const size_t k = GetParam();
  auto moduleOp = buildMczModule(context(), k);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  // MCZ shares the MCX elementary sequences / cores (no extra CX from the
  // outer H sandwich on X).
  EXPECT_EQ(countElementaryCxOps(moduleOp.get()), K_EXPECTED_MCX_CX[k])
      << "k=" << k;

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  if (k <= K_MATRIX_DD_MAX_PAULI) {
    expectImplementsControlledPauli(funcOp, k, ControlledPauli::Z);
  } else {
    expectMatchesReferenceOnBasisStates(funcOp, k, ControlledPauli::Z);
  }
}

TEST_P(McpDdTest, EquivalenceAndCxCount) {
  const size_t k = GetParam();
  constexpr double theta = 0.7;
  auto moduleOp = buildMcpModule(context(), k, theta);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  EXPECT_EQ(countEffectiveCxOps(moduleOp.get()), expectedMcpCx(k)) << "k=" << k;

  if (k <= K_MATRIX_DD_MAX_MCP) {
    auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
    expectImplementsMcp(funcOp, k, theta);
  }
}

INSTANTIATE_TEST_SUITE_P(DdRange, McxDdTest,
                         testing::ValuesIn(K_DD_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<size_t>& info) {
                           return "k" + std::to_string(info.param);
                         });
INSTANTIATE_TEST_SUITE_P(DdRange, MczDdTest,
                         testing::ValuesIn(K_DD_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<size_t>& info) {
                           return "k" + std::to_string(info.param);
                         });
INSTANTIATE_TEST_SUITE_P(DdRange, McpDdTest,
                         testing::ValuesIn(K_DD_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<size_t>& info) {
                           return "k" + std::to_string(info.param);
                         });

TEST_F(MultiControlledDecompositionTest,
       CoherentStatesMatchAcrossHp24PolicyBoundaries) {
  for (const auto k : K_COHERENT_HP24_CONTROL_COUNTS) {
    for (const auto pauli : {ControlledPauli::X, ControlledPauli::Z}) {
      SCOPED_TRACE(testing::Message()
                   << "k=" << k
                   << " pauli=" << (pauli == ControlledPauli::X ? "X" : "Z"));
      auto moduleOp = buildControlledPauliModule(context(), k, pauli);
      ASSERT_TRUE(moduleOp);
      ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
      auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
      expectMatchesControlledPauliOnCoherentState(funcOp, k, pauli);
    }
  }
}

TEST_F(MultiControlledDecompositionTest, CoherentStatesMatchForLargerSp22Mcp) {
  constexpr double theta = 0.7;
  for (const auto k : K_COHERENT_MCP_CONTROL_COUNTS) {
    SCOPED_TRACE(testing::Message() << "k=" << k);
    auto moduleOp = buildMcpModule(context(), k, theta);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
    auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
    expectMatchesMcpOnCoherentState(funcOp, k, theta);
  }
}

//===----------------------------------------------------------------------===//
// Additional smoke checks for k > 20 — fully lowered, pinned CX
//===----------------------------------------------------------------------===//

TEST_P(McxSmokeTest, FullyLowersWithExpectedCx) {
  const size_t k = GetParam();
  auto moduleOp = buildMcxModule(context(), k);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  EXPECT_EQ(countElementaryCxOps(moduleOp.get()), K_EXPECTED_MCX_CX[k])
      << "k=" << k;
}

TEST_P(MczSmokeTest, FullyLowersWithExpectedCx) {
  const size_t k = GetParam();
  auto moduleOp = buildMczModule(context(), k);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  EXPECT_EQ(countElementaryCxOps(moduleOp.get()), K_EXPECTED_MCX_CX[k])
      << "k=" << k;
}

TEST_P(McpSmokeTest, FullyLowersWithExpectedCx) {
  const size_t k = GetParam();
  constexpr double theta = std::numbers::pi / 3.0;
  auto moduleOp = buildMcpModule(context(), k, theta);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  EXPECT_EQ(countEffectiveCxOps(moduleOp.get()), expectedMcpCx(k)) << "k=" << k;
}

INSTANTIATE_TEST_SUITE_P(SmokeRange, McxSmokeTest,
                         testing::ValuesIn(K_SMOKE_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<size_t>& info) {
                           return "k" + std::to_string(info.param);
                         });
INSTANTIATE_TEST_SUITE_P(SmokeRange, MczSmokeTest,
                         testing::ValuesIn(K_SMOKE_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<size_t>& info) {
                           return "k" + std::to_string(info.param);
                         });
INSTANTIATE_TEST_SUITE_P(SmokeRange, McpSmokeTest,
                         testing::ValuesIn(K_SMOKE_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<size_t>& info) {
                           return "k" + std::to_string(info.param);
                         });

//===----------------------------------------------------------------------===//
// Pass behavior
//===----------------------------------------------------------------------===//

TEST_F(MultiControlledDecompositionTest, LeavesSingleControlledUntouched) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.cx(builder.staticQubit(0), builder.staticQubit(1));
        builder.cz(builder.staticQubit(2), builder.staticQubit(3));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 0U);
  size_t singleControlled = 0;
  moduleOp->walk([&](CtrlOp op) {
    if (op.getNumControls() == 1) {
      ++singleControlled;
    }
  });
  EXPECT_EQ(singleControlled, 2U);
}

TEST_F(MultiControlledDecompositionTest, DecomposesRCCX) {
  auto moduleOp = buildRCCXModule(context());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  EXPECT_EQ(countRCCXOps(moduleOp.get()), 0U);
}

TEST_F(MultiControlledDecompositionTest, LeavesRCCXWhenMinQubitsIsFour) {
  auto moduleOp = buildRCCXModule(context());
  ASSERT_TRUE(moduleOp);
  DecomposeMultiControlledOptions options;
  options.minQubits = 4;
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
  EXPECT_EQ(countRCCXOps(moduleOp.get()), 1U);
}

TEST_F(MultiControlledDecompositionTest, MinQubitsThreshold) {
  auto moduleOp = buildMcxModule(context(), 2);
  ASSERT_TRUE(moduleOp);
  DecomposeMultiControlledOptions options;
  options.minQubits = 4;
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 1U);

  options.minQubits = 2;
  EXPECT_FALSE(
      runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
}

TEST_F(MultiControlledDecompositionTest, DecomposesSingleControlledSwap) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        std::ignore =
            builder.cswap(builder.staticQubit(0), builder.staticQubit(1),
                          builder.staticQubit(2));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectFullyDecomposed(funcOp);

  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, 3U);
  const auto dd = std::make_unique<dd::Package>(numQubits);
  const auto decomposedDD = buildFunctionality(funcOp, *dd);
  ASSERT_TRUE(succeeded(decomposedDD));

  qc::QuantumComputation referenceQc(numQubits);
  referenceQc.cswap(0, 1, 2);
  const dd::MatrixDD referenceDD = dd::buildFunctionality(referenceQc, *dd);
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
  dd->decRef(referenceDD);
}

TEST_F(MultiControlledDecompositionTest, DecomposesMultipleControlledSwap) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        std::ignore =
            builder.mcswap({builder.staticQubit(0), builder.staticQubit(1)},
                           builder.staticQubit(2), builder.staticQubit(3));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectFullyDecomposed(funcOp);

  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, 4U);
  const auto dd = std::make_unique<dd::Package>(numQubits);
  const auto decomposedDD = buildFunctionality(funcOp, *dd);
  ASSERT_TRUE(succeeded(decomposedDD));

  qc::QuantumComputation referenceQc(numQubits);
  referenceQc.mcswap({0, 1}, 2, 3);
  const dd::MatrixDD referenceDD = dd::buildFunctionality(referenceQc, *dd);
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
  dd->decRef(referenceDD);
}

TEST_F(MultiControlledDecompositionTest,
       LeavesSingleControlledSwapWhenMinQubitsIsFour) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        std::ignore =
            builder.cswap(builder.staticQubit(0), builder.staticQubit(1),
                          builder.staticQubit(2));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  DecomposeMultiControlledOptions options;
  options.minQubits = 4;
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get(), options).succeeded());

  size_t controlledSwap = 0;
  moduleOp->walk([&](CtrlOp op) {
    if (op.getNumControls() == 1 && op.getNumTargets() == 2) {
      auto inner =
          mlir::mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
      if (inner && isa<SWAPOp>(inner.getOperation())) {
        ++controlledSwap;
      }
    }
  });
  EXPECT_EQ(controlledSwap, 1U);
}

TEST_F(MultiControlledDecompositionTest,
       DecomposesTwoControlledSwapWhenMinQubitsIsFour) {
  // Two-control SWAP acts on 4 qubits, so min-qubits=4 still rewrites it.
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        std::ignore =
            builder.mcswap({builder.staticQubit(0), builder.staticQubit(1)},
                           builder.staticQubit(2), builder.staticQubit(3));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  DecomposeMultiControlledOptions options;
  options.minQubits = 4;
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
  expectFullyLowered(moduleOp.get());

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectFullyDecomposed(funcOp);
}

TEST_F(MultiControlledDecompositionTest, LeavesUnsupportedCtrlUntouched) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.mch({builder.staticQubit(0), builder.staticQubit(1)},
                    builder.staticQubit(2));
        builder.ctrl({builder.staticQubit(3), builder.staticQubit(4)},
                     builder.staticQubit(5), [&](Value targetArg) -> Value {
                       return builder.y(builder.x(targetArg));
                     });
        // Two-target non-SWAP body: passes min-qubits but is not lowered.
        std::ignore =
            builder.cdcx(builder.staticQubit(6), builder.staticQubit(7),
                         builder.staticQubit(8));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 2U);

  size_t multiOpCtrl = 0;
  size_t mchCount = 0;
  size_t controlledDcx = 0;
  moduleOp->walk([&](CtrlOp op) {
    if (op.getNumTargets() == 2) {
      auto inner =
          mlir::mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
      if (inner && isa<DCXOp>(inner.getOperation())) {
        ++controlledDcx;
      }
    }
    if (op.getNumControls() < 2) {
      return;
    }
    if (op.getNumBodyUnitaries() == 2) {
      ++multiOpCtrl;
    }
    if (op.getNumBodyUnitaries() == 1 &&
        isa<HOp>(op.getBodyUnitary(0).getOperation())) {
      ++mchCount;
    }
  });
  EXPECT_EQ(multiOpCtrl, 1U);
  EXPECT_EQ(mchCount, 1U);
  EXPECT_EQ(controlledDcx, 1U);
}

TEST_F(MultiControlledDecompositionTest, PhasePiRoutesThroughMcz) {
  for (const double theta : {std::numbers::pi, -std::numbers::pi}) {
    for (const size_t k : {2U, 3U, 4U, 5U}) {
      auto moduleOp = buildMcpModule(context(), k, theta);
      ASSERT_TRUE(moduleOp) << "k=" << k << " theta=" << theta;
      ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded())
          << "k=" << k << " theta=" << theta;
      expectFullyLowered(moduleOp.get());
      // ±π must take the Z path, so CX counts match MCZ/MCX.
      EXPECT_EQ(countElementaryCxOps(moduleOp.get()), K_EXPECTED_MCX_CX[k])
          << "k=" << k << " theta=" << theta;
      auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
      expectImplementsMcp(funcOp, k, theta);
    }
  }
}
