/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/DDAdapter.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"

#include <gtest/gtest.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/Error.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

/// DD for k=2…20 plus the first HP24 width (k=33): full matrix DD through k=8
/// (MCX/MCY/MCZ) or k=6 (MCP); basis-state DD for larger Pauli widths;
/// coherent-state DD at selected policy boundaries and representative larger
/// MCP widths.
static constexpr std::array<size_t, 20> K_DD_CONTROL_COUNTS = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 33,
};
static constexpr size_t K_MATRIX_DD_MAX_PAULI = 8;
static constexpr size_t K_MATRIX_DD_MAX_MCP = 6;
/// Retain representative SP22 widths and cover the HP24 crossover, both
/// dirty-helper modes, and wide phase ladders that can fold to identity.
static constexpr std::array<size_t, 12> K_COHERENT_PAULI_CONTROL_COUNTS = {
    10, 11, 21, 22, 23, 32, 33, 34, 47, 48, 63, 64,
};
static constexpr std::array<size_t, 2> K_COHERENT_MCP_CONTROL_COUNTS = {7, 12};
/// Additional fully-lowered/CX smoke checks for k > 20 through the SP22 MCX
/// limit (k=32) and the first HP24 width (k=33). Selected widths also receive
/// coherent DD coverage above.
static constexpr std::array<size_t, 13> K_SMOKE_CONTROL_COUNTS = {
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
};

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

/// CX regression budget for the borrowed-helper rotation construction.
/// Each half-MCX occurs twice and costs 1/6/14 CX at 1/2/3 controls.
/// Above that, its two passes each use a 6-CX CCX, a 3-CX RCCX,
/// and 2(n-3) two-CX gadgets: 8n-6 CX. Both halves reach this case at k=8.
[[nodiscard]] static constexpr size_t expectedMcrCxBudget(size_t k) {
  if (k >= 8) {
    return (16 * k) - 24;
  }
  constexpr std::array<size_t, 8> small = {0, 0, 4, 14, 24, 40, 56, 80};
  return small[k];
}

namespace {
enum class ControlledPauli : uint8_t { X, Y, Z };
enum class RotationAxis : uint8_t { X, Y, Z };
} // namespace

[[nodiscard]] static dd::Controls makeControls(size_t numControls) {
  dd::Controls controls;
  for (size_t i = 0; i < numControls; ++i) {
    controls.emplace(static_cast<dd::Qubit>(i));
  }
  return controls;
}

[[nodiscard]] static dd::GateMatrix pauliMatrix(ControlledPauli pauli) {
  const auto matrix = pauli == ControlledPauli::X   ? XOp::getUnitaryMatrix()
                      : pauli == ControlledPauli::Y ? YOp::getUnitaryMatrix()
                                                    : ZOp::getUnitaryMatrix();
  return {matrix(0, 0), matrix(0, 1), matrix(1, 0), matrix(1, 1)};
}

[[nodiscard]] static dd::GateMatrix phaseMatrix(double theta) {
  const auto matrix = POp::unitaryMatrix(theta);
  return {matrix(0, 0), matrix(0, 1), matrix(1, 0), matrix(1, 1)};
}

[[nodiscard]] static dd::MatrixDD
makeControlledGateDD(dd::Package& package, size_t numControls,
                     const dd::GateMatrix& matrix) {
  return package.makeGateDD(matrix, makeControls(numControls),
                            static_cast<dd::Qubit>(numControls));
}

namespace {

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

class McPauliDdTest
    : public MultiControlledDecompositionTest,
      public testing::WithParamInterface<std::tuple<ControlledPauli, size_t>> {
};
class McpDdTest : public MultiControlledDecompositionTest,
                  public testing::WithParamInterface<size_t> {};
class McPauliSmokeTest
    : public MultiControlledDecompositionTest,
      public testing::WithParamInterface<std::tuple<ControlledPauli, size_t>> {
};
class McpSmokeTest : public MultiControlledDecompositionTest,
                     public testing::WithParamInterface<size_t> {};
class McrDdTest
    : public MultiControlledDecompositionTest,
      public testing::WithParamInterface<std::tuple<RotationAxis, size_t>> {};

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
        } else if (pauli == ControlledPauli::Y) {
          b.mcy(controls, target);
        } else {
          b.mcz(controls, target);
        }
        return SmallVector<Value>{};
      });
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

[[nodiscard]] static Value applyRotation(QCOProgramBuilder& builder,
                                         RotationAxis axis, Value theta,
                                         Value target) {
  if (axis == RotationAxis::X) {
    return builder.rx(theta, target);
  }
  if (axis == RotationAxis::Y) {
    return builder.ry(theta, target);
  }
  return builder.rz(theta, target);
}

static void buildControlledRotation(QCOProgramBuilder& builder,
                                    size_t numControls, RotationAxis axis,
                                    Value theta, bool regionLocal = false) {
  SmallVector<Value> wires;
  for (size_t i = 0; i <= numControls; ++i) {
    wires.push_back(builder.staticQubit(i));
  }
  const size_t target = numControls / 2;
  SmallVector<Value> controls;
  for (size_t i = numControls + 1; i-- > 0;) {
    if (i != target) {
      controls.push_back(wires[i]);
    }
  }
  builder.ctrl(controls, wires[target], [&](Value targetArg) {
    auto angle =
        regionLocal ? arith::NegFOp::create(builder, theta).getResult() : theta;
    return applyRotation(builder, axis, angle, targetArg);
  });
}

[[nodiscard]] static OwningOpRef<ModuleOp>
buildMcrModule(MLIRContext* context, size_t numControls, RotationAxis axis,
               double theta, bool runtimeAngle = false) {
  Value parameter;
  auto moduleOp =
      QCOProgramBuilder::build(context, [&](QCOProgramBuilder& builder) {
        parameter = builder.floatConstant(theta);
        buildControlledRotation(builder, numControls, axis, parameter);
        return SmallVector<Value>{};
      });
  if (moduleOp && runtimeAngle) {
    auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
    funcOp.insertArgument(0, Float64Type::get(context), {}, funcOp.getLoc());
    parameter.replaceAllUsesWith(funcOp.getArgument(0));
  }
  return moduleOp;
}

// R_a(theta) = cos(theta/2) I - i sin(theta/2) sigma_a.
[[nodiscard]] static dd::GateMatrix rotationMatrix(RotationAxis axis,
                                                   double theta) {
  const double cosine = std::cos(theta / 2);
  const double sine = std::sin(theta / 2);
  if (axis == RotationAxis::X) {
    return {
        cosine,
        std::complex<double>{0, -sine},
        std::complex<double>{0, -sine},
        cosine,
    };
  }
  if (axis == RotationAxis::Y) {
    return {cosine, -sine, sine, cosine};
  }
  return {
      std::complex<double>{cosine, -sine},
      0,
      0,
      std::complex<double>{cosine, sine},
  };
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

static void expectImplementsControlledRotation(
    func::FuncOp funcOp, size_t numControls, RotationAxis axis, double theta,
    const DDArgumentBindings& bindings = DDArgumentBindings()) {
  ASSERT_EQ(countStaticQubits(funcOp), numControls + 1);
  const auto package = std::make_unique<dd::Package>(numControls + 1);
  const auto actual = buildFunctionality(funcOp, *package, bindings);
  ASSERT_TRUE(succeeded(actual));
  const auto target = static_cast<dd::Qubit>(numControls / 2);
  dd::Controls controls;
  for (size_t i = 0; i <= numControls; ++i) {
    if (i != static_cast<size_t>(target)) {
      controls.emplace(static_cast<dd::Qubit>(i));
    }
  }
  const auto expected =
      package->makeGateDD(rotationMatrix(axis, theta), controls, target);
  // Full operator equality preserves phase and restores every borrowed control,
  // including when controls are entangled with other qubits.
  EXPECT_EQ(*actual, expected);
  package->decRef(*actual);
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

  const auto referenceDD =
      makeControlledGateDD(*dd, numControls, pauliMatrix(pauli));
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
}

/// Compare the complete states, including phase, without requiring identical
/// DD nodes after floating-point synthesis.
static void expectStatesNear(dd::Package& package, const dd::VectorDD& actual,
                             const dd::VectorDD& expected) {
  auto negativeExpected = expected;
  negativeExpected.w = package.cn.lookup(-dd::RealNumber::val(expected.w.r),
                                         -dd::RealNumber::val(expected.w.i));
  const auto difference = package.add(actual, negativeExpected);
  package.incRef(difference);
  constexpr double tolerance = 1e-11;
  EXPECT_LE(package.innerProduct(difference, difference).r,
            tolerance * tolerance);
  package.decRef(difference);
}

static void expectMatchesReferenceOnBasisStates(func::FuncOp funcOp,
                                                size_t numControls,
                                                ControlledPauli pauli) {
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);

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
  const auto referenceGate =
      makeControlledGateDD(*dd, numControls, pauliMatrix(pauli));
  dd->incRef(referenceGate);
  std::mt19937_64 rng(0);
  for (const auto& basisState : basisStates) {
    const auto decomposedOutput = simulate(
        funcOp, dd::makeBasisState(numQubits, basisState, *dd), *dd, rng);
    ASSERT_TRUE(succeeded(decomposedOutput));
    const auto referenceOutput = dd->applyOperation(
        referenceGate, dd::makeBasisState(numQubits, basisState, *dd));
    expectStatesNear(*dd, *decomposedOutput, referenceOutput);
    dd->decRef(*decomposedOutput);
    dd->decRef(referenceOutput);
  }
  dd->decRef(referenceGate);
}

[[nodiscard]] static dd::VectorDD
makeCoherentControlInput(size_t numControls, bool targetOne, dd::Package& dd) {
  const auto numQubits = numControls + 1;
  const auto coherentControl = numControls / 2;
  std::vector<dd::BasisStates> basisState(numQubits, dd::BasisStates::one);
  basisState[coherentControl] = dd::BasisStates::plus;
  basisState[numControls] =
      targetOne ? dd::BasisStates::one : dd::BasisStates::zero;
  return dd::makeBasisState(numQubits, basisState, dd);
}

static void
expectMatchesReferenceOnCoherentState(func::FuncOp funcOp, size_t numControls,
                                      bool targetOne,
                                      const dd::GateMatrix& referenceMatrix) {
  /// Resolve small SP22 ladder phases before comparing the whole-state error.
  const auto previousTolerance = dd::RealNumber::eps;
  const auto restoreTolerance = llvm::make_scope_exit([previousTolerance] {
    dd::ComplexNumbers::setTolerance(previousTolerance);
  });
  dd::ComplexNumbers::setTolerance(1e-15);
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);

  const auto dd = std::make_unique<dd::Package>(numQubits);
  std::mt19937_64 rng(0);
  const auto decomposedOutput = simulate(
      funcOp, makeCoherentControlInput(numControls, targetOne, *dd), *dd, rng);
  ASSERT_TRUE(succeeded(decomposedOutput));
  const auto referenceOutput = dd->applyOperation(
      makeControlledGateDD(*dd, numControls, referenceMatrix),
      makeCoherentControlInput(numControls, targetOne, *dd));

  expectStatesNear(*dd, *decomposedOutput, referenceOutput);

  dd->decRef(*decomposedOutput);
  dd->decRef(referenceOutput);
}

static void expectMatchesControlledPauliOnCoherentState(func::FuncOp funcOp,
                                                        size_t numControls,
                                                        ControlledPauli pauli) {
  expectMatchesReferenceOnCoherentState(
      funcOp, numControls, pauli == ControlledPauli::Z, pauliMatrix(pauli));
}

static void expectMatchesMcpOnCoherentState(func::FuncOp funcOp,
                                            size_t numControls, double theta) {
  expectMatchesReferenceOnCoherentState(funcOp, numControls, true,
                                        phaseMatrix(theta));
}

static void expectImplementsMcp(func::FuncOp funcOp, size_t numControls,
                                double theta) {
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);

  const auto dd = std::make_unique<dd::Package>(numQubits);
  const auto decomposedDD = buildFunctionality(funcOp, *dd);
  ASSERT_TRUE(succeeded(decomposedDD));

  const auto referenceDD =
      makeControlledGateDD(*dd, numControls, phaseMatrix(theta));
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
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
  if (failed(verify(moduleOp)) || failed(verifyLinearity(moduleOp))) {
    return failure();
  }
  PassManager pm(moduleOp.getContext());
  pm.enableVerifier();
  pm.addPass(createDecomposeMultiControlled(options));
  if (failed(pm.run(moduleOp))) {
    return failure();
  }
  // The pass manager already verifies the output IR.
  return verifyLinearity(moduleOp);
}

//===----------------------------------------------------------------------===//
// Multi-controlled rotations
//===----------------------------------------------------------------------===//

TEST_P(McrDdTest, PreservesFullOperatorAndBorrowedControls) {
  const auto [axis, numControls] = GetParam();
  for (const double theta : {0.0, 0.73, -1.21, 2 * std::numbers::pi}) {
    SCOPED_TRACE(testing::Message() << "theta=" << theta);
    auto moduleOp = buildMcrModule(context(), numControls, axis, theta);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(moduleOp.get())));
    expectFullyLowered(moduleOp.get());
    auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
    expectFullyDecomposed(funcOp);
    expectImplementsControlledRotation(funcOp, numControls, axis, theta);
  }
}

INSTANTIATE_TEST_SUITE_P(
    DdRange, McrDdTest,
    testing::Combine(testing::Values(RotationAxis::X, RotationAxis::Y,
                                     RotationAxis::Z),
                     testing::Values(2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U)),
    ([](const testing::TestParamInfo<std::tuple<RotationAxis, size_t>>& info) {
      const auto [axis, numControls] = info.param;
      return std::string(axis == RotationAxis::X   ? "Rx"
                         : axis == RotationAxis::Y ? "Ry"
                                                   : "Rz") +
             "k" + std::to_string(numControls);
    }));

TEST_F(MultiControlledDecompositionTest, RotationsPreserveRuntimeAngles) {
  constexpr size_t numControls = 8;
  for (const auto axis : {RotationAxis::X, RotationAxis::Y, RotationAxis::Z}) {
    for (const bool regionLocal : {false, true}) {
      SCOPED_TRACE(testing::Message() << "axis=" << static_cast<unsigned>(axis)
                                      << " regionLocal=" << regionLocal);
      Value parameter;
      auto moduleOp =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& builder) {
            parameter = builder.floatConstant(0.73);
            buildControlledRotation(builder, numControls, axis, parameter,
                                    regionLocal);
            return SmallVector<Value>{};
          });
      ASSERT_TRUE(moduleOp);
      auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
      funcOp.insertArgument(0, Float64Type::get(context()), {},
                            funcOp.getLoc());
      parameter.replaceAllUsesWith(funcOp.getArgument(0));
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(moduleOp.get())));
      expectFullyLowered(moduleOp.get());
      expectFullyDecomposed(funcOp);
      for (const double theta : {-0.91, 2 * std::numbers::pi}) {
        const DDArgumentBindings bindings{
            {
                funcOp.getArgument(0),
                FloatAttr::get(Float64Type::get(context()), theta),
            },
        };
        expectImplementsControlledRotation(
            funcOp, numControls, axis, regionLocal ? -theta : theta, bindings);
      }
    }
  }
}

TEST_F(MultiControlledDecompositionTest,
       RotationsUseLinearResourcesWithoutExtraQubits) {
  for (const auto axis : {RotationAxis::X, RotationAxis::Y, RotationAxis::Z}) {
    for (const size_t numControls : {
             2U,
             3U,
             4U,
             5U,
             6U,
             7U,
             8U,
             9U,
             15U,
             16U,
             17U,
             31U,
             32U,
             33U,
             63U,
             64U,
         }) {
      for (const bool runtimeAngle : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "axis=" << static_cast<unsigned>(axis) << " controls="
                     << numControls << " runtimeAngle=" << runtimeAngle);
        auto moduleOp =
            buildMcrModule(context(), numControls, axis, 0.73, runtimeAngle);
        ASSERT_TRUE(moduleOp);
        ASSERT_TRUE(succeeded(runDecomposeMultiControlled(moduleOp.get())));
        expectFullyLowered(moduleOp.get());
        auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
        expectFullyDecomposed(funcOp);
        EXPECT_EQ(countStaticQubits(funcOp), numControls + 1);
        funcOp.walk(
            [](AllocOp) { ADD_FAILURE() << "unexpected helper qubit"; });
        // The same bound covers every axis and permits future cancellation.
        EXPECT_LE(countElementaryCxOps(moduleOp.get()),
                  expectedMcrCxBudget(numControls));
        size_t elementaryGates = 0;
        funcOp.walk([&](UnitaryOpInterface op) {
          if (op->getNumRegions() == 0) {
            ++elementaryGates;
          }
        });
        EXPECT_LE(elementaryGates, 60 * numControls);
      }
    }
  }
}

TEST_F(MultiControlledDecompositionTest, RotationsRespectMinQubits) {
  for (const auto axis : {RotationAxis::X, RotationAxis::Y, RotationAxis::Z}) {
    auto moduleOp = buildMcrModule(context(), 3, axis, 0.73);
    ASSERT_TRUE(moduleOp);
    DecomposeMultiControlledOptions options;
    options.minQubits = 5;
    ASSERT_TRUE(
        succeeded(runDecomposeMultiControlled(moduleOp.get(), options)));
    EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 1U);
    auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
    expectImplementsControlledRotation(funcOp, 3, axis, 0.73);

    options.minQubits = 4;
    ASSERT_TRUE(
        succeeded(runDecomposeMultiControlled(moduleOp.get(), options)));
    EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
    expectImplementsControlledRotation(funcOp, 3, axis, 0.73);
  }
}

TEST_F(MultiControlledDecompositionTest, PreservesTargetNativeRotations) {
  using TargetOperation = CompilerTarget::Operation;
  const std::vector operations{
      llvm::cantFail(TargetOperation::create(
          "rx", TargetOperation::Arity::variadic(4), 1)),
      llvm::cantFail(TargetOperation::create(
          "ry", TargetOperation::Arity::variadic(4), 1)),
      llvm::cantFail(TargetOperation::create(
          "rz", TargetOperation::Arity::variadic(4), 1)),
  };
  const auto target = llvm::cantFail(CompilerTarget::create(
      4, CompilerTarget::Connectivity::allToAll(),
      CompilerTarget::NativeOperations::fromOperations(operations)));
  for (const auto axis : {RotationAxis::X, RotationAxis::Y, RotationAxis::Z}) {
    auto moduleOp = buildMcrModule(context(), 3, axis, 0.73);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(moduleOp.get())));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp.get())));
    PassManager pm(context());
    pm.addPass(createDecomposeMultiControlled(target));
    ASSERT_TRUE(succeeded(pm.run(moduleOp.get())));
    ASSERT_TRUE(succeeded(verify(moduleOp.get())));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp.get())));
    EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 1U);
    auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
    expectImplementsControlledRotation(funcOp, 3, axis, 0.73);
  }
}

//===----------------------------------------------------------------------===//
// MCX / MCY / MCZ / MCP: DD + CX for k = 2..20 and k = 33
//===----------------------------------------------------------------------===//

TEST_P(McPauliDdTest, EquivalenceAndCxCount) {
  const auto [pauli, k] = GetParam();
  auto moduleOp = buildControlledPauliModule(context(), k, pauli);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  // MCY and MCZ share MCX's CX budget; their basis changes add no CX.
  EXPECT_EQ(countElementaryCxOps(moduleOp.get()), K_EXPECTED_MCX_CX[k])
      << "k=" << k;

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  if (k <= K_MATRIX_DD_MAX_PAULI) {
    expectImplementsControlledPauli(funcOp, k, pauli);
  } else {
    expectMatchesReferenceOnBasisStates(funcOp, k, pauli);
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

static std::string pauliTestName(
    const testing::TestParamInfo<std::tuple<ControlledPauli, size_t>>& info) {
  const auto [pauli, k] = info.param;
  return std::string(pauli == ControlledPauli::X   ? "X"
                     : pauli == ControlledPauli::Y ? "Y"
                                                   : "Z") +
         "k" + std::to_string(k);
}

INSTANTIATE_TEST_SUITE_P(
    DdRange, McPauliDdTest,
    testing::Combine(testing::Values(ControlledPauli::X, ControlledPauli::Y,
                                     ControlledPauli::Z),
                     testing::ValuesIn(K_DD_CONTROL_COUNTS)),
    pauliTestName);
INSTANTIATE_TEST_SUITE_P(DdRange, McpDdTest,
                         testing::ValuesIn(K_DD_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<size_t>& info) {
                           return "k" + std::to_string(info.param);
                         });

TEST_F(MultiControlledDecompositionTest,
       CoherentStatesMatchAcrossSynthesisBoundaries) {
  for (const auto k : K_COHERENT_PAULI_CONTROL_COUNTS) {
    for (const auto pauli :
         {ControlledPauli::X, ControlledPauli::Y, ControlledPauli::Z}) {
      SCOPED_TRACE(testing::Message()
                   << "k=" << k << " pauli=" << static_cast<unsigned>(pauli));
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

TEST_P(McPauliSmokeTest, FullyLowersWithExpectedCx) {
  const auto [pauli, k] = GetParam();
  auto moduleOp = buildControlledPauliModule(context(), k, pauli);
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

INSTANTIATE_TEST_SUITE_P(
    SmokeRange, McPauliSmokeTest,
    testing::Combine(testing::Values(ControlledPauli::X, ControlledPauli::Y,
                                     ControlledPauli::Z),
                     testing::ValuesIn(K_SMOKE_CONTROL_COUNTS)),
    pauliTestName);
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
        builder.crx(0.73, builder.staticQubit(4), builder.staticQubit(5));
        builder.cry(0.73, builder.staticQubit(6), builder.staticQubit(7));
        builder.crz(0.73, builder.staticQubit(8), builder.staticQubit(9));
        builder.cy(builder.staticQubit(10), builder.staticQubit(11));
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
  EXPECT_EQ(singleControlled, 6U);
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

TEST_F(MultiControlledDecompositionTest,
       PreservesTargetNativeControlledRCCXBody) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        std::ignore =
            builder.crccx(builder.staticQubit(0), builder.staticQubit(1),
                          builder.staticQubit(2), builder.staticQubit(3));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);

  using Operation = CompilerTarget::Operation;
  std::vector operations{
      llvm::cantFail(
          Operation::create("rccx", Operation::Arity::variadic(4), 0)),
  };
  const auto target = llvm::cantFail(CompilerTarget::create(
      4, CompilerTarget::Connectivity::allToAll(),
      CompilerTarget::NativeOperations::fromOperations(operations)));

  CtrlOp shell;
  moduleOp->walk([&](CtrlOp op) { shell = op; });
  ASSERT_TRUE(shell);
  ASSERT_EQ(shell.getNumBodyUnitaries(), 1U);
  ASSERT_TRUE(target.supports(shell.getOperation()));
  ASSERT_FALSE(target.supports(shell.getBodyUnitary(0).getOperation()));

  PassManager pm(context());
  pm.addPass(createDecomposeMultiControlled(target));
  ASSERT_TRUE(pm.run(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 1), 1U);
  EXPECT_EQ(countRCCXOps(moduleOp.get()), 1U);
}

TEST_F(MultiControlledDecompositionTest, MinQubitsThreshold) {
  for (const auto pauli :
       {ControlledPauli::X, ControlledPauli::Y, ControlledPauli::Z}) {
    SCOPED_TRACE(testing::Message()
                 << "pauli=" << static_cast<unsigned>(pauli));
    auto moduleOp = buildControlledPauliModule(context(), 2, pauli);
    ASSERT_TRUE(moduleOp);
    DecomposeMultiControlledOptions options;
    options.minQubits = 4;
    ASSERT_TRUE(
        runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
    EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 1U);

    options.minQubits = 3;
    ASSERT_TRUE(
        runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
    auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
    expectImplementsControlledPauli(funcOp, 2, pauli);

    options.minQubits = 2;
    EXPECT_FALSE(
        runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
  }
}

TEST_F(MultiControlledDecompositionTest, PreservesTargetNativeMcy) {
  auto moduleOp = buildControlledPauliModule(context(), 3, ControlledPauli::Y);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(moduleOp.get())));
  ASSERT_TRUE(succeeded(verifyLinearity(moduleOp.get())));
  using TargetOperation = CompilerTarget::Operation;
  const std::vector operations{
      llvm::cantFail(
          TargetOperation::create("y", TargetOperation::Arity::variadic(4), 0)),
  };
  const auto target = llvm::cantFail(CompilerTarget::create(
      4, CompilerTarget::Connectivity::allToAll(),
      CompilerTarget::NativeOperations::fromOperations(operations)));
  PassManager pm(context());
  pm.addPass(createDecomposeMultiControlled(target));
  ASSERT_TRUE(succeeded(pm.run(moduleOp.get())));
  EXPECT_TRUE(succeeded(verify(moduleOp.get())));
  EXPECT_TRUE(succeeded(verifyLinearity(moduleOp.get())));
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 1U);
  moduleOp->walk([](CtrlOp op) {
    ASSERT_EQ(op.getNumBodyUnitaries(), 1U);
    EXPECT_TRUE(isa<YOp>(op.getBodyUnitary(0).getOperation()));
  });
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

  const auto referenceDD = makeGateDD(
      *dd, DynamicMatrix{SWAPOp::getUnitaryMatrix()}, numQubits, {1, 2}, {{0}});
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
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

  const auto referenceDD =
      makeGateDD(*dd, DynamicMatrix{SWAPOp::getUnitaryMatrix()}, numQubits,
                 {2, 3}, {{0}, {1}});
  EXPECT_EQ(*decomposedDD, referenceDD);
  dd->decRef(*decomposedDD);
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
