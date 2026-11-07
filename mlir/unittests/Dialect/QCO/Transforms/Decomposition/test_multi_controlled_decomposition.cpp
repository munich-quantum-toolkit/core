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
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/DDAdapter.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

/// DD coverage for k=2…20 and a representative HP24 width (k=33).
///
/// Full matrix DD through k=9 (MCX/MCY/MCZ) or k=6 (MCP); basis-state DD for
/// larger Pauli widths; coherent-state DD at SP22/HP24 boundaries and
/// representative larger MCP widths.
static constexpr std::array<size_t, 20> K_DD_CONTROL_COUNTS = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 33,
};
static constexpr size_t K_MATRIX_DD_MAX_PAULI = 9;
static constexpr size_t K_MATRIX_DD_MAX_MCP = 6;
/// Retain representative SP22 widths and cover the HP24 crossover, both
/// dirty-ancilla modes, and wide phase ladders that can fold to identity.
static constexpr std::array<size_t, 13> K_COHERENT_PAULI_CONTROL_COUNTS = {
    10, 11, 21, 22, 23, 24, 32, 33, 34, 47, 48, 63, 64,
};
static constexpr std::array<size_t, 2> K_COHERENT_MCP_CONTROL_COUNTS = {7, 12};
static constexpr std::array K_ANCILLA_ANGLES = {
    0.0,
    0.73,
    -1.21,
    std::numbers::pi,
    -std::numbers::pi,
    2 * std::numbers::pi,
    -2 * std::numbers::pi,
};
/// Additional fully-lowered/CX checks across the SP22/HP24 crossover.
///
/// Selected widths also receive coherent DD coverage above.
static constexpr std::array<size_t, 13> K_SMOKE_CONTROL_COUNTS = {
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
};

/// Expected elementary Ctrl@X counts after default `min-qubits=3` lowering.
///
/// At k=4…8, phase-corrected RZ lowers entirely to CX and single-qubit gates.
/// For `9 ≤ k ≤ 22`, SP22 MCP(π) leaves CP gates and emits 4(k−1)² CX.
/// Relative-phase HP24 wins from k=23: odd k leaves CP gates; even k lowers
/// its two-controlled phases completely to CX and single-qubit gates.
[[nodiscard]] static constexpr size_t expectedMcxCx(size_t k) {
  if (k >= 23) {
    return k % 2 == 1 ? 84 * k - 52 : 96 * k - 154;
  }
  if (k >= 9) {
    return 4 * (k - 1) * (k - 1);
  }
  if (k >= 4) {
    return 6 * k * k - 26 * k + 40;
  }
  constexpr std::array<size_t, 4> small = {0, 0, 6, 14};
  return small[k];
}

/// Effective CX for MCP: elementary Ctrl@X plus ~2 CX per leftover
/// single-controlled P.
///
/// Use recursive RZ phase correction through k=8, then the cheaper SP22 LDD.
[[nodiscard]] static constexpr size_t expectedMcpCx(size_t k) {
  if (k >= 9) {
    return (4 * k * k) - (4 * k) + 2;
  }
  return k == 2 ? 6 : (6 * k * k) - (26 * k) + 40;
}

/// CX regression budget for relative-phase halves with no external ancillas.
///
/// Each half occurs twice and costs 1 CX at one control, else 6n−8 CX.
[[nodiscard]] static constexpr size_t expectedMcrCxBudget(size_t k) {
  return k == 2 ? 4 : k == 3 ? 10 : 12 * k - 32;
}

namespace {
enum class ControlledPauli : uint8_t { X, Y, Z };
enum class RotationAxis : uint8_t { X, Y, Z };
enum class AncillaFamily : uint8_t {
  X,
  Y,
  Z,
  Swap,
  Phase,
  PhasePi,
  RX,
  RY,
  RZ,
};
} // namespace

static const char* ancillaFamilyName(AncillaFamily family) {
  constexpr std::array names = {
      "X", "Y", "Z", "Swap", "Phase", "PhasePi", "RX", "RY", "RZ",
  };
  return names[static_cast<size_t>(family)];
}

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
                     const dd::GateMatrix& matrix, size_t target) {
  dd::Controls controls;
  for (size_t i = 0; i <= numControls; ++i) {
    if (i != target) {
      controls.emplace(static_cast<dd::Qubit>(i));
    }
  }
  return llvm::cantFail(
      package.makeGateDD(matrix, controls, static_cast<dd::Qubit>(target)));
}

namespace {

class MultiControlledDecompositionTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, qtensor::QTensorDialect, arith::ArithDialect,
                    func::FuncDialect>();
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
class AncillaFamilyTest : public MultiControlledDecompositionTest,
                          public testing::WithParamInterface<AncillaFamily> {};
class AncillaAngleTest : public MultiControlledDecompositionTest,
                         public testing::WithParamInterface<AncillaFamily> {};

} // namespace

[[nodiscard]] static OwningOpRef<ModuleOp>
buildControlledPauliModule(MLIRContext* context, size_t numControls,
                           ControlledPauli pauli, bool reverseWires = false) {
  return QCOProgramBuilder::build(
      context, [numControls, pauli, reverseWires](QCOProgramBuilder& b) {
        SmallVector<Value> wires;
        wires.reserve(numControls + 1);
        for (size_t i = 0; i <= numControls; ++i) {
          wires.push_back(b.staticQubit(reverseWires ? numControls - i : i));
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
    if (failed(funcOp.insertArgument(0, Float64Type::get(context), {},
                                     funcOp.getLoc()))) {
      return {};
    }
    parameter.replaceAllUsesWith(funcOp.getArgument(0));
  }
  return moduleOp;
}

// R_a(θ) = cos(θ/2) I - i sin(θ/2) sigma_a.
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

static void expectImplementsControlledGate(
    func::FuncOp funcOp, size_t numControls, const dd::GateMatrix& matrix,
    size_t target, const DDArgumentBindings& bindings = DDArgumentBindings()) {
  ASSERT_EQ(countStaticQubits(funcOp), numControls + 1);
  const auto package = llvm::cantFail(dd::Package::create(numControls + 1));
  const auto actual = buildFunctionality(funcOp, *package, bindings);
  ASSERT_TRUE(succeeded(actual));
  const auto expected =
      makeControlledGateDD(*package, numControls, matrix, target);
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

/// Compare the complete states, including phase, without requiring identical
/// DD nodes after floating-point synthesis.
static void expectStatesNear(dd::Package& package, const dd::VectorDD& actual,
                             const dd::VectorDD& expected) {
  auto negativeExpected = expected;
  negativeExpected.w = package.cn.lookup(-dd::RealNumber::val(expected.w.r),
                                         -dd::RealNumber::val(expected.w.i));
  const auto difference = package.add(actual, negativeExpected);
  package.incRef(difference);
  constexpr double tolerance = 3e-11;
  EXPECT_LE(package.innerProduct(difference, difference).r,
            tolerance * tolerance);
  package.decRef(difference);
}

/// Compare every output amplitude for the inputs with zero high-order ancillas.
///
/// Includes ancilla leakage and exact phase; stops at the first mismatch.
static void expectOperatorColumnsNear(const dd::MatrixDD& actual,
                                      const dd::MatrixDD& expected,
                                      size_t width, size_t inputWidth) {
  for (size_t column = 0; column < (size_t{1} << inputWidth); ++column) {
    for (size_t row = 0; row < (size_t{1} << width); ++row) {
      const auto error = std::abs(
          llvm::cantFail(dd::getValueByIndex(actual, width, row, column)) -
          llvm::cantFail(dd::getValueByIndex(expected, width, row, column)));
      if (!(error <= 1e-10)) {
        FAIL() << "row=" << row << " column=" << column << " error=" << error;
      }
    }
  }
}

static void expectMatchesReferenceOnStates(
    func::FuncOp funcOp, size_t numControls, const dd::GateMatrix& matrix,
    const std::vector<std::vector<dd::BasisStates>>& states) {
  const auto numQubits = countStaticQubits(funcOp);
  ASSERT_EQ(numQubits, numControls + 1);
  expectFullyDecomposed(funcOp);
  const auto dd = llvm::cantFail(dd::Package::create(numQubits));
  const auto referenceGate =
      makeControlledGateDD(*dd, numControls, matrix, numControls);
  dd->incRef(referenceGate);
  std::mt19937_64 rng(0);
  for (const auto& state : states) {
    const auto actual = simulate(
        funcOp, llvm::cantFail(dd::makeBasisState(numQubits, state, *dd)), *dd,
        rng);
    ASSERT_TRUE(succeeded(actual));
    const auto expected = dd->applyOperation(
        referenceGate,
        llvm::cantFail(dd::makeBasisState(numQubits, state, *dd)));
    expectStatesNear(*dd, *actual, expected);
    dd->decRef(*actual);
    dd->decRef(expected);
  }
  dd->decRef(referenceGate);
}

static void expectMatchesReferenceOnBasisStates(func::FuncOp funcOp,
                                                size_t numControls,
                                                ControlledPauli pauli) {
  std::vector<std::vector<dd::BasisStates>> states(
      2, std::vector(numControls + 1, dd::BasisStates::one));
  states.front()[numControls] = dd::BasisStates::zero;
  for (const size_t inactiveControl :
       std::array<size_t, 3>{0U, numControls / 2U, numControls - 1U}) {
    states.emplace_back(numControls + 1, dd::BasisStates::one);
    states.back()[inactiveControl] = dd::BasisStates::zero;
    states.back()[numControls] = dd::BasisStates::zero;
  }
  expectMatchesReferenceOnStates(funcOp, numControls, pauliMatrix(pauli),
                                 states);
}

static void
expectMatchesReferenceOnCoherentState(func::FuncOp funcOp, size_t numControls,
                                      bool targetOne,
                                      const dd::GateMatrix& matrix) {
  std::vector<dd::BasisStates> state(numControls + 1, dd::BasisStates::one);
  state[numControls / 2] = dd::BasisStates::plus;
  state[numControls] = targetOne ? dd::BasisStates::one : dd::BasisStates::zero;
  expectMatchesReferenceOnStates(funcOp, numControls, matrix, {state});
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

template <typename Gate>
[[nodiscard]] static size_t countSingleControlledGates(ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&count](CtrlOp op) {
    auto inner =
        mlir::mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (op.getNumControls() == 1 && op.getNumTargets() == 1 && inner &&
        isa<Gate>(inner.getOperation())) {
      ++count;
    }
  });
  return count;
}

[[nodiscard]] static size_t countEffectiveCxOps(ModuleOp moduleOp) {
  return countSingleControlledGates<XOp>(moduleOp) +
         (2 * countSingleControlledGates<POp>(moduleOp));
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

static std::optional<RotationAxis> ancillaRotationAxis(AncillaFamily family) {
  if (family == AncillaFamily::RX) {
    return RotationAxis::X;
  }
  if (family == AncillaFamily::RY) {
    return RotationAxis::Y;
  }
  if (family == AncillaFamily::RZ) {
    return RotationAxis::Z;
  }
  return std::nullopt;
}

static Value applyAncillaAngleGate(QCOProgramBuilder& builder,
                                   AncillaFamily family, Value theta,
                                   Value target) {
  if (auto axis = ancillaRotationAxis(family)) {
    return applyRotation(builder, *axis, theta, target);
  }
  return builder.p(theta, target);
}

static void buildAncillaFamilyGate(QCOProgramBuilder& builder,
                                   AncillaFamily family, ValueRange data) {
  if (auto axis = ancillaRotationAxis(family)) {
    builder.ctrl(data.drop_back(), data.back(), [&](Value target) {
      return applyRotation(builder, *axis, builder.floatConstant(0.73), target);
    });
  } else if (family == AncillaFamily::Swap) {
    builder.mcswap(data.drop_back(2), data[data.size() - 2], data.back());
  } else if (family == AncillaFamily::X) {
    builder.mcx(data.drop_back(), data.back());
  } else if (family == AncillaFamily::Y) {
    builder.mcy(data.drop_back(), data.back());
  } else if (family == AncillaFamily::Phase ||
             family == AncillaFamily::PhasePi) {
    builder.mcp(family == AncillaFamily::Phase ? 0.73 : -std::numbers::pi,
                data.drop_back(), data.back());
  } else {
    builder.mcz(data.drop_back(), data.back());
  }
}

static dd::MatrixDD ancillaFamilyReference(dd::Package& package,
                                           AncillaFamily family, size_t k) {
  if (auto axis = ancillaRotationAxis(family)) {
    return makeControlledGateDD(package, k, rotationMatrix(*axis, 0.73), k);
  }
  if (family == AncillaFamily::Swap) {
    return llvm::cantFail(
        makeGateDD(package, DynamicMatrix{SWAPOp::getUnitaryMatrix()},
                   {static_cast<dd::Qubit>(k - 1), static_cast<dd::Qubit>(k)},
                   makeControls(k - 1)));
  }
  if (family == AncillaFamily::Phase || family == AncillaFamily::PhasePi) {
    return makeControlledGateDD(
        package, k,
        phaseMatrix(family == AncillaFamily::Phase ? 0.73 : -std::numbers::pi),
        k);
  }
  return makeControlledGateDD(
      package, k,
      pauliMatrix(family == AncillaFamily::X   ? ControlledPauli::X
                  : family == AncillaFamily::Y ? ControlledPauli::Y
                                               : ControlledPauli::Z),
      k);
}

TEST_P(AncillaFamilyTest, FullAncillaLadderPreservesOperator) {
  const auto family = GetParam();
  // Cover the smallest useful Pauli ladders and the rotation cost crossover.
  for (const auto [k, clean] : {
           std::pair{3U, true},
           std::pair{5U, true},
           std::pair{4U, false},
           std::pair{6U, false},
           std::pair{7U, false},
       }) {
    SCOPED_TRACE(testing::Message() << "k=" << k << " clean=" << clean);
    const size_t width = 2 * k - 1;
    auto module =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          SmallVector<Value> wires;
          for (size_t i = 0; i < width; ++i) {
            wires.push_back(clean ? b.allocQubit() : b.staticQubit(i));
          }
          buildAncillaFamilyGate(b, family,
                                 ValueRange(wires).take_front(k + 1));
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module)));
    expectFullyLowered(*module);
    auto package = llvm::cantFail(dd::Package::create(width));
    auto actual =
        buildFunctionality(*module->getOps<func::FuncOp>().begin(), *package);
    ASSERT_TRUE(succeeded(actual));
    auto expected = ancillaFamilyReference(*package, family, k);
    if (clean) {
      package->incRef(expected);
      expectOperatorColumnsNear(*actual, expected, width, k + 1);
      package->decRef(expected);
    } else {
      EXPECT_EQ(*actual, expected);
    }
    package->decRef(*actual);

    if (ancillaRotationAxis(family)) {
      const size_t ladderCost = clean ? 6 * k - 8 : 8 * k - 6;
      EXPECT_EQ(countEffectiveCxOps(*module),
                std::min(ladderCost, expectedMcrCxBudget(k)));
    } else if (family == AncillaFamily::Phase) {
      EXPECT_EQ(countEffectiveCxOps(*module),
                clean ? 6 * k - 6 : expectedMcpCx(k));
    } else {
      const size_t shell = family == AncillaFamily::Swap ? 2 : 0;
      EXPECT_EQ(countEffectiveCxOps(*module),
                shell + (clean ? 6 * k - 6 : 8 * k - 6));
    }
  }
}

TEST_P(AncillaFamilyTest, FallsBackWithoutAvailableAncillas) {
  const auto family = GetParam();
  auto module = QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
    b.sink(b.allocQubit());
    auto tensor = b.qtensorAlloc(6);
    SmallVector<Value> data;
    for (int64_t i = 0; i < 6; ++i) {
      auto [rest, qubit] = b.qtensorExtract(tensor, i);
      tensor = rest;
      data.push_back(qubit);
    }
    buildAncillaFamilyGate(b, family, data);
    std::ignore = b.allocQubit();
    return SmallVector<Value>{};
  });
  ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module)));
  // SWAP targets, tensor holes, dead wires, and future wires are ineligible.
  const size_t expected = ancillaRotationAxis(family)     ? 28
                          : family == AncillaFamily::Swap ? 62
                                                          : 60;
  EXPECT_EQ(countEffectiveCxOps(*module), expected);
}

TEST_P(AncillaFamilyTest, RespectsNativeGatesAndWidthWithAvailableAncillas) {
  context()->loadDialect<mlir::mqt::MQTDialect>();
  const auto family = GetParam();
  for (const StringRef mode : {"width", "target", "environment"}) {
    SCOPED_TRACE(mode.str());
    auto module =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          SmallVector<Value> wires;
          for (size_t i = 0; i < 9; ++i) {
            wires.push_back(b.allocQubit());
          }
          buildAncillaFamilyGate(b, family, ValueRange(wires).take_front(6));
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(succeeded(verify(*module)));
    ASSERT_TRUE(succeeded(verifyLinearity(*module)));
    if (mode != "width") {
      using Capability = CompilerTarget::OperationCapability;
      const bool phase =
          family == AncillaFamily::Phase || family == AncillaFamily::PhasePi;
      const std::vector operations{
          llvm::cantFail(
              Capability::create(family == AncillaFamily::X    ? "x"
                                 : family == AncillaFamily::Y  ? "y"
                                 : family == AncillaFamily::Z  ? "z"
                                 : phase                       ? "p"
                                 : family == AncillaFamily::RX ? "rx"
                                 : family == AncillaFamily::RY ? "ry"
                                 : family == AncillaFamily::RZ ? "rz"
                                                               : "swap",
                                 Capability::Arity::variadic(6),
                                 phase || ancillaRotationAxis(family) ? 1 : 0)),
      };
      const auto target = llvm::cantFail(CompilerTarget::create(
          9, CompilerTarget::Connectivity::allToAll(),
          CompilerTarget::NativeOperations::fromOperations(operations)));
      if (mode == "environment") {
        attachTargetEnvironment(
            *module,
            TargetEnvironment(target,
                              llvm::cantFail(PayloadSpecification::create(
                                  PayloadFormat{.id = "mqt.test.payload"}))));
      }
      PassManager pm(context());
      pm.addPass(mode == "target"
                     ? createDecomposeMultiControlled(target)
                     : createDecomposeMultiControlled(
                           DecomposeMultiControlledOptions{.useTarget = true}));
      ASSERT_TRUE(succeeded(pm.run(*module)));
      ASSERT_TRUE(succeeded(verifyLinearity(*module)));
    } else {
      DecomposeMultiControlledOptions options;
      options.minQubits = 7;
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module, options)));
    }
    EXPECT_EQ(countMultiControlledOps(*module), 1U);
    EXPECT_EQ(countEffectiveCxOps(*module), 0U);
    auto package = llvm::cantFail(dd::Package::create(9));
    auto actual =
        buildFunctionality(*module->getOps<func::FuncOp>().begin(), *package);
    ASSERT_TRUE(succeeded(actual));
    EXPECT_EQ(*actual, ancillaFamilyReference(*package, family, 5));
    package->decRef(*actual);
  }
}

TEST_F(MultiControlledDecompositionTest, TargetLimitsAncillaBorrowing) {
  context()->loadDialect<mlir::mqt::MQTDialect>();
  using Capability = CompilerTarget::OperationCapability;
  for (const bool preplaced : {false, true}) {
    SCOPED_TRACE(preplaced);
    const auto target = llvm::cantFail(CompilerTarget::create(
        preplaced ? 9 : 8,
        preplaced ? CompilerTarget::Connectivity::fromCouplings({
                        {0, 1},
                        {1, 2},
                        {2, 3},
                        {3, 4},
                        {4, 5},
                        {5, 6},
                        {6, 7},
                        {7, 8},
                    })
                  : CompilerTarget::Connectivity::allToAll(),
        CompilerTarget::NativeOperations::fromOperations(
            {llvm::cantFail(Capability::create("x", 2, 0))})));
    const TargetEnvironment environment(
        target, llvm::cantFail(PayloadSpecification::create(
                    PayloadFormat{.id = "mqt.test.payload"})));
    for (const bool useEnvironment : {false, true}) {
      SCOPED_TRACE(useEnvironment);
      auto moduleOp =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& builder) {
            SmallVector<Value> wires;
            for (size_t site = 0; site < 9; ++site) {
              wires.push_back(preplaced ? builder.staticQubit(site)
                                        : builder.allocQubit());
            }
            builder.mcx(ValueRange(wires).take_front(5), wires[5]);
            return SmallVector<Value>{};
          });
      ASSERT_TRUE(moduleOp);
      if (useEnvironment) {
        attachTargetEnvironment(*moduleOp, environment);
      }
      PassManager pm(context());
      pm.addPass(useEnvironment
                     ? createDecomposeMultiControlled(
                           DecomposeMultiControlledOptions{.useTarget = true})
                     : createDecomposeMultiControlled(target));
      ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
      ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
      // Capacity and fixed connectivity can forbid borrowing spare wires.
      EXPECT_EQ(countEffectiveCxOps(*moduleOp), expectedMcxCx(5));
      auto package = llvm::cantFail(dd::Package::create(9));
      auto actual = buildFunctionality(
          *moduleOp->getOps<func::FuncOp>().begin(), *package);
      ASSERT_TRUE(succeeded(actual));
      EXPECT_EQ(*actual, ancillaFamilyReference(*package, AncillaFamily::X, 5));
      package->decRef(*actual);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Ancilla, AncillaFamilyTest,
                         testing::Values(AncillaFamily::X, AncillaFamily::Y,
                                         AncillaFamily::Z, AncillaFamily::Swap,
                                         AncillaFamily::Phase,
                                         AncillaFamily::PhasePi,
                                         AncillaFamily::RX, AncillaFamily::RY,
                                         AncillaFamily::RZ),
                         [](const testing::TestParamInfo<AncillaFamily>& info) {
                           return ancillaFamilyName(info.param);
                         });

TEST_P(AncillaAngleTest, RuntimeAnglesAndAncillaRestoration) {
  const auto family = GetParam();
  const auto axis = ancillaRotationAxis(family);
  // Prefer cheaper clean ancillas even when a dirty ancilla appears first.
  for (const bool mixed : {false, true}) {
    SCOPED_TRACE(testing::Message() << "mixed=" << mixed);
    Value parameter;
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          parameter = b.floatConstant(0.73);
          SmallVector<Value> wires;
          for (size_t i = 0; i < 9; ++i) {
            wires.push_back(mixed ? b.allocQubit() : b.staticQubit(i));
          }
          if (mixed) {
            wires[6] = b.x(wires[6]);
          }
          SmallVector<Value> controls{
              wires[5], wires[4], wires[3], wires[1], wires[0],
          };
          b.ctrl(controls, wires[2], [&](Value target) {
            auto angle = arith::NegFOp::create(b, parameter);
            return applyAncillaAngleGate(b, family, angle, target);
          });
          if (mixed) {
            b.x(wires[6]);
          }
          return SmallVector<Value>{};
        });
    auto function = *moduleOp->getOps<func::FuncOp>().begin();
    ASSERT_TRUE(succeeded(function.insertArgument(
        0, Float64Type::get(context()), {}, function.getLoc())));
    parameter.replaceAllUsesWith(function.getArgument(0));
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    expectFullyLowered(*moduleOp);
    EXPECT_EQ(countEffectiveCxOps(*moduleOp),
              axis ? (mixed ? 22U : 28U) : (mixed ? 28U : 60U));
    for (double theta : K_ANCILLA_ANGLES) {
      SCOPED_TRACE(theta);
      auto package = llvm::cantFail(dd::Package::create(9));
      const DDArgumentBindings bindings{
          {
              function.getArgument(0),
              FloatAttr::get(Float64Type::get(context()), theta),
          },
      };
      auto actual = buildFunctionality(function, *package, bindings);
      ASSERT_TRUE(succeeded(actual));
      auto expected = makeControlledGateDD(
          *package, 5,
          axis ? rotationMatrix(*axis, -theta) : phaseMatrix(-theta), 2);
      package->incRef(expected);
      if (mixed) {
        expectOperatorColumnsNear(*actual, expected, 9, 7);
      } else {
        EXPECT_EQ(*actual, expected);
      }
      package->decRef(expected);
      package->decRef(*actual);
    }
  }
}

TEST_P(AncillaAngleTest, DirtyConstantAnglesAndMixedSelection) {
  const auto family = GetParam();
  const auto axis = ancillaRotationAxis(family);
  for (const bool mixed : {false, true}) {
    for (double theta : K_ANCILLA_ANGLES) {
      SCOPED_TRACE(testing::Message()
                   << "mixed=" << mixed << " theta=" << theta);
      constexpr size_t k = 6;
      auto moduleOp =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
            auto reg = b.allocQubitRegister(11);
            SmallVector<Value> dirty;
            for (size_t i = 7; i < (mixed ? 10U : 11U); ++i) {
              dirty.push_back(b.x(reg[i]));
            }
            SmallVector<Value> controls{
                reg[6], reg[5], reg[4], reg[3], reg[1], reg[0],
            };
            b.ctrl(controls, reg[2], [&](Value target) {
              return applyAncillaAngleGate(b, family, b.floatConstant(theta),
                                           target);
            });
            for (auto ancilla : dirty) {
              b.x(ancilla);
            }
            return SmallVector<Value>{};
          });
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
      if (theta == 0.73) {
        EXPECT_EQ(countEffectiveCxOps(*moduleOp), axis    ? (mixed ? 34U : 40U)
                                                  : mixed ? 66U
                                                          : 100U);
      }
      auto package = llvm::cantFail(dd::Package::create(11));
      auto actual = buildFunctionality(
          *moduleOp->getOps<func::FuncOp>().begin(), *package);
      ASSERT_TRUE(succeeded(actual));
      auto expected = makeControlledGateDD(
          *package, k, axis ? rotationMatrix(*axis, theta) : phaseMatrix(theta),
          2);
      package->incRef(expected);
      if (mixed) {
        // Only the last wire is known zero. All dirty/data states remain free.
        expectOperatorColumnsNear(*actual, expected, 11, 10);
      } else {
        EXPECT_EQ(*actual, expected);
      }
      package->decRef(expected);
      package->decRef(*actual);
    }
  }
}

TEST_P(AncillaAngleTest, CleanAncillaWithBorrowedResidualControls) {
  const auto family = GetParam();
  const auto axis = ancillaRotationAxis(family);
  for (const size_t k : {3U, 4U, 8U, 9U}) {
    SCOPED_TRACE(k);
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          auto reg = b.allocQubitRegister(static_cast<int64_t>(k + 2));
          b.ctrl(ValueRange(reg.qubits).take_front(k), reg[k],
                 [&](Value target) {
                   return applyAncillaAngleGate(b, family,
                                                b.floatConstant(0.73), target);
                 });
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    expectFullyLowered(*moduleOp);
    const auto fallback = axis ? expectedMcrCxBudget(k) : expectedMcpCx(k);
    const auto residual =
        axis ? expectedMcrCxBudget(k - 1) : expectedMcpCx(k - 1);
    EXPECT_EQ(countEffectiveCxOps(*moduleOp), std::min(fallback, 6 + residual));
    auto package = llvm::cantFail(dd::Package::create(k + 2));
    auto actual =
        buildFunctionality(*moduleOp->getOps<func::FuncOp>().begin(), *package);
    ASSERT_TRUE(succeeded(actual));
    auto expected = makeControlledGateDD(
        *package, k, axis ? rotationMatrix(*axis, 0.73) : phaseMatrix(0.73), k);
    package->incRef(expected);
    expectOperatorColumnsNear(*actual, expected, k + 2, k + 1);
    package->decRef(expected);
    package->decRef(*actual);
  }
}

TEST_P(AncillaFamilyTest, OneDirtyAncillaFullOperator) {
  const auto family = GetParam();
  const auto axis = ancillaRotationAxis(family);
  for (const size_t k : {5U, 8U, 9U}) {
    SCOPED_TRACE(k);
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          SmallVector<Value> wires;
          for (size_t i = 0; i < k + 2; ++i) {
            wires.push_back(b.staticQubit(i));
          }
          buildAncillaFamilyGate(b, family,
                                 ValueRange(wires).take_front(k + 1));
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    expectFullyLowered(*moduleOp);
    if (axis || family == AncillaFamily::Phase) {
      EXPECT_EQ(countEffectiveCxOps(*moduleOp),
                axis ? expectedMcrCxBudget(k) : expectedMcpCx(k));
    } else {
      EXPECT_LT(countEffectiveCxOps(*moduleOp),
                expectedMcpCx(k) + (family == AncillaFamily::Swap ? 2 : 0));
    }
    auto package = llvm::cantFail(dd::Package::create(k + 2));
    auto actual =
        buildFunctionality(*moduleOp->getOps<func::FuncOp>().begin(), *package);
    ASSERT_TRUE(succeeded(actual));
    EXPECT_EQ(*actual, ancillaFamilyReference(*package, family, k));
    package->decRef(*actual);
  }
}

TEST_F(MultiControlledDecompositionTest, RuntimePhaseNeedsFullDirtyLadder) {
  // Nine controls are the first width where the full dirty ladder wins.
  constexpr size_t k = 9;
  constexpr double theta = 0.73;
  for (const size_t numAncillas : {6U, 7U}) {
    SCOPED_TRACE(numAncillas);
    const size_t width = k + 1 + numAncillas;
    Value parameter;
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          SmallVector<Value> wires;
          for (size_t i = 0; i < width; ++i) {
            wires.push_back(b.staticQubit(i));
          }
          parameter = b.floatConstant(theta);
          b.ctrl(ValueRange(wires).take_front(k), wires[k], [&](Value target) {
            return b.p(arith::NegFOp::create(b, parameter), target);
          });
          return SmallVector<Value>{};
        });
    auto function = *moduleOp->getOps<func::FuncOp>().begin();
    ASSERT_TRUE(succeeded(function.insertArgument(
        0, Float64Type::get(context()), {}, function.getLoc())));
    parameter.replaceAllUsesWith(function.getArgument(0));
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    expectFullyLowered(*moduleOp);
    EXPECT_EQ(countEffectiveCxOps(*moduleOp),
              numAncillas == k - 2 ? 282U : 290U);
    auto package = llvm::cantFail(dd::Package::create(width));
    const DDArgumentBindings bindings{
        {
            function.getArgument(0),
            FloatAttr::get(Float64Type::get(context()), theta),
        },
    };
    auto actual = buildFunctionality(function, *package, bindings);
    ASSERT_TRUE(succeeded(actual));
    EXPECT_EQ(*actual,
              makeControlledGateDD(*package, k, phaseMatrix(-theta), k));
    package->decRef(*actual);
  }
}

TEST_P(AncillaAngleTest, ResetAndRestoredAncillaReuse) {
  const auto family = GetParam();
  for (const bool reset : {false, true}) {
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          auto reg = b.allocQubitRegister(7);
          auto ancilla = b.measure(b.h(reg[6])).first;
          if (reset) {
            ancilla = b.reset(ancilla);
          }
          for (size_t i = 0; i < 5; ++i) {
            reg.qubits[i] = b.h(reg[i]);
          }
          auto [controls, target] =
              b.ctrl(ValueRange(reg.qubits).take_front(5), b.ry(0.43, reg[5]),
                     [&](Value arg) {
                       return applyAncillaAngleGate(b, family,
                                                    b.floatConstant(0.73), arg);
                     });
          b.ctrl(controls, b.rz(0.37, target), [&](Value arg) {
            return applyAncillaAngleGate(b, family, b.floatConstant(-1.21),
                                         arg);
          });
          b.measure(ancilla);
          return SmallVector<Value>{};
        });
    auto package = llvm::cantFail(dd::Package::create(7));
    auto function = *moduleOp->getOps<func::FuncOp>().begin();
    std::mt19937_64 beforeRng(7), afterRng(7);
    std::vector<dd::BasisStates> zero(7, dd::BasisStates::zero);
    auto expected = simulate(
        function, llvm::cantFail(dd::makeBasisState(7, zero, *package)),
        *package, beforeRng);
    ASSERT_TRUE(succeeded(expected));
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    auto actual = simulate(
        function, llvm::cantFail(dd::makeBasisState(7, zero, *package)),
        *package, afterRng);
    ASSERT_TRUE(succeeded(actual));
    expectStatesNear(*package, *actual, *expected);
    package->decRef(*actual);
    package->decRef(*expected);
    // Measurement alone is not zero; reset is. The restored zero proof must
    // survive the first gate so the second can borrow the same tensor slot.
    EXPECT_EQ(countEffectiveCxOps(*moduleOp), ancillaRotationAxis(family)
                                                  ? (reset ? 44U : 56U)
                                                  : (reset ? 76U : 120U));
  }
}

INSTANTIATE_TEST_SUITE_P(Ancilla, AncillaAngleTest,
                         testing::Values(AncillaFamily::Phase,
                                         AncillaFamily::RX, AncillaFamily::RY,
                                         AncillaFamily::RZ),
                         [](const testing::TestParamInfo<AncillaFamily>& info) {
                           return ancillaFamilyName(info.param);
                         });

TEST_F(MultiControlledDecompositionTest,
       AncillaTensorSlotsAndTerminalMeasurements) {
  for (const bool dirty : {false, true}) {
    for (const bool measured : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "dirty=" << dirty << " measured=" << measured);
      auto module =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
            auto tensor = b.qtensorAlloc(7);
            if (dirty) {
              auto [rest, ancilla] = b.qtensorExtract(tensor, 6);
              tensor = b.qtensorInsert(b.h(ancilla), rest, 6);
            }
            SmallVector<Value> wires;
            for (int64_t i = 0; i < 6; ++i) {
              auto [rest, qubit] = b.qtensorExtract(tensor, i);
              tensor = rest;
              wires.push_back(i < 5 ? b.h(qubit) : b.ry(0.43, qubit));
            }
            auto [controls, target] =
                b.mcx(ValueRange(wires).take_front(5), wires[5]);
            SmallVector<Value> outputs(controls);
            outputs.push_back(target);
            for (size_t i = 0; i < outputs.size(); ++i) {
              auto qubit = outputs[i];
              if (measured) {
                qubit = b.measure(qubit).first;
              }
              tensor = b.qtensorInsert(qubit, tensor, static_cast<int64_t>(i));
            }
            return SmallVector<Value>{};
          });
      auto package = llvm::cantFail(dd::Package::create(7));
      auto function = *module->getOps<func::FuncOp>().begin();
      auto expected = simulateStatevector(function, *package);
      ASSERT_TRUE(succeeded(expected));
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module)));
      auto actual = simulateStatevector(function, *package);
      ASSERT_TRUE(succeeded(actual));
      expectStatesNear(*package, *actual, *expected);
      package->decRef(*actual);
      package->decRef(*expected);
      EXPECT_EQ(countEffectiveCxOps(*module), dirty ? 56U : 42U);
      size_t measurements = 0;
      module->walk([&](MeasureOp) { ++measurements; });
      EXPECT_EQ(measurements, measured ? 6U : 0U);
    }
  }
}

TEST_F(MultiControlledDecompositionTest, AncillaCleanExtractedSlotCanBeReused) {
  auto module = QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(7);
    for (size_t i = 0; i < 5; ++i) {
      reg.qubits[i] = b.h(reg.qubits[i]);
    }
    auto [controls, target] =
        b.mcx(ValueRange(reg.qubits).take_front(5), reg[5]);
    target = b.rz(0.37, target);
    b.mcx(controls, target);
    return SmallVector<Value>{};
  });
  auto package = llvm::cantFail(dd::Package::create(7));
  auto function = *module->getOps<func::FuncOp>().begin();
  auto expected = simulateStatevector(function, *package);
  ASSERT_TRUE(succeeded(expected));
  ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module)));
  auto actual = simulateStatevector(function, *package);
  ASSERT_TRUE(succeeded(actual));
  expectStatesNear(*package, *actual, *expected);
  package->decRef(*actual);
  package->decRef(*expected);
  EXPECT_EQ(countEffectiveCxOps(*module), 84U);
}

TEST_F(MultiControlledDecompositionTest, AncillaDiscoveryRestartsForEachBlock) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
        const SmallVector<Type> types(7, QubitType::get(context()));
        for (const auto* name : {"first", "second"}) {
          b.createFunction(name, types, [&](ValueRange wires) {
            auto [controls, target] = b.mcx(wires.take_front(5), wires[5]);
            auto [nextControls, nextTarget] =
                b.mcx(controls, b.rz(0.37, target));
            SmallVector<Value> outputs(nextControls);
            outputs.push_back(nextTarget);
            outputs.push_back(wires[6]);
            return outputs;
          });
        }
        return SmallVector<Value>{};
      });
  auto first = *moduleOp->getOps<func::FuncOp>().begin();
  auto package = llvm::cantFail(dd::Package::create(7));
  auto expected = buildFunctionality(first, *package);
  ASSERT_TRUE(succeeded(expected));
  ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
  EXPECT_EQ(countEffectiveCxOps(*moduleOp), 224U);
  for (auto function : moduleOp->getOps<func::FuncOp>()) {
    if (function.getNumArguments() == 0) {
      continue;
    }
    auto actual = buildFunctionality(function, *package);
    ASSERT_TRUE(succeeded(actual));
    expectOperatorColumnsNear(*actual, *expected, 7, 7);
    package->decRef(*actual);
  }
  package->decRef(*expected);
}

TEST_F(MultiControlledDecompositionTest, BorrowsMultipleAncillasFromOneTensor) {
  auto module = QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
    auto tensor = b.qtensorAlloc(7);
    SmallVector<Value> wires;
    for (int64_t i = 0; i < 5; ++i) {
      auto [rest, qubit] = b.qtensorExtract(tensor, i);
      tensor = rest;
      wires.push_back(qubit);
    }
    b.mcx(ValueRange(wires).take_front(4), wires[4]);
    return SmallVector<Value>{};
  });
  ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module)));
  EXPECT_EQ(countEffectiveCxOps(*module), 18U);
  auto package = llvm::cantFail(dd::Package::create(7));
  auto actual =
      buildFunctionality(*module->getOps<func::FuncOp>().begin(), *package);
  ASSERT_TRUE(succeeded(actual));
  auto expected =
      makeControlledGateDD(*package, 4, pauliMatrix(ControlledPauli::X), 4);
  package->incRef(expected);
  expectOperatorColumnsNear(*actual, expected, 7, 5);
  package->decRef(expected);
  package->decRef(*actual);
}

TEST_F(MultiControlledDecompositionTest,
       AncillaDiscoveryPrefersLaterCleanSlotsInLargeTensor) {
  // Discovery needs at most controls - 2 ancillas, even in a large register.
  // An earlier dirty wire cannot prevent a later clean slot from being used.
  for (const int64_t slots : {1, 65536}) {
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          SmallVector<Value> wires;
          for (size_t i = 0; i < 6; ++i) {
            wires.push_back(b.allocQubit());
          }
          auto dirty = b.h(b.allocQubit());
          b.qtensorAlloc(slots);
          b.mcx(ValueRange(wires).take_front(5), wires[5]);
          b.measure(dirty);
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    EXPECT_EQ(countEffectiveCxOps(*moduleOp), slots == 1 ? 42U : 24U);
  }
}

TEST_F(MultiControlledDecompositionTest,
       AncillaFromElementsPrefersLaterCleanSlots) {
  auto module = QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
    auto data = b.allocQubitRegister(5);
    SmallVector<Value> ancillas;
    for (size_t i = 0; i < 4; ++i) {
      Value qubit = b.allocQubit();
      ancillas.push_back(i < 2 ? b.h(qubit) : qubit);
    }
    b.qtensorFromElements(ancillas);
    b.mcx(ValueRange(data.qubits).take_front(4), data[4]);
    return SmallVector<Value>{};
  });
  ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module)));
  EXPECT_EQ(countEffectiveCxOps(*module), 18U);
}

TEST_F(MultiControlledDecompositionTest,
       AncillaDiscoveryHandlesSparseOccupancyInHugeTensor) {
  constexpr int64_t slots = 1099511627776;
  for (const bool restore : {false, true}) {
    qtensor::ExtractOp unknownExtract;
    auto module =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          auto tensor = b.qtensorAlloc(slots);
          SmallVector<Value> wires;
          for (int64_t i = 0; i < 6; ++i) {
            auto [rest, qubit] = b.qtensorExtract(tensor, i);
            tensor = rest;
            wires.push_back(qubit);
          }
          auto [rest, ancilla] = b.qtensorExtract(tensor, slots - 1);
          unknownExtract = ancilla.getDefiningOp<qtensor::ExtractOp>();
          if (restore) {
            b.qtensorInsert(b.reset(ancilla), rest, slots - 1);
          } else {
            b.sink(ancilla);
          }
          b.mcx(ValueRange(wires).take_front(5), wires[5]);
          return SmallVector<Value>{};
        });
    auto function = *module->getOps<func::FuncOp>().begin();
    ASSERT_TRUE(succeeded(function.insertArgument(0, IndexType::get(context()),
                                                  {}, function.getLoc())));
    // Only the extraction becomes dynamic; the later insertion proves
    // occupancy.
    unknownExtract.getIndexMutable().assign(function.getArgument(0));
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*module)));
    EXPECT_EQ(countEffectiveCxOps(*module), restore ? 42U : 60U);
  }
}

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
    expectImplementsControlledGate(
        funcOp, numControls, rotationMatrix(axis, theta), numControls / 2);
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
      ASSERT_TRUE(succeeded(funcOp.insertArgument(
          0, Float64Type::get(context()), {}, funcOp.getLoc())));
      parameter.replaceAllUsesWith(funcOp.getArgument(0));
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(moduleOp.get())));
      expectFullyLowered(moduleOp.get());
      expectFullyDecomposed(funcOp);
      for (const double theta : K_ANCILLA_ANGLES) {
        const DDArgumentBindings bindings{
            {
                funcOp.getArgument(0),
                FloatAttr::get(Float64Type::get(context()), theta),
            },
        };
        expectImplementsControlledGate(
            funcOp, numControls,
            rotationMatrix(axis, regionLocal ? -theta : theta), numControls / 2,
            bindings);
      }
    }
  }
}

TEST_F(MultiControlledDecompositionTest, PhasesPreserveRuntimeAngles) {
  for (const size_t numControls : {2U, 3U, 4U, 5U, 8U, 9U}) {
    for (const bool nested : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "controls=" << numControls << " nested=" << nested);
      Value parameter;
      auto moduleOp =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& builder) {
            parameter = builder.floatConstant(0.73);
            SmallVector<Value> wires;
            for (size_t i = numControls + 1; i-- > 0;) {
              wires.push_back(builder.staticQubit(i));
            }
            const auto phase = [&](Value target) {
              // Exercise supporting arithmetic inside the modifier body.
              auto angle = arith::NegFOp::create(builder, parameter);
              return builder.p(angle, target);
            };
            if (nested) {
              builder.ctrl(wires.front(), ValueRange(wires).drop_front(),
                           [&](ValueRange targets) {
                             auto [controls, target] = builder.ctrl(
                                 targets.drop_back(), targets.back(), phase);
                             SmallVector<Value> results(controls);
                             results.push_back(target);
                             return results;
                           });
            } else {
              builder.ctrl(ValueRange(wires).drop_back(), wires.back(), phase);
            }
            return SmallVector<Value>{};
          });
      ASSERT_TRUE(moduleOp);
      auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
      ASSERT_TRUE(succeeded(funcOp.insertArgument(
          0, Float64Type::get(context()), {}, funcOp.getLoc())));
      parameter.replaceAllUsesWith(funcOp.getArgument(0));
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(moduleOp.get())));
      expectFullyLowered(moduleOp.get());
      EXPECT_LE(countEffectiveCxOps(moduleOp.get()),
                expectedMcpCx(numControls));
      funcOp.walk([](AllocOp) { ADD_FAILURE() << "unexpected ancilla qubit"; });
      for (const double theta : K_ANCILLA_ANGLES) {
        SCOPED_TRACE(testing::Message() << "theta=" << theta);
        const DDArgumentBindings bindings{
            {
                funcOp.getArgument(0),
                FloatAttr::get(Float64Type::get(context()), theta),
            },
        };
        // Full operator equality includes the phase on coherent controls.
        expectFullyDecomposed(funcOp);
        expectImplementsControlledGate(funcOp, numControls, phaseMatrix(-theta),
                                       0, bindings);
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
            [](AllocOp) { ADD_FAILURE() << "unexpected ancilla qubit"; });
        // The same bound covers every axis and permits future cancellation.
        EXPECT_LE(countSingleControlledGates<XOp>(moduleOp.get()),
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
    expectImplementsControlledGate(funcOp, 3, rotationMatrix(axis, 0.73), 1);

    options.minQubits = 4;
    ASSERT_TRUE(
        succeeded(runDecomposeMultiControlled(moduleOp.get(), options)));
    EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
    expectImplementsControlledGate(funcOp, 3, rotationMatrix(axis, 0.73), 1);
  }
}

//===----------------------------------------------------------------------===//
// MCX / MCY / MCZ / MCP: DD + CX for k = 2..20 and k = 33
//===----------------------------------------------------------------------===//

TEST_P(McPauliDdTest, EquivalenceAndCxCount) {
  const auto [pauli, k] = GetParam();
  auto moduleOp = buildControlledPauliModule(context(), k, pauli,
                                             k <= K_MATRIX_DD_MAX_PAULI);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  expectFullyLowered(moduleOp.get());
  // MCY and MCZ share MCX's CX budget; their basis changes add no CX.
  EXPECT_EQ(countSingleControlledGates<XOp>(moduleOp.get()), expectedMcxCx(k))
      << "k=" << k;
  if (k <= 8) {
    // Count every entangler, including any leftover controlled phase gates.
    EXPECT_EQ(countSingleControlledGates<POp>(moduleOp.get()), 0U);
  }
  if (k <= K_MATRIX_DD_MAX_PAULI) {
    EXPECT_EQ(countEffectiveCxOps(moduleOp.get()),
              k == 9 ? expectedMcpCx(k) : expectedMcxCx(k));
  }

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  if (k <= K_MATRIX_DD_MAX_PAULI) {
    expectFullyDecomposed(funcOp);
    funcOp.walk([](AllocOp) { ADD_FAILURE() << "unexpected ancilla qubit"; });
    expectImplementsControlledGate(funcOp, k, pauliMatrix(pauli), 0);
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
    expectFullyDecomposed(funcOp);
    expectImplementsControlledGate(funcOp, k, phaseMatrix(theta), k);
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
      expectMatchesReferenceOnCoherentState(
          funcOp, k, pauli == ControlledPauli::Z, pauliMatrix(pauli));
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
    expectMatchesReferenceOnCoherentState(funcOp, k, true, phaseMatrix(theta));
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
  EXPECT_EQ(countSingleControlledGates<XOp>(moduleOp.get()), expectedMcxCx(k))
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

TEST_F(MultiControlledDecompositionTest, RCCXRespectsMinQubits) {
  for (const uint64_t minQubits : {3U, 4U}) {
    SCOPED_TRACE(minQubits);
    auto moduleOp = buildRCCXModule(context());
    ASSERT_TRUE(moduleOp);
    DecomposeMultiControlledOptions options;
    options.minQubits = minQubits;
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp, options)));
    EXPECT_EQ(countRCCXOps(*moduleOp), minQubits == 3 ? 0U : 1U);
  }
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

  using OperationCapability = CompilerTarget::OperationCapability;
  std::vector operations{
      llvm::cantFail(OperationCapability::create(
          "rccx", OperationCapability::Arity::variadic(4), 0)),
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
  for (const size_t controls : {2U, 3U, 4U}) {
    for (const auto pauli :
         {ControlledPauli::X, ControlledPauli::Y, ControlledPauli::Z}) {
      SCOPED_TRACE(testing::Message() << "controls=" << controls << " pauli="
                                      << static_cast<unsigned>(pauli));
      auto moduleOp = buildControlledPauliModule(context(), controls, pauli);
      ASSERT_TRUE(moduleOp);
      DecomposeMultiControlledOptions options;
      options.minQubits = controls + 2;
      ASSERT_TRUE(
          runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
      EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 1U);

      options.minQubits = controls + 1;
      ASSERT_TRUE(
          runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
      auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
      // A width threshold may retain smaller RCCX gates.
      funcOp.walk([&](UnitaryOpInterface gate) {
        EXPECT_LT(gate.getNumQubits(), options.minQubits);
      });
      expectImplementsControlledGate(funcOp, controls, pauliMatrix(pauli),
                                     controls);

      options.minQubits = 3;
      ASSERT_TRUE(
          runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
      expectFullyDecomposed(funcOp);
      expectImplementsControlledGate(funcOp, controls, pauliMatrix(pauli),
                                     controls);

      options.minQubits = 2;
      EXPECT_FALSE(
          runDecomposeMultiControlled(moduleOp.get(), options).succeeded());
    }
  }
}

TEST_F(MultiControlledDecompositionTest, AncillaWideFallbackPhasesAndCounts) {
  // Single-gate compact-DD fixtures, not dense matrices at these widths.
  for (size_t k : {22U, 23U, 24U}) {
    for (const double theta : {0.0, std::numbers::pi, -std::numbers::pi}) {
      const bool swap = theta == 0.0;
      SCOPED_TRACE(testing::Message()
                   << "controls=" << k << " theta=" << theta);
      auto moduleOp =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
            SmallVector<Value> wires;
            for (size_t i = 0; i <= k; ++i) {
              wires.push_back(b.staticQubit(i));
            }
            // Reverse the controls; keep both SWAP targets inside the data set.
            SmallVector<Value> controls(wires.begin(),
                                        wires.end() - (swap ? 2 : 1));
            std::ranges::reverse(controls);
            if (swap) {
              b.mcswap(controls, wires[k], wires[k - 1]);
            } else {
              b.mcp(theta, controls, wires[k]);
            }
            return SmallVector<Value>{};
          });
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
      auto function = *moduleOp->getOps<func::FuncOp>().begin();
      expectFullyLowered(*moduleOp);
      EXPECT_EQ(countStaticQubits(function), k + 1);
      function.walk(
          [](AllocOp) { ADD_FAILURE() << "unexpected ancilla qubit"; });
      const size_t cost =
          k < 23 ? expectedMcpCx(k) : (k % 2 == 1 ? 88 * k - 54 : 96 * k - 154);
      EXPECT_EQ(countEffectiveCxOps(*moduleOp), cost + (swap ? 2 : 0));
      auto package = llvm::cantFail(dd::Package::create(k + 1));
      auto reference =
          swap ? llvm::cantFail(makeGateDD(
                     *package, DynamicMatrix{SWAPOp::getUnitaryMatrix()},
                     {static_cast<dd::Qubit>(k), static_cast<dd::Qubit>(k - 1)},
                     makeControls(k - 1)))
               : makeControlledGateDD(*package, k,
                                      pauliMatrix(ControlledPauli::Z), k);
      package->incRef(reference);
      std::mt19937_64 rng(7);
      for (size_t scenario = 0; scenario < 6; ++scenario) {
        std::vector<dd::BasisStates> state(k + 1, dd::BasisStates::one);
        state[k] =
            scenario % 2 == 0 ? dd::BasisStates::zero : dd::BasisStates::right;
        if (scenario >= 2) {
          state[0] = dd::BasisStates::plus;
          state[k / 2] = dd::BasisStates::right;
          state[k - 1] = dd::BasisStates::plus;
        }
        if (scenario >= 4) {
          state[k / 2 + 1] = dd::BasisStates::zero;
        }
        auto expected = package->applyOperation(
            reference,
            llvm::cantFail(dd::makeBasisState(k + 1, state, *package)));
        auto actual =
            simulate(function,
                     llvm::cantFail(dd::makeBasisState(k + 1, state, *package)),
                     *package, rng);
        ASSERT_TRUE(succeeded(actual));
        expectStatesNear(*package, *actual, expected);
        package->decRef(*actual);
        package->decRef(expected);
      }
      package->decRef(reference);
    }
  }
}

TEST_F(MultiControlledDecompositionTest,
       AncillaWideBorrowingAndDirtyRestoration) {
  for (size_t k : {23U, 24U}) {
    SCOPED_TRACE(k);
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          SmallVector<Value> reg;
          for (size_t i = 0; i < k + 2; ++i) {
            reg.push_back(b.staticQubit(i));
          }
          auto ancilla = b.h(reg[k + 1]);
          auto [control, target] = b.cx(ancilla, reg[k / 2]);
          reg[k / 2] = target;
          b.mcx(ValueRange(reg).take_front(k), reg[k]);
          b.rz(0.37, control);
          return SmallVector<Value>{};
        });
    auto function = *moduleOp->getOps<func::FuncOp>().begin();
    auto package = llvm::cantFail(dd::Package::create(k + 2));
    std::vector<dd::BasisStates> state(k + 2, dd::BasisStates::one);
    state[k] = dd::BasisStates::right;
    std::mt19937_64 beforeRng(7), afterRng(7);
    auto expected = simulate(
        function, llvm::cantFail(dd::makeBasisState(k + 2, state, *package)),
        *package, beforeRng);
    ASSERT_TRUE(succeeded(expected));
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    auto actual = simulate(
        function, llvm::cantFail(dd::makeBasisState(k + 2, state, *package)),
        *package, afterRng);
    ASSERT_TRUE(succeeded(actual));
    expectStatesNear(*package, *actual, *expected);
    package->decRef(*actual);
    package->decRef(*expected);
    EXPECT_EQ(countEffectiveCxOps(*moduleOp), 1 + 16 * k - 8);
  }
}

TEST_F(MultiControlledDecompositionTest, AncillaFreeSwapReorderedWires) {
  // A SWAP with k-1 controls uses a central MCX with k controls.
  for (size_t k = 2; k <= 9; ++k) {
    SCOPED_TRACE(k);
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& b) {
          auto target0 = b.staticQubit(0);
          auto target1 = b.staticQubit(k);
          SmallVector<Value> controls;
          for (size_t i = k - 1; i > 0; --i) {
            controls.push_back(b.staticQubit(i));
          }
          std::ignore = b.mcswap(controls, target1, target0);
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    auto function = *moduleOp->getOps<func::FuncOp>().begin();
    expectFullyLowered(*moduleOp);
    EXPECT_EQ(countStaticQubits(function), k + 1);
    function.walk([](AllocOp) { ADD_FAILURE() << "unexpected ancilla qubit"; });
    EXPECT_EQ(countEffectiveCxOps(*moduleOp),
              (k <= 8 ? expectedMcxCx(k) : expectedMcpCx(k)) + 2);
    auto package = llvm::cantFail(dd::Package::create(k + 1));
    auto actual = buildFunctionality(function, *package);
    ASSERT_TRUE(succeeded(actual));
    dd::Controls controls;
    for (size_t i = 1; i < k; ++i) {
      controls.emplace(static_cast<dd::Qubit>(i));
    }
    auto expected = llvm::cantFail(
        makeGateDD(*package, DynamicMatrix{SWAPOp::getUnitaryMatrix()},
                   {static_cast<dd::Qubit>(k), 0}, controls));
    // Full operator comparison, without discarding a global phase.
    EXPECT_EQ(*actual, expected);
    package->decRef(*actual);
  }
}

TEST_F(MultiControlledDecompositionTest, ControlledSwapRespectsMinQubits) {
  for (const size_t numControls : {1U, 2U}) {
    for (const uint64_t minQubits : {3U, 4U}) {
      SCOPED_TRACE(testing::Message() << "controls=" << numControls
                                      << " min-qubits=" << minQubits);
      auto moduleOp =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& builder) {
            auto c0 = builder.staticQubit(0);
            if (numControls == 1) {
              std::ignore = builder.cswap(c0, builder.staticQubit(1),
                                          builder.staticQubit(2));
            } else {
              std::ignore = builder.mcswap({c0, builder.staticQubit(1)},
                                           builder.staticQubit(2),
                                           builder.staticQubit(3));
            }
            return SmallVector<Value>{};
          });
      ASSERT_TRUE(moduleOp);
      DecomposeMultiControlledOptions options;
      options.minQubits = minQubits;
      ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp, options)));
      auto funcOp = *moduleOp->getOps<func::FuncOp>().begin();
      ASSERT_EQ(countStaticQubits(funcOp), numControls + 2);
      if (numControls + 2 < minQubits) {
        size_t controlledSwaps = 0;
        moduleOp->walk([&](CtrlOp op) {
          auto inner =
              mlir::mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
          if (op.getNumControls() == numControls && op.getNumTargets() == 2 &&
              inner && isa<SWAPOp>(inner.getOperation())) {
            ++controlledSwaps;
          }
        });
        EXPECT_EQ(controlledSwaps, 1U);
      } else {
        expectFullyLowered(*moduleOp);
        expectFullyDecomposed(funcOp);
      }

      auto package = llvm::cantFail(dd::Package::create(numControls + 2));
      const auto actual = buildFunctionality(funcOp, *package);
      ASSERT_TRUE(succeeded(actual));
      const auto expected = llvm::cantFail(
          makeGateDD(*package, DynamicMatrix{SWAPOp::getUnitaryMatrix()},
                     {
                         static_cast<dd::Qubit>(numControls),
                         static_cast<dd::Qubit>(numControls + 1),
                     },
                     makeControls(numControls)));
      EXPECT_EQ(*actual, expected);
      package->decRef(*actual);
    }
  }
}

TEST_F(MultiControlledDecompositionTest, LeavesUnsupportedCtrlUntouched) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.mch({builder.staticQubit(0), builder.staticQubit(1)},
                    builder.staticQubit(2));
        // Two-target non-SWAP body: passes min-qubits but is not lowered.
        std::ignore =
            builder.cdcx(builder.staticQubit(3), builder.staticQubit(4),
                         builder.staticQubit(5));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 1U);

  size_t mchCount = 0;
  size_t controlledDcx = 0;
  moduleOp->walk([&](CtrlOp op) {
    auto inner =
        mlir::mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (op.getNumTargets() == 2) {
      if (inner && isa<DCXOp>(inner.getOperation())) {
        ++controlledDcx;
      }
    }
    if (op.getNumControls() < 2) {
      return;
    }
    if (inner && isa<HOp>(inner.getOperation())) {
      ++mchCount;
    }
  });
  EXPECT_EQ(mchCount, 1U);
  EXPECT_EQ(controlledDcx, 1U);
}

TEST_F(MultiControlledDecompositionTest,
       KeepsAdjacentUnsupportedControlsSeparate) {
  for (const size_t numControls : {1U, 2U}) {
    SCOPED_TRACE(numControls);
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& builder) {
          SmallVector<Value> controls{builder.staticQubit(0)};
          if (numControls == 2) {
            controls.push_back(builder.staticQubit(1));
          }
          auto target = builder.staticQubit(numControls);
          for (size_t i = 0; i < 2; ++i) {
            auto [nextControls, nextTarget] =
                builder.ctrl(controls, target,
                             [&](Value qubit) { return builder.h(qubit); });
            controls.assign(nextControls.begin(), nextControls.end());
            target = nextTarget;
          }
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    size_t count = 0;
    moduleOp->walk([&](CtrlOp op) {
      EXPECT_EQ(op.getNumControls(), numControls);
      EXPECT_EQ(op.getNumBodyUnitaries(), 1U);
      EXPECT_TRUE(isa<HOp>(op.getBodyUnitary(0).getOperation()));
      ++count;
    });
    EXPECT_EQ(count, 2U);
  }
}

TEST_F(MultiControlledDecompositionTest, CompositeControlsRespectMinQubits) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.ctrl({builder.staticQubit(0), builder.staticQubit(1)},
                     builder.staticQubit(2), [&](Value targetArg) -> Value {
                       return builder.y(builder.x(targetArg));
                     });
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  auto original = OwningOpRef<ModuleOp>(moduleOp->clone());
  DecomposeMultiControlledOptions options;
  options.minQubits = 4;
  ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp, options)));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      *moduleOp, *original, OperationEquivalence::Flags::None));

  options.minQubits = 3;
  ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp, options)));
  expectFullyLowered(*moduleOp);
}

TEST_F(MultiControlledDecompositionTest,
       NestedModifiersRespectWidthNativeTargetAndControlScope) {
  for (const auto [controlled, native, minQubits] :
       {std::tuple{false, false, 3U}, {true, false, 4U}, {true, true, 3U}}) {
    for (const bool power : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "controlled=" << controlled << ", native=" << native
                   << ", power=" << power);
      auto moduleOp =
          QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& builder) {
            auto q0 = builder.staticQubit(0);
            auto q1 = builder.staticQubit(1);
            auto q2 = builder.staticQubit(2);
            const auto modifier = [&](ValueRange args) -> SmallVector<Value> {
              const auto body = [&](ValueRange targets) -> SmallVector<Value> {
                return {
                    builder.rx(0.37, targets[0]),
                    builder.ry(0.61, targets[1]),
                };
              };
              return power ? builder.pow(2.0, args, body)
                           : builder.inv(args, body);
            };
            if (controlled) {
              builder.ctrl(ValueRange{q0}, ValueRange{q1, q2}, modifier);
            } else {
              modifier(ValueRange{q1, q2});
            }
            return SmallVector<Value>{};
          });
      DecomposeMultiControlledOptions options;
      options.minQubits = minQubits;
      if (native) {
        const auto target = llvm::cantFail(CompilerTarget::create(
            3, CompilerTarget::Connectivity::allToAll(),
            CompilerTarget::NativeOperations::unrestricted()));
        PassManager pm(context());
        pm.addPass(createDecomposeMultiControlled(target));
        ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
      } else {
        ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp, options)));
      }
      size_t modifiers = 0;
      moduleOp->walk([&](Operation* op) {
        if (isa<InvOp, PowOp>(op)) {
          ++modifiers;
          EXPECT_EQ(mlir::mqt::getNumBodyUnitaries<UnitaryOpInterface>(
                        op->getRegion(0).front()),
                    2U);
        }
      });
      EXPECT_EQ(modifiers, 1U);
    }
  }
}

TEST_F(MultiControlledDecompositionTest, CompositePowerLimits) {
  for (const auto [overlap, exponent, runtime] : {
           std::tuple{true, 2.0, false},
           {false, 0.5, false},
           {false, 0.0, true},
       }) {
    SCOPED_TRACE(testing::Message() << "overlap=" << overlap << ", exponent="
                                    << exponent << ", runtime=" << runtime);
    Value parameter;
    auto moduleOp =
        QCOProgramBuilder::build(context(), [&](QCOProgramBuilder& builder) {
          parameter = builder.floatConstant(exponent);
          builder.ctrl(
              ValueRange{builder.staticQubit(0)},
              ValueRange{builder.staticQubit(1), builder.staticQubit(2)},
              [&](ValueRange args) -> SmallVector<Value> {
                return builder.pow(
                    parameter, args,
                    [&](ValueRange targets) -> SmallVector<Value> {
                      auto a = builder.rx(0.37, targets[0]);
                      if (overlap) {
                        a = builder.ry(0.61, a);
                      }
                      return {a, builder.rz(0.29, targets[1])};
                    });
              });
          return SmallVector<Value>{};
        });
    if (runtime) {
      auto function = *moduleOp->getOps<func::FuncOp>().begin();
      ASSERT_TRUE(succeeded(function.insertArgument(
          0, Float64Type::get(context()), {}, function.getLoc())));
      parameter.replaceAllUsesWith(function.getArgument(0));
      auto* constant = parameter.getDefiningOp();
      parameter = function.getArgument(0);
      constant->erase();
    }

    ASSERT_TRUE(succeeded(runDecomposeMultiControlled(*moduleOp)));
    SmallVector<PowOp> powers;
    moduleOp->walk([&](PowOp op) { powers.push_back(op); });
    ASSERT_EQ(powers.size(), 1U);
    EXPECT_EQ(powers.front().getNumBodyUnitaries(), overlap ? 3U : 2U);
    if (runtime) {
      EXPECT_EQ(powers.front().getExponent(), parameter);
    } else {
      EXPECT_EQ(powers.front().getExponentValue(), exponent);
    }
  }
}

TEST_F(MultiControlledDecompositionTest, PhasePiRoutesThroughMcz) {
  for (const double theta : {std::numbers::pi, -std::numbers::pi}) {
    for (const size_t k : {2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U}) {
      auto moduleOp = buildMcpModule(context(), k, theta);
      ASSERT_TRUE(moduleOp) << "k=" << k << " theta=" << theta;
      ASSERT_TRUE(runDecomposeMultiControlled(moduleOp.get()).succeeded())
          << "k=" << k << " theta=" << theta;
      expectFullyLowered(moduleOp.get());
      // ±π must take the Z path, so CX counts match MCZ/MCX.
      EXPECT_EQ(countSingleControlledGates<XOp>(moduleOp.get()),
                expectedMcxCx(k))
          << "k=" << k << " theta=" << theta;
      auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
      expectFullyDecomposed(funcOp);
      expectImplementsControlledGate(funcOp, k, phaseMatrix(theta), k);
    }
  }
}
