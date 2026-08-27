/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mqt::test::qco {

using Target = mlir::CompilerTarget;
using Operation = Target::Operation;
using Site = Target::Site;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using mlir::Value;
using mlir::ValueRange;
using mlir::qco::CtrlOp;
using mlir::qco::HOp;
using mlir::qco::QCOProgramBuilder;
using mlir::qco::RXXOp;
using mlir::qco::RYOp;
using mlir::qco::RZOp;
using mlir::qco::SWAPOp;
using mlir::qco::UnitaryOp;
using mlir::qco::UOp;
using mlir::qco::XOp;
using mlir::qco::ZOp;

template <class T> [[nodiscard]] static T valid(llvm::Expected<T> value) {
  return llvm::cantFail(std::move(value));
}

[[nodiscard]] static mlir::func::FuncOp mainFunction(ModuleOp module) {
  return *module.getBody()->getOps<mlir::func::FuncOp>().begin();
}

[[nodiscard]] static size_t countStaticQubits(mlir::func::FuncOp function) {
  size_t numQubits = 0;
  for (auto staticOp : function.getOps<mlir::qco::StaticOp>()) {
    numQubits =
        std::max(numQubits, static_cast<size_t>(staticOp.getIndex()) + 1);
  }
  return numQubits;
}

[[nodiscard]] static mlir::qco::DynamicMatrix
matrixFromDD(const dd::CMat& matrix) {
  const auto dimension = static_cast<int64_t>(matrix.size());
  mlir::qco::DynamicMatrix result(dimension);
  for (int64_t row = 0; row < dimension; ++row) {
    for (int64_t column = 0; column < dimension; ++column) {
      result(row, column) =
          matrix[static_cast<size_t>(row)][static_cast<size_t>(column)];
    }
  }
  return result;
}

static void expectEquivalent(const OwningOpRef<ModuleOp>& expected,
                             const OwningOpRef<ModuleOp>& actual) {
  auto expectedFunction = mainFunction(*expected);
  auto actualFunction = mainFunction(*actual);
  const auto numQubits = countStaticQubits(expectedFunction);
  ASSERT_EQ(numQubits, countStaticQubits(actualFunction));
  ASSERT_GT(numQubits, 0U);

  auto package = std::make_unique<dd::Package>(numQubits);
  const auto expectedUnitary =
      mlir::qco::buildFunctionality(expectedFunction, *package);
  ASSERT_TRUE(mlir::succeeded(expectedUnitary));
  const auto actualUnitary =
      mlir::qco::buildFunctionality(actualFunction, *package);
  ASSERT_TRUE(mlir::succeeded(actualUnitary));

  const auto expectedMatrix =
      matrixFromDD(expectedUnitary->getMatrix(numQubits));
  const auto actualMatrix = matrixFromDD(actualUnitary->getMatrix(numQubits));
  package->decRef(*expectedUnitary);
  package->decRef(*actualUnitary);
  EXPECT_TRUE(expectedMatrix.isApprox(actualMatrix));
}

template <class Op> [[nodiscard]] static size_t countOps(ModuleOp module) {
  size_t count = 0;
  module.walk([&](Op) { ++count; });
  return count;
}

[[nodiscard]] static std::string printModule(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

[[nodiscard]] static mlir::LogicalResult
runPass(ModuleOp module, std::unique_ptr<mlir::Pass> pass) {
  mlir::PassManager manager(module.getContext());
  manager.addPass(std::move(pass));
  return manager.run(module);
}

[[nodiscard]] static Target
makeUCxTarget(std::optional<std::vector<Site>> sites = std::nullopt) {
  if (!sites) {
    sites = std::vector{valid(Site::create(0)), valid(Site::create(1))};
  }
  std::vector operations{valid(Operation::create("u", 1, 3)),
                         valid(Operation::create("cx", 2, 0))};
  return valid(
      Target::create(std::move(*sites), std::nullopt, std::move(operations)));
}

[[nodiscard]] static mlir::DenseElementsAttr
denseMatrix(QCOProgramBuilder& builder, const int64_t dimension,
            const llvm::ArrayRef<std::complex<double>> values) {
  const auto type = mlir::RankedTensorType::get(
      {dimension, dimension}, mlir::ComplexType::get(builder.getF64Type()));
  return mlir::DenseElementsAttr::get(type, values);
}

constexpr std::array<std::complex<double>, 4> X_MATRIX{{
    {0.0, 0.0},
    {1.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
}};

// CX with operand 0 as the most-significant (control) qubit.
constexpr std::array<std::complex<double>, 16> CX_MATRIX{{
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
}};

namespace {

class TargetSynthesisTest : public testing::Test {
protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::qco::QCODialect, mlir::qtensor::QTensorDialect>();
    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp>
  build(const mlir::function_ref<Value(QCOProgramBuilder&)>& builder) const {
    return QCOProgramBuilder::build(context.get(), builder);
  }

  [[nodiscard]] std::string
  expectFailure(ModuleOp module, std::unique_ptr<mlir::Pass> pass) const {
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler handler(context.get(),
                                          [&](mlir::Diagnostic& diagnostic) {
                                            diagnostics += diagnostic.str();
                                            diagnostics += '\n';
                                            return mlir::success();
                                          });
    EXPECT_TRUE(mlir::failed(runPass(module, std::move(pass))));
    return diagnostics;
  }

  std::unique_ptr<mlir::MLIRContext> context;
};

} // namespace

TEST(TargetSynthesisPassContract, FactoriesAreIndependentlyConstructible) {
  const auto target = valid(Target::create(2));
  auto fusion = mlir::qco::createFuseTwoQubitGates();
  auto synthesis = mlir::qco::createTargetNativeSynthesis(target);
  auto conformance = mlir::qco::createVerifyTargetConformance(target);

  ASSERT_NE(fusion, nullptr);
  ASSERT_NE(synthesis, nullptr);
  ASSERT_NE(conformance, nullptr);

  mlir::DialectRegistry fusionDialects;
  fusion->getDependentDialects(fusionDialects);
  EXPECT_TRUE(fusionDialects.getDialectAllocator(
      mlir::arith::ArithDialect::getDialectNamespace()));

  mlir::DialectRegistry synthesisDialects;
  synthesis->getDependentDialects(synthesisDialects);
  EXPECT_TRUE(synthesisDialects.getDialectAllocator(
      mlir::arith::ArithDialect::getDialectNamespace()));
}

TEST_F(TargetSynthesisTest, TwoQubitGateFusionRequiresStrictImprovement) {
  const auto adjacentCx = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.cx(q0, q1);
    std::tie(q0, q1) = builder.cx(q0, q1);
    return builder.intConstant(0);
  };
  auto expected = build(adjacentCx);
  auto optimized = build(adjacentCx);
  ASSERT_TRUE(mlir::succeeded(
      runPass(*optimized, mlir::qco::createFuseTwoQubitGates())));
  EXPECT_EQ(countOps<CtrlOp>(*optimized), 0U);
  expectEquivalent(expected, optimized);

  auto nonImproving = build([](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.cx(q0, q1);
    std::tie(q1, q0) = builder.cx(q1, q0);
    std::tie(q0, q1) = builder.cx(q0, q1);
    return builder.intConstant(0);
  });
  ASSERT_TRUE(mlir::succeeded(
      runPass(*nonImproving, mlir::qco::createFuseTwoQubitGates())));
  EXPECT_EQ(countOps<CtrlOp>(*nonImproving), 3U);
}

TEST_F(TargetSynthesisTest,
       TwoQubitGateFusionFusesInterleavedSingleQubitGates) {
  const auto interleaved = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.cx(q0, q1);
    q1 = builder.x(q1);
    q0 = builder.z(q0);
    std::tie(q0, q1) = builder.cx(q0, q1);
    return builder.intConstant(0);
  };
  auto expected = build(interleaved);
  auto optimized = build(interleaved);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*optimized, mlir::qco::createFuseTwoQubitGates())));
  EXPECT_EQ(countOps<CtrlOp>(*optimized), 0U);
  expectEquivalent(expected, optimized);
}

TEST_F(TargetSynthesisTest, TwoQubitGateFusionEmitsSymmetricEntangler) {
  const auto reducible = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.cx(q0, q1);
    std::tie(q1, q0) = builder.cx(q1, q0);
    std::tie(q1, q0) = builder.cx(q1, q0);
    return builder.intConstant(0);
  };
  auto expected = build(reducible);
  auto optimized = build(reducible);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*optimized, mlir::qco::createFuseTwoQubitGates())));
  EXPECT_EQ(countOps<CtrlOp>(*optimized), 1U);
  EXPECT_EQ(countOps<ZOp>(*optimized), 1U);
  EXPECT_EQ(countOps<XOp>(*optimized), 0U);
  expectEquivalent(expected, optimized);
}

TEST_F(TargetSynthesisTest, TwoQubitGateFusionLeavesIndividualOpsAlone) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.swap(q0, q1);
    return builder.intConstant(0);
  });
  const auto before = printModule(*module);
  ASSERT_TRUE(
      mlir::succeeded(runPass(*module, mlir::qco::createFuseTwoQubitGates())));
  EXPECT_EQ(countOps<SWAPOp>(*module), 1U);
  EXPECT_EQ(countOps<CtrlOp>(*module), 0U);
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest,
       TwoQubitGateFusionLeavesRuntimeParameterizedRunsAlone) {
  auto module = mlir::parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) -> (!qco.qubit, !qco.qubit) {
        %q0 = qco.static 0 : !qco.qubit
        %q1 = qco.static 1 : !qco.qubit
        %q2, %q3 = qco.rxx(%theta) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        %q4, %q5 = qco.rxx(%theta) %q2, %q3 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        return %q4, %q5 : !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(module);
  const auto before = printModule(*module);
  ASSERT_TRUE(
      mlir::succeeded(runPass(*module, mlir::qco::createFuseTwoQubitGates())));
  EXPECT_EQ(countOps<RXXOp>(*module), 2U);
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest, TargetNativeSynthesisRemovesOrdinarySwap) {
  const auto swap = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.swap(q0, q1);
    return builder.intConstant(0);
  };
  auto expected = build(swap);
  auto synthesized = build(swap);
  const auto target = makeUCxTarget();

  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesized, mlir::qco::createTargetNativeSynthesis(target))));
  EXPECT_EQ(countOps<SWAPOp>(*synthesized), 0U);
  EXPECT_GT(countOps<CtrlOp>(*synthesized), 0U);
  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesized, mlir::qco::createVerifyTargetConformance(target))));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*synthesized)));
  expectEquivalent(expected, synthesized);
}

TEST_F(TargetSynthesisTest,
       TargetNativeSynthesisLowersConstantSingleQubitGate) {
  const auto hadamard = [](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    qubit = builder.h(qubit);
    return builder.intConstant(0);
  };
  auto expected = build(hadamard);
  auto synthesized = build(hadamard);
  const auto target = makeUCxTarget();

  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesized, mlir::qco::createTargetNativeSynthesis(target))));
  EXPECT_EQ(countOps<HOp>(*synthesized), 0U);
  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesized, mlir::qco::createVerifyTargetConformance(target))));
  expectEquivalent(expected, synthesized);
}

TEST_F(TargetSynthesisTest,
       TargetNativeSynthesisLowersRuntimeParameterizedSingleQubitGates) {
  auto moduleOp = mlir::parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) -> !qco.qubit {
        %q0 = qco.static 0 : !qco.qubit
        %q1 = qco.rz(%theta) %q0 : !qco.qubit -> !qco.qubit
        %q2 = qco.ry(%theta) %q1 : !qco.qubit -> !qco.qubit
        return %q2 : !qco.qubit
      }
    }
  )mlir",
                                                    context.get());
  ASSERT_TRUE(moduleOp);
  const auto target = makeUCxTarget();

  ASSERT_TRUE(mlir::succeeded(
      runPass(*moduleOp, mlir::qco::createTargetNativeSynthesis(target))));
  EXPECT_EQ(countOps<RZOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<RYOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<UOp>(*moduleOp), 2U);
  EXPECT_EQ(countOps<mlir::math::SinOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<mlir::math::CosOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<mlir::math::AbsFOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<mlir::math::FloorOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<mlir::math::AcosOp>(*moduleOp), 0U);
  EXPECT_EQ(countOps<mlir::math::Atan2Op>(*moduleOp), 0U);
  ASSERT_TRUE(mlir::succeeded(
      runPass(*moduleOp, mlir::qco::createVerifyTargetConformance(target))));
}

TEST_F(TargetSynthesisTest, DenseUnitaryHasAsymmetricTwoQubitDDSemantics) {
  const auto denseCx = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    builder.unitary(ValueRange{q0, q1}, denseMatrix(builder, 4, CX_MATRIX));
    return builder.intConstant(0);
  };
  const auto cxReference = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.cx(q0, q1);
    return builder.intConstant(0);
  };
  auto expected = build(cxReference);
  auto actual = build(denseCx);

  ASSERT_TRUE(mlir::succeeded(mlir::verify(*actual)));
  expectEquivalent(expected, actual);
}

TEST_F(TargetSynthesisTest,
       TargetNativeSynthesisLowersDenseOneAndTwoQubitMatrices) {
  const auto denseX = [](QCOProgramBuilder& builder) {
    const auto qubit = builder.staticQubit(0);
    builder.unitary(ValueRange{qubit}, denseMatrix(builder, 2, X_MATRIX));
    return builder.intConstant(0);
  };
  const auto xReference = [](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    qubit = builder.x(qubit);
    return builder.intConstant(0);
  };
  auto expectedX = build(xReference);
  auto synthesizedX = build(denseX);
  const auto target = makeUCxTarget();

  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesizedX, mlir::qco::createTargetNativeSynthesis(target))));
  EXPECT_EQ(countOps<UnitaryOp>(*synthesizedX), 0U);
  ASSERT_TRUE(mlir::succeeded(runPass(
      *synthesizedX, mlir::qco::createVerifyTargetConformance(target))));
  expectEquivalent(expectedX, synthesizedX);

  const auto denseCx = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    builder.unitary(ValueRange{q0, q1}, denseMatrix(builder, 4, CX_MATRIX));
    return builder.intConstant(0);
  };
  const auto cxReference = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.cx(q0, q1);
    return builder.intConstant(0);
  };
  auto expectedCx = build(cxReference);
  auto synthesizedCx = build(denseCx);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesizedCx, mlir::qco::createTargetNativeSynthesis(target))));
  EXPECT_EQ(countOps<UnitaryOp>(*synthesizedCx), 0U);
  ASSERT_TRUE(mlir::succeeded(runPass(
      *synthesizedCx, mlir::qco::createVerifyTargetConformance(target))));
  expectEquivalent(expectedCx, synthesizedCx);
}

TEST_F(TargetSynthesisTest, TargetNativeSynthesisPreservesNativeSwap) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.swap(q0, q1);
    return builder.intConstant(0);
  });
  const auto swapTarget = valid(Target::create(
      2, std::nullopt, std::vector{valid(Operation::create("swap", 2, 0))}));
  ASSERT_FALSE(swapTarget.synthesisBasis());
  const auto before = printModule(*module);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createTargetNativeSynthesis(swapTarget))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createVerifyTargetConformance(swapTarget))));
  EXPECT_EQ(countOps<SWAPOp>(*module), 1U);
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest, TargetNativeSynthesisUsesHomogeneousCapability) {
  const auto swap = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.swap(q0, q1);
    return builder.intConstant(0);
  };
  auto expected = build(swap);
  auto synthesized = build(swap);
  const auto target =
      valid(Target::create(2, std::nullopt,
                           std::vector{valid(Operation::create("u", 1, 3)),
                                       valid(Operation::create("cz", 2, 0))}));
  ASSERT_TRUE(target.synthesisBasis());
  ASSERT_EQ(target.synthesisBasis()->entangler, Target::GateKind::CZ);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesized, mlir::qco::createTargetNativeSynthesis(target))));
  EXPECT_EQ(countOps<SWAPOp>(*synthesized), 0U);
  EXPECT_GT(countOps<CtrlOp>(*synthesized), 0U);
  ASSERT_TRUE(mlir::succeeded(
      runPass(*synthesized, mlir::qco::createVerifyTargetConformance(target))));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*synthesized)));
  expectEquivalent(expected, synthesized);
}

TEST_F(TargetSynthesisTest, AbsentOperationSetTreatsEveryOperationAsNative) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    qubit = builder.h(qubit);
    return builder.intConstant(0);
  });
  const auto permissive = valid(Target::create(1));
  const auto before = printModule(*module);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createTargetNativeSynthesis(permissive))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createVerifyTargetConformance(permissive))));
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest, NativePowShellHidesItsImplementationBody) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    qubit = builder.pow(2.0, qubit,
                        [&](Value argument) { return builder.h(argument); });
    return builder.intConstant(0);
  });
  const auto powOnly = valid(Target::create(
      1, std::nullopt, std::vector{valid(Operation::create("pow", 1, 1))}));
  ASSERT_FALSE(powOnly.synthesisBasis());
  const auto before = printModule(*module);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createTargetNativeSynthesis(powOnly))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createVerifyTargetConformance(powOnly))));
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest, MissingBasisIsDiagnosedOnlyWhenLoweringIsNeeded) {
  const auto hOnly = valid(Target::create(
      1, std::nullopt, std::vector{valid(Operation::create("h", 1, 0))}));
  ASSERT_FALSE(hOnly.synthesisBasis());

  auto supported = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    qubit = builder.h(qubit);
    return builder.intConstant(0);
  });
  const auto before = printModule(*supported);
  ASSERT_TRUE(mlir::succeeded(
      runPass(*supported, mlir::qco::createTargetNativeSynthesis(hOnly))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*supported, mlir::qco::createVerifyTargetConformance(hOnly))));
  EXPECT_EQ(printModule(*supported), before);

  auto unsupported = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    qubit = builder.x(qubit);
    return builder.intConstant(0);
  });
  const auto diagnostics = expectFailure(
      *unsupported, mlir::qco::createTargetNativeSynthesis(hOnly));
  EXPECT_NE(diagnostics.find("target-native synthesis cannot lower operation "
                             "'qco.x'"),
            std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("no usable synthesis basis"), std::string::npos)
      << diagnostics;
}

TEST_F(TargetSynthesisTest, SupportedRuntimeParameterizedGateStaysUntouched) {
  auto module = mlir::parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) -> (!qco.qubit, !qco.qubit) {
        %q0 = qco.static 0 : !qco.qubit
        %q1 = qco.static 1 : !qco.qubit
        %q2, %q3 = qco.rxx(%theta) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        return %q2, %q3 : !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(module);
  const auto target =
      valid(Target::create(2, std::nullopt,
                           std::vector{valid(Operation::create("u", 1, 3)),
                                       valid(Operation::create("rxx", 2, 1))}));
  const auto before = printModule(*module);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createTargetNativeSynthesis(target))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createVerifyTargetConformance(target))));
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest,
       UnsupportedRuntimeParameterizedGateHasLocalDiagnostic) {
  auto module = mlir::parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) -> (!qco.qubit, !qco.qubit) {
        %q0 = qco.static 0 : !qco.qubit
        %q1 = qco.static 1 : !qco.qubit
        %q2, %q3 = qco.rxx(%theta) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        return %q2, %q3 : !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(module);
  const auto diagnostics = expectFailure(
      *module, mlir::qco::createTargetNativeSynthesis(makeUCxTarget()));
  EXPECT_NE(diagnostics.find("target-native synthesis cannot lower operation "
                             "'qco.rxx'"),
            std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("unitary matrix is not available at compile time"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TargetSynthesisTest,
       UnsupportedRuntimeParameterizedGateDoesNotPartiallyRewrite) {
  auto module = mlir::parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%theta: f64) -> (!qco.qubit, !qco.qubit) {
        %q0 = qco.static 0 : !qco.qubit
        %q1 = qco.static 1 : !qco.qubit
        %q2 = qco.rz(%theta) %q0 : !qco.qubit -> !qco.qubit
        %q3, %q4 = qco.rxx(%theta) %q2, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        return %q3, %q4 : !qco.qubit, !qco.qubit
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(module);
  const auto before = printModule(*module);

  static_cast<void>(expectFailure(
      *module, mlir::qco::createTargetNativeSynthesis(makeUCxTarget())));
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest,
       ConformanceUsesHomogeneousCapabilitiesAndValidatesSites) {
  const auto target = valid(Target::create(
      std::vector{valid(Site::create(10)), valid(Site::create(20))},
      std::nullopt, std::vector{valid(Operation::create("cx", 2, 0))}));
  ASSERT_FALSE(target.synthesisBasis());

  auto reversed = build([](QCOProgramBuilder& builder) {
    auto q10 = builder.staticQubit(10);
    auto q20 = builder.staticQubit(20);
    std::tie(q20, q10) = builder.cx(q20, q10);
    return builder.intConstant(0);
  });
  ASSERT_TRUE(mlir::succeeded(
      runPass(*reversed, mlir::qco::createTargetNativeSynthesis(target))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*reversed, mlir::qco::createVerifyTargetConformance(target))));

  auto unknownSite = build([](QCOProgramBuilder& builder) {
    auto q30 = builder.staticQubit(30);
    auto q20 = builder.staticQubit(20);
    std::tie(q30, q20) = builder.cx(q30, q20);
    return builder.intConstant(0);
  });
  const auto diagnostics = expectFailure(
      *unknownSite, mlir::qco::createVerifyTargetConformance(target));
  EXPECT_NE(diagnostics.find("target does not contain static site 30"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TargetSynthesisTest, ConformanceRejectsDynamicAllocations) {
  const auto target = valid(Target::create(
      1, std::nullopt, std::vector{valid(Operation::create("x", 1, 0))}));
  const auto expectDynamicAllocationFailure =
      [&](OwningOpRef<ModuleOp> module) {
        const auto diagnostics = expectFailure(
            *module, mlir::qco::createVerifyTargetConformance(target));
        EXPECT_NE(
            diagnostics.find("requires qubits to be assigned to qco.static"),
            std::string::npos)
            << diagnostics;
      };

  expectDynamicAllocationFailure(build([](QCOProgramBuilder& builder) {
    auto qubit = builder.allocQubit();
    qubit = builder.x(qubit);
    return builder.intConstant(0);
  }));
  expectDynamicAllocationFailure(build([](QCOProgramBuilder& builder) {
    auto qubits = builder.allocQubitRegister(1);
    qubits[0] = builder.x(qubits[0]);
    return builder.intConstant(0);
  }));
}

TEST_F(TargetSynthesisTest, ConformanceRejectsQuantumFunctionInputs) {
  auto module = mlir::parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main(%qubit: !qco.qubit) -> !qco.qubit {
        %result = qco.x %qubit : !qco.qubit -> !qco.qubit
        return %result : !qco.qubit
      }
    }
  )mlir",
                                                  context.get());
  ASSERT_TRUE(module);
  const auto target = valid(Target::create(
      1, std::nullopt, std::vector{valid(Operation::create("x", 1, 0))}));

  const auto diagnostics =
      expectFailure(*module, mlir::qco::createVerifyTargetConformance(target));
  EXPECT_NE(diagnostics.find("requires quantum function inputs to be assigned "
                             "to qco.static target sites"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TargetSynthesisTest, ConformanceChecksTypeArityAndParameters) {
  const auto expectUnsupported = [&](const Target& target,
                                     OwningOpRef<ModuleOp> module,
                                     const std::string& operation,
                                     const std::string& details) {
    const auto diagnostics = expectFailure(
        *module, mlir::qco::createVerifyTargetConformance(target));
    EXPECT_NE(diagnostics.find(operation), std::string::npos) << diagnostics;
    EXPECT_NE(diagnostics.find(details), std::string::npos) << diagnostics;
  };

  expectUnsupported(
      valid(Target::create(std::vector{valid(Site::create(10))}, std::nullopt,
                           std::vector{valid(Operation::create("x", 1, 0))})),
      build([](QCOProgramBuilder& builder) {
        auto qubit = builder.staticQubit(10);
        qubit = builder.h(qubit);
        return builder.intConstant(0);
      }),
      "'qco.h'", "arity 1 and 0 parameter(s)");

  expectUnsupported(
      valid(Target::create(
          std::vector{valid(Site::create(10)), valid(Site::create(20))},
          std::nullopt, std::vector{valid(Operation::create("x", 2, 0))})),
      build([](QCOProgramBuilder& builder) {
        auto qubit = builder.staticQubit(10);
        qubit = builder.x(qubit);
        return builder.intConstant(0);
      }),
      "'qco.x'", "arity 1 and 0 parameter(s)");

  expectUnsupported(
      valid(Target::create(std::vector{valid(Site::create(10))}, std::nullopt,
                           std::vector{valid(Operation::create("rz", 1, 0))})),
      build([](QCOProgramBuilder& builder) {
        auto qubit = builder.staticQubit(10);
        qubit = builder.rz(0.25, qubit);
        return builder.intConstant(0);
      }),
      "'qco.rz'", "arity 1 and 1 parameter(s)");
}

TEST_F(TargetSynthesisTest, ConformanceChecksNonUnitaryCapabilities) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    auto [measured, result] = builder.measure(qubit);
    static_cast<void>(result);
    measured = builder.reset(measured);
    return builder.intConstant(0);
  });
  const auto xOnly = valid(Target::create(
      1, std::nullopt, std::vector{valid(Operation::create("x", 1, 0))}));
  const auto diagnostics =
      expectFailure(*module, mlir::qco::createVerifyTargetConformance(xOnly));
  EXPECT_NE(diagnostics.find("'qco.measure' with arity 1 and 0 parameter(s)"),
            std::string::npos)
      << diagnostics;
}

} // namespace mqt::test::qco
