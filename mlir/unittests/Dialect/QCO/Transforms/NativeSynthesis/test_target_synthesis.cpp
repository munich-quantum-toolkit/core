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
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
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
using OperationLocus = Target::OperationLocus;
using Site = Target::Site;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using mlir::Value;
using mlir::qco::CtrlOp;
using mlir::qco::QCOProgramBuilder;
using mlir::qco::RXXOp;
using mlir::qco::SWAPOp;

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
  const auto expectedFunction = mainFunction(*expected);
  const auto actualFunction = mainFunction(*actual);
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

[[nodiscard]] static Target makeUCxTarget(std::vector<Site> sites = std::vector{
                                              Site{0}, Site{1}}) {
  return Target{std::move(sites), std::nullopt,
                std::vector{Operation{"u", 1, 3}, Operation{"cx", 2, 0}}};
}

namespace {

class TargetSynthesisTest : public testing::Test {
protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::qco::QCODialect, mlir::qtensor::QTensorDialect,
                    mlir::scf::SCFDialect>();
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
  const Target target{2};
  auto optimization = mlir::qco::createOptimizeTwoQubitUnitaryRuns();
  auto synthesis = mlir::qco::createTargetNativeSynthesis(target);
  auto conformance = mlir::qco::createVerifyTargetConformance(target);

  ASSERT_NE(optimization, nullptr);
  ASSERT_NE(synthesis, nullptr);
  ASSERT_NE(conformance, nullptr);

  mlir::DialectRegistry optimizationDialects;
  optimization->getDependentDialects(optimizationDialects);
  EXPECT_TRUE(optimizationDialects.getDialectAllocator(
      mlir::arith::ArithDialect::getDialectNamespace()));

  mlir::DialectRegistry synthesisDialects;
  synthesis->getDependentDialects(synthesisDialects);
  EXPECT_TRUE(synthesisDialects.getDialectAllocator(
      mlir::arith::ArithDialect::getDialectNamespace()));
}

TEST_F(TargetSynthesisTest, PreRoutingOptimizationRequiresStrictImprovement) {
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
      runPass(*optimized, mlir::qco::createOptimizeTwoQubitUnitaryRuns())));
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
      runPass(*nonImproving, mlir::qco::createOptimizeTwoQubitUnitaryRuns())));
  EXPECT_EQ(countOps<CtrlOp>(*nonImproving), 3U);
}

TEST_F(TargetSynthesisTest, PreRoutingOptimizationLeavesIndividualOpsAlone) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.swap(q0, q1);
    return builder.intConstant(0);
  });
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createOptimizeTwoQubitUnitaryRuns())));
  EXPECT_EQ(countOps<SWAPOp>(*module), 1U);
  EXPECT_EQ(countOps<CtrlOp>(*module), 0U);
}

TEST_F(TargetSynthesisTest,
       PreRoutingOptimizationLeavesRuntimeParameterizedRunsAlone) {
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
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createOptimizeTwoQubitUnitaryRuns())));
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

TEST_F(TargetSynthesisTest, TargetNativeSynthesisPreservesNativeSwap) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.swap(q0, q1);
    return builder.intConstant(0);
  });
  const Target swapTarget{2, std::nullopt,
                          std::vector{Operation{"swap", 2, 0}}};
  ASSERT_FALSE(swapTarget.synthesisBasis());
  const auto before = printModule(*module);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createTargetNativeSynthesis(swapTarget))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createVerifyTargetConformance(swapTarget))));
  EXPECT_EQ(countOps<SWAPOp>(*module), 1U);
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest,
       TargetNativeSynthesisUsesSupportedSymmetricOrientation) {
  const auto swap = [](QCOProgramBuilder& builder) {
    auto q0 = builder.staticQubit(0);
    auto q1 = builder.staticQubit(1);
    std::tie(q0, q1) = builder.swap(q0, q1);
    return builder.intConstant(0);
  };
  auto expected = build(swap);
  auto synthesized = build(swap);
  const Target reverseOnly{
      std::vector{Site{0}, Site{1}}, std::nullopt,
      std::vector{Operation{"u", 1, 3},
                  Operation{"cz", 2, 0, std::vector{OperationLocus{{1, 0}}}}}};
  ASSERT_TRUE(reverseOnly.synthesisBasis());
  ASSERT_EQ(reverseOnly.synthesisBasis()->entangler, Target::GateKind::CZ);

  ASSERT_TRUE(mlir::succeeded(runPass(
      *synthesized, mlir::qco::createTargetNativeSynthesis(reverseOnly))));
  EXPECT_EQ(countOps<SWAPOp>(*synthesized), 0U);
  EXPECT_GT(countOps<CtrlOp>(*synthesized), 0U);
  ASSERT_TRUE(mlir::succeeded(runPass(
      *synthesized, mlir::qco::createVerifyTargetConformance(reverseOnly))));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*synthesized)));
  expectEquivalent(expected, synthesized);
}

TEST_F(TargetSynthesisTest, AbsentOperationSetTreatsEveryOperationAsNative) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    qubit = builder.h(qubit);
    return builder.intConstant(0);
  });
  const Target permissive{1};
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
  const Target powOnly{1, std::nullopt, std::vector{Operation{"pow", 1, 1}}};
  ASSERT_FALSE(powOnly.synthesisBasis());
  const auto before = printModule(*module);

  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createTargetNativeSynthesis(powOnly))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createVerifyTargetConformance(powOnly))));
  EXPECT_EQ(printModule(*module), before);
}

TEST_F(TargetSynthesisTest, MissingBasisIsDiagnosedOnlyWhenLoweringIsNeeded) {
  const Target hOnly{1, std::nullopt, std::vector{Operation{"h", 1, 0}}};
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
                             "'qco.x' at ordered provider locus [0]"),
            std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("no globally usable synthesis basis"),
            std::string::npos)
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
  const Target target{
      2, std::nullopt,
      std::vector{Operation{"u", 1, 3}, Operation{"rxx", 2, 1}}};
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
                             "'qco.rxx' at ordered provider locus [0, 1]"),
            std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("unitary matrix is not available at compile time"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TargetSynthesisTest, ConformanceUsesProviderIdsAndOrderedLoci) {
  const Target directional{
      std::vector{Site{10}, Site{20}}, std::nullopt,
      std::vector{
          Operation{"cx", 2, 0, std::vector{OperationLocus{{10, 20}}}}}};
  ASSERT_FALSE(directional.synthesisBasis());

  auto supported = build([](QCOProgramBuilder& builder) {
    auto q10 = builder.staticQubit(10);
    auto q20 = builder.staticQubit(20);
    std::tie(q10, q20) = builder.cx(q10, q20);
    return builder.intConstant(0);
  });
  ASSERT_TRUE(mlir::succeeded(runPass(
      *supported, mlir::qco::createTargetNativeSynthesis(directional))));
  ASSERT_TRUE(mlir::succeeded(runPass(
      *supported, mlir::qco::createVerifyTargetConformance(directional))));

  auto reversed = build([](QCOProgramBuilder& builder) {
    auto q10 = builder.staticQubit(10);
    auto q20 = builder.staticQubit(20);
    std::tie(q20, q10) = builder.cx(q20, q10);
    return builder.intConstant(0);
  });
  const auto diagnostics = expectFailure(
      *reversed, mlir::qco::createVerifyTargetConformance(directional));
  EXPECT_NE(diagnostics.find("'qco.ctrl' with arity 2 and 0 parameter(s)"),
            std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("[20, 10]"), std::string::npos) << diagnostics;
}

TEST_F(TargetSynthesisTest, ConformanceChecksTypeArityParametersAndSite) {
  const auto expectUnsupported = [&](const Target& target,
                                     OwningOpRef<ModuleOp> module,
                                     const std::string& operation,
                                     const std::string& details,
                                     const std::string& locus) {
    const auto diagnostics = expectFailure(
        *module, mlir::qco::createVerifyTargetConformance(target));
    EXPECT_NE(diagnostics.find(operation), std::string::npos) << diagnostics;
    EXPECT_NE(diagnostics.find(details), std::string::npos) << diagnostics;
    EXPECT_NE(diagnostics.find(locus), std::string::npos) << diagnostics;
  };

  expectUnsupported(Target{std::vector{Site{10}}, std::nullopt,
                           std::vector{Operation{"x", 1, 0}}},
                    build([](QCOProgramBuilder& builder) {
                      auto qubit = builder.staticQubit(10);
                      qubit = builder.h(qubit);
                      return builder.intConstant(0);
                    }),
                    "'qco.h'", "arity 1 and 0 parameter(s)", "[10]");

  expectUnsupported(Target{std::vector{Site{10}, Site{20}}, std::nullopt,
                           std::vector{Operation{"x", 2, 0}}},
                    build([](QCOProgramBuilder& builder) {
                      auto qubit = builder.staticQubit(10);
                      qubit = builder.x(qubit);
                      return builder.intConstant(0);
                    }),
                    "'qco.x'", "arity 1 and 0 parameter(s)", "[10]");

  expectUnsupported(Target{std::vector{Site{10}}, std::nullopt,
                           std::vector{Operation{"rz", 1, 0}}},
                    build([](QCOProgramBuilder& builder) {
                      auto qubit = builder.staticQubit(10);
                      qubit = builder.rz(0.25, qubit);
                      return builder.intConstant(0);
                    }),
                    "'qco.rz'", "arity 1 and 1 parameter(s)", "[10]");

  expectUnsupported(Target{std::vector{Site{10}}, std::nullopt,
                           std::vector{Operation{"x", 1, 0}}},
                    build([](QCOProgramBuilder& builder) {
                      auto qubit = builder.staticQubit(30);
                      qubit = builder.x(qubit);
                      return builder.intConstant(0);
                    }),
                    "'qco.x'", "arity 1 and 0 parameter(s)", "[30]");
}

TEST_F(TargetSynthesisTest, ConformanceChecksNonUnitaryCapabilities) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(0);
    auto [measured, result] = builder.measure(qubit);
    static_cast<void>(result);
    measured = builder.reset(measured);
    return builder.intConstant(0);
  });
  const Target xOnly{1, std::nullopt, std::vector{Operation{"x", 1, 0}}};
  const auto diagnostics =
      expectFailure(*module, mlir::qco::createVerifyTargetConformance(xOnly));
  EXPECT_NE(diagnostics.find("'qco.measure' with arity 1 and 0 parameter(s)"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TargetSynthesisTest, ConformanceTracesStructuredControlFlow) {
  auto module = build([](QCOProgramBuilder& builder) {
    auto qubit = builder.staticQubit(10);
    qubit = builder.qcoIf(
        true, qubit, [&](Value argument) { return builder.h(argument); },
        [&](Value argument) { return builder.h(argument); });
    qubit =
        builder.scfFor(0, 1, 1, qubit, [&](Value, mlir::ValueRange arguments) {
          return mlir::SmallVector<Value>{builder.x(arguments[0])};
        })[0];
    qubit = builder.h(qubit);
    return builder.intConstant(0);
  });
  const Target target{std::vector{Site{10}}, std::nullopt,
                      std::vector{Operation{"h", 1, 0}, Operation{"x", 1, 0}}};

  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createTargetNativeSynthesis(target))));
  ASSERT_TRUE(mlir::succeeded(
      runPass(*module, mlir::qco::createVerifyTargetConformance(target))));
}

} // namespace mqt::test::qco
