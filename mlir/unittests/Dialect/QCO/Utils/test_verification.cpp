/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/Verification.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FormatVariadic.h"

#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <future>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <stop_token>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

namespace {
using Verdict = VerificationVerdict;
using Method = VerificationMethod;
using Relation = VerificationRelation;
using Phase = VerificationPhase;

class NativeVerification : public ::testing::Test {
protected:
  MLIRContext context;
  ScopedDiagnosticHandler diagnostics{&context,
                                      [](Diagnostic&) { return success(); }};
  NativeVerification() {
    context.loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                        mqt::MQTDialect, QCODialect, qtensor::QTensorDialect,
                        cbit::CBitDialect>();
  }
  OwningOpRef<ModuleOp> parse(StringRef source) {
    return parseSourceString<ModuleOp>(source, &context);
  }
  static func::FuncOp entry(ModuleOp moduleOp) {
    return moduleOp.lookupSymbol<func::FuncOp>("main");
  }
  static VerificationOptions constructionOptions() {
    VerificationOptions options;
    options.method = Method::Construction;
    return options;
  }
  VerificationResult
  check(ModuleOp left, ModuleOp right,
        const VerificationOptions& options = constructionOptions(),
        const VerificationContract& contract = {}) {
    const auto result =
        qco::verify(entry(left), entry(right), contract, DDArgumentBindings(),
                    DDArgumentBindings(), options);
    EXPECT_TRUE(succeeded(result));
    return succeeded(result) ? *result
                             : VerificationResult{.reason = "invalid input"};
  }
  static std::string rotation(double angle) {
    std::ostringstream number;
    number << std::scientific << std::setprecision(17) << angle;
    return "module { func.func @main(%q: !qco.qubit) -> !qco.qubit { "
           "%a = arith.constant " +
           number.str() +
           " : f64 "
           "%r = qco.rz(%a) %q : !qco.qubit return %r : !qco.qubit } }";
  }
};

constexpr StringLiteral IDENTITY = R"(module {
  func.func @main(%q: !qco.qubit) -> !qco.qubit {
    return %q : !qco.qubit
  }
})";
constexpr StringLiteral X = R"(module {
  func.func @main(%q: !qco.qubit) -> !qco.qubit {
    %r = qco.x %q : !qco.qubit
    return %r : !qco.qubit
  }
})";
constexpr StringLiteral GLOBAL_PHASE = R"(module {
  func.func @main(%q: !qco.qubit) -> !qco.qubit {
    %p = arith.constant 0.37 : f64
    qco.gphase(%p)
    return %q : !qco.qubit
  }
})";

TEST_F(NativeVerification,
       ExactRelationsPreservePhaseAndEachCheckerFindsFaults) {
  auto identity = parse(IDENTITY);
  auto x = parse(X);
  auto phased = parse(GLOBAL_PHASE);
  ASSERT_TRUE(identity && x && phased);
  for (const auto method :
       {Method::Construction, Method::Alternating, Method::Auto}) {
    auto options = constructionOptions();
    options.method = method;
    EXPECT_EQ(check(*identity, *identity, options).verdict,
              Verdict::Equivalent);
    EXPECT_EQ(check(*identity, *x, options).verdict, Verdict::NotEquivalent);
    EXPECT_EQ(check(*identity, *phased, options).verdict,
              Verdict::NotEquivalent);
    options.phase = Phase::Ignore;
    EXPECT_EQ(check(*identity, *phased, options).verdict, Verdict::Equivalent);
    EXPECT_EQ(check(*identity, *x, options).verdict, Verdict::NotEquivalent);
  }
}

TEST_F(NativeVerification, SimulationPassIsNotAnAllInputProof) {
  auto identity = parse(IDENTITY);
  auto x = parse(X);
  ASSERT_TRUE(identity && x);
  auto options = constructionOptions();
  options.method = Method::Simulation;
  EXPECT_EQ(check(*identity, *identity, options).verdict,
            Verdict::Inconclusive);
  const auto mismatch = check(*identity, *x, options);
  EXPECT_EQ(mismatch.verdict, Verdict::NotEquivalent);
  ASSERT_EQ(mismatch.checkers.size(), 1);
  EXPECT_EQ(mismatch.checkers.front().samples, 1);
}

TEST_F(NativeVerification,
       ApproximationBoundaryUsesStableResidualAndSeparateGuard) {
  auto identity = parse(IDENTITY);
  ASSERT_TRUE(identity);
  auto options = constructionOptions();
  options.relation = Relation::ProcessDistance;
  options.phase = Phase::Ignore;
  options.tolerance = .01;
  options.numericalTolerance = 1e-9;
  for (const auto method :
       {Method::Construction, Method::Alternating, Method::Auto}) {
    options.method = method;
    for (const auto [ratio, expected] : std::array{
             std::pair{.999, Verdict::Equivalent},
             std::pair{1., Verdict::Inconclusive},
             std::pair{1.001, Verdict::NotEquivalent},
         }) {
      auto rotated = parse(rotation(2 * std::asin(options.tolerance * ratio)));
      ASSERT_TRUE(rotated);
      const auto result = check(*identity, *rotated, options);
      EXPECT_EQ(result.verdict, expected) << result.reason;
      if (result.distance) {
        EXPECT_NEAR(*result.distance, options.tolerance * ratio, 1e-12);
      }
    }
  }
  auto tiny = parse(rotation(1e-8));
  ASSERT_TRUE(tiny);
  options.method = Method::Construction;
  options.tolerance = 1e-8;
  options.numericalTolerance = 1e-12;
  const auto result = check(*identity, *tiny, options);
  ASSERT_TRUE(result.distance);
  EXPECT_NEAR(*result.distance, 5e-9, 1e-13);
  options.tolerance = 0;
  EXPECT_EQ(check(*identity, *identity, options).verdict,
            Verdict::Inconclusive);
  options.method = Method::Simulation;
  EXPECT_EQ(check(*identity, *tiny, options).verdict, Verdict::Unsupported);
}

TEST_F(NativeVerification, AmplitudeDistanceIncludesGlobalPhase) {
  auto identity = parse(IDENTITY);
  auto phased = parse(GLOBAL_PHASE);
  ASSERT_TRUE(identity && phased);
  auto options = constructionOptions();
  options.relation = Relation::AmplitudeDistance;
  options.tolerance = .4;
  const auto result = check(*identity, *phased, options);
  EXPECT_EQ(result.verdict, Verdict::Equivalent);
  ASSERT_TRUE(result.distance);
  EXPECT_NEAR(*result.distance, 2 * std::sin(.37 / 2), 1e-12);
}

TEST_F(NativeVerification, SmallAverageDistanceDoesNotBoundWorstInput) {
  // A sign on one of 256 basis states is close on average, but orthogonalizes
  // (|00000000> + |11111111>)/sqrt(2). This analytic oracle uses no DD code.
  std::string arguments, returns, types;
  for (size_t i = 0; i < 8; ++i) {
    if (i != 0) {
      arguments += ", ";
      returns += ", ";
      types += ", ";
    }
    arguments += "%q" + std::to_string(i) + ": !qco.qubit";
    returns += "%q" + std::to_string(i);
    types += "!qco.qubit";
  }
  auto source = "module { func.func @main(" + arguments + ") -> (" + types +
                ") { return " + returns + " : " + types + " } }";
  auto identity = parse(source);
  ASSERT_TRUE(identity);
  // Keep the source simple: use a diagonal unitary matrix for the rare phase.
  // The independent expected normalized process distance is
  // sqrt(1-(254/256)^2).
  const auto expected = std::sqrt(1 - std::pow(254. / 256, 2));
  EXPECT_LT(expected, .13);
  const std::array<std::complex<double>, 2> input{
      std::numbers::sqrt2 / 2,
      std::numbers::sqrt2 / 2,
  };
  const auto overlap =
      std::conj(input[0]) * input[0] - std::conj(input[1]) * input[1];
  EXPECT_DOUBLE_EQ(std::abs(overlap), 0);
  // The gate fixture below exercises the same effect with seven controls.
  const auto* const body = R"(
    %r:8 = qco.ctrl (%q0, %q1, %q2, %q3, %q4, %q5, %q6) targets(%t = %q7) {
      %z = qco.z %t : !qco.qubit
      qco.yield %z : !qco.qubit
    } : ({!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit}, {!qco.qubit})
    return %r#0, %r#1, %r#2, %r#3, %r#4, %r#5, %r#6, %r#7 : )";
  auto rare = parse("module { func.func @main(" + arguments + ") -> (" + types +
                    ") {" + body + types + " } }");
  ASSERT_TRUE(rare);
  auto options = constructionOptions();
  options.relation = Relation::ProcessDistance;
  options.phase = Phase::Ignore;
  options.tolerance = .13;
  const auto result = check(*identity, *rare, options);
  EXPECT_EQ(result.verdict, Verdict::Equivalent) << result.reason;
  ASSERT_TRUE(result.distance);
  EXPECT_NEAR(*result.distance, expected, 1e-12);
}

TEST_F(NativeVerification, InitializedInputsAreExplicitAndArgumentsStayFree) {
  auto left = parse(R"(module { func.func @main() {
    %q = qco.alloc : !qco.qubit
    %x = qco.x %q : !qco.qubit
    qco.sink %x : !qco.qubit
    return
  } })");
  auto right = parse(R"(module { func.func @main() {
    %q = qco.alloc : !qco.qubit
    %z = qco.z %q : !qco.qubit
    %x = qco.x %z : !qco.qubit
    qco.sink %x : !qco.qubit
    return
  } })");
  ASSERT_TRUE(left && right);
  EXPECT_EQ(check(*left, *right).verdict, Verdict::NotEquivalent);
  VerificationContract initialized;
  initialized.domain = VerificationDomain::InitializedProgram;
  EXPECT_EQ(check(*left, *right, constructionOptions(), initialized).verdict,
            Verdict::Equivalent);
  auto identity = parse(IDENTITY);
  auto x = parse(X);
  ASSERT_TRUE(identity && x);
  const auto result = check(*identity, *x, constructionOptions(), initialized);
  EXPECT_EQ(result.verdict, Verdict::NotEquivalent);
  EXPECT_EQ(result.numFreeInputs, 1);
}

TEST_F(NativeVerification, ConcreteBindingsAndLoopBoundsArePartOfTheClaim) {
  auto source = parse(R"(module {
    func.func @main(%q: !qco.qubit, %n: index) -> !qco.qubit {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %r = scf.for %i = %c0 to %n step %c1 iter_args(%v = %q) -> (!qco.qubit) {
        %x = qco.x %v : !qco.qubit
        scf.yield %x : !qco.qubit
      }
      return %r : !qco.qubit
    }
  })");
  auto identity = parse(IDENTITY);
  ASSERT_TRUE(source && identity);
  auto func = entry(*source);
  auto options = constructionOptions();
  DDArgumentBindings bindings{
      {func.getArgument(1), IntegerAttr::get(IndexType::get(&context), 2)},
  };
  auto result = qco::verify(func, entry(*identity), {}, bindings,
                            DDArgumentBindings(), options);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, Verdict::Equivalent);
  bindings[func.getArgument(1)] = IntegerAttr::get(IndexType::get(&context), 3);
  result = qco::verify(func, entry(*identity), {}, bindings,
                       DDArgumentBindings(), options);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, Verdict::NotEquivalent);
  EXPECT_EQ(check(*source, *identity).verdict, Verdict::Unsupported);
  options.limits.maxSteps = 4;
  result = qco::verify(func, entry(*identity), {}, bindings,
                       DDArgumentBindings(), options);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, Verdict::Inconclusive);
}

TEST_F(NativeVerification, TerminalEffectsDifferFromZeroStateProbabilities) {
  auto left = parse(R"(module { func.func @main() -> i1 {
    %q = qco.static 0 : !qco.qubit
    %h = qco.h %q : !qco.qubit
    %measured, %m = qco.measure %h : !qco.qubit
    qco.sink %measured : !qco.qubit
    return %m : i1
  } })");
  auto right = parse(R"(module { func.func @main() -> i1 {
    %q = qco.static 0 : !qco.qubit
    %z = qco.z %q : !qco.qubit
    %h = qco.h %z : !qco.qubit
    %measured, %m = qco.measure %h : !qco.qubit
    qco.sink %measured : !qco.qubit
    return %m : i1
  } })");
  ASSERT_TRUE(left && right);
  auto options = constructionOptions();
  options.relation = Relation::TerminalProbabilities;
  EXPECT_EQ(check(*left, *right, options).verdict, Verdict::NotEquivalent);
  options.limits.maxOutcomes = 1;
  EXPECT_EQ(check(*left, *right, options).verdict, Verdict::Inconclusive);
}

TEST_F(NativeVerification,
       TerminalEffectsPermitGarbageWithoutClaimingReducedStateEquality) {
  auto left = parse(R"(module { func.func @main() -> i1 {
    %data = qco.static 0 : !qco.qubit
    %aux = qco.static 1 : !qco.qubit
    qco.sink %aux : !qco.qubit
    %measured, %m = qco.measure %data : !qco.qubit
    qco.sink %measured : !qco.qubit
    return %m : i1
  } })");
  auto right = parse(R"(module { func.func @main() -> i1 {
    %data = qco.static 0 : !qco.qubit
    %aux = qco.static 1 : !qco.qubit
    %c, %t = qco.ctrl (%data) targets(%a = %aux) {
      %x = qco.x %a : !qco.qubit
      qco.yield %x : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    qco.sink %t : !qco.qubit
    %measured, %m = qco.measure %c : !qco.qubit
    qco.sink %measured : !qco.qubit
    return %m : i1
  } })");
  ASSERT_TRUE(left && right);
  auto options = constructionOptions();
  VerificationContract contract;
  contract.zeroAncillas = {1};
  EXPECT_EQ(check(*left, *right, options, contract).verdict,
            Verdict::NotEquivalent);
  options.relation = Relation::TerminalProbabilities;
  EXPECT_EQ(check(*left, *right, options, contract).verdict,
            Verdict::Equivalent);
  // For data |+>, discarded CNOT target dephases data: off-diagonal .5 -> 0.
  const auto reducedStateTraceDistance = .5;
  EXPECT_DOUBLE_EQ(reducedStateTraceDistance, .5);
}

TEST_F(NativeVerification, CancellationAndMemoryLimitsAreInconclusive) {
  auto identity = parse(IDENTITY);
  ASSERT_TRUE(identity);
  auto options = constructionOptions();
  options.limits.maxDDBytes = 1;
  const auto exhausted = check(*identity, *identity, options);
  EXPECT_EQ(exhausted.verdict, Verdict::Inconclusive);
  std::stop_source stop;
  stop.request_stop();
  const auto cancelled =
      qco::verify(entry(*identity), entry(*identity), {}, DDArgumentBindings(),
                  DDArgumentBindings(), {}, stop.get_token());
  ASSERT_TRUE(succeeded(cancelled));
  EXPECT_EQ(cancelled->verdict, Verdict::Inconclusive);
}

TEST_F(NativeVerification, TimeoutOptionsDoNotOverflowTheClock) {
  auto identity = parse(IDENTITY);
  ASSERT_TRUE(identity);
  auto options = constructionOptions();
  options.limits.timeout = std::chrono::milliseconds::max();
  EXPECT_EQ(check(*identity, *identity, options).verdict, Verdict::Equivalent);
  options.limits.timeout = std::chrono::milliseconds{-1};
  EXPECT_TRUE(
      failed(qco::verify(entry(*identity), entry(*identity), {},
                         DDArgumentBindings(), DDArgumentBindings(), options)));
}

TEST_F(NativeVerification,
       WorkerBudgetsAndConcurrentCallsDoNotChangeNumericalSettings) {
  auto identity = parse(IDENTITY);
  auto phased = parse(GLOBAL_PHASE);
  ASSERT_TRUE(identity && phased);
  for (size_t workers = 1; workers <= 3; ++workers) {
    VerificationOptions options;
    options.limits.workers = workers;
    const auto result = check(*identity, *phased, options);
    EXPECT_EQ(result.verdict, Verdict::NotEquivalent);
    EXPECT_GT(result.peakDDBytes, 0);
  }
  auto first = std::async(std::launch::async,
                          [&] { return check(*identity, *identity); });
  auto second =
      std::async(std::launch::async, [&] { return check(*identity, *phased); });
  const auto a = first.get();
  const auto b = second.get();
  EXPECT_EQ(a.verdict, Verdict::Equivalent);
  EXPECT_EQ(b.verdict, Verdict::NotEquivalent);
  EXPECT_EQ(a.ddTolerance, b.ddTolerance);
}
} // namespace
