/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Verification.h"
#include "mqt/Dialect/QCO/Utils/Verification.h"

#include "gtest/gtest.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#include <type_traits>
#include <utility>

namespace mlir {
namespace {
constexpr auto ROTATION = R"mlir(
module {
  func.func @main(%angle: f64) attributes {mqt.entry_point} {
    %q = qco.static 0 : !qco.qubit
    %r = qco.rx(%angle) %q : !qco.qubit
    qco.sink %r : !qco.qubit
    return
  }
})mlir";

static_assert(!std::is_convertible_v<qco::VerificationResult, bool>);

TEST(CompilerVerification, BorrowsProgramsAndBindsEntryArguments) {
  auto program = QCOProgram::fromMLIRString(ROTATION);
  ASSERT_TRUE(succeeded(program));
  const auto source = program->str();
  const auto type = Float64Type::get(program->context().get());
  const EntryArgumentBindings bindings{{0, FloatAttr::get(type, 0.3)}};
  auto result = verify(*program, *program, {}, bindings, bindings);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, qco::VerificationVerdict::Equivalent);
  EXPECT_EQ(result->leftBindings, result->rightBindings);
  EXPECT_EQ(result->leftBindings.size(), 1);
  EXPECT_EQ(program->str(), source);

  const EntryArgumentBindings changed{{0, FloatAttr::get(type, 0.7)}};
  result = verify(*program, *program, {}, bindings, changed);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, qco::VerificationVerdict::NotEquivalent);
}

TEST(CompilerVerification, RejectsInvalidIndicesContextsAndConsumedOwners) {
  auto program = QCOProgram::fromMLIRString(ROTATION);
  auto other = QCOProgram::fromMLIRString(ROTATION);
  ASSERT_TRUE(succeeded(program));
  ASSERT_TRUE(succeeded(other));
  const auto context = program->context();
  const ScopedDiagnosticHandler suppress(context.get(),
                                         [](Diagnostic&) { return success(); });
  const auto type = Float64Type::get(program->context().get());
  const EntryArgumentBindings invalidIndex{{1, FloatAttr::get(type, 0.)}};
  EXPECT_TRUE(failed(verify(*program, *program, {}, invalidIndex)));
  const auto otherType = Float64Type::get(other->context().get());
  const EntryArgumentBindings foreign{{0, FloatAttr::get(otherType, 0.)}};
  EXPECT_TRUE(failed(verify(*program, *program, {}, foreign)));
  auto moved = std::move(*program);
  ASSERT_FALSE(program->isValid());
  EXPECT_TRUE(failed(verify(*program, moved)));
}

TEST(CompilerVerification, RejectsProgramsWithoutMarkedEntries) {
  auto program = QCOProgram::fromMLIRString(R"mlir(
    module { func.func @helper() { return } }
  )mlir");
  ASSERT_TRUE(succeeded(program));
  const auto context = program->context();
  const ScopedDiagnosticHandler suppress(context.get(),
                                         [](Diagnostic&) { return success(); });
  EXPECT_TRUE(failed(verify(*program, *program)));
}
} // namespace
} // namespace mlir
