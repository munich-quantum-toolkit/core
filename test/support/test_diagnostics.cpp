/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "support/Diagnostics.hpp"

#include "gtest/gtest.h"

#include "mlir/Support/LogicalResult.h"

#include <thread>
#include <tuple>
#include <vector>

TEST(Diagnostics, NestedHandlersAndMetadata) {
  std::vector<int> order;
  mqt::ScopedDiagnosticHandler const outer(
      [&](const mqt::Diagnostic& diagnostic) {
        order.push_back(1);
        EXPECT_EQ(diagnostic.message, "original");
        EXPECT_EQ(diagnostic.category, mqt::ErrorCategory::IO);
        EXPECT_EQ(diagnostic.severity, mqt::DiagnosticSeverity::Warning);
        EXPECT_EQ(diagnostic.status, 31415);
        return mlir::success();
      });
  {
    mqt::ScopedDiagnosticHandler const inner([&](const mqt::Diagnostic&) {
      order.push_back(2);
      return mlir::failure();
    });
    mqt::emitDiagnostic({
        .message = "original",
        .category = mqt::ErrorCategory::IO,
        .severity = mqt::DiagnosticSeverity::Warning,
        .status = 31415,
    });
  }
  EXPECT_EQ(order, (std::vector<int>{2, 1}));
  {
    mqt::ScopedDiagnosticHandler const consume([&](const mqt::Diagnostic&) {
      order.push_back(3);
      return mlir::success();
    });
    std::ignore = mqt::emitError("consumed");
  }
  EXPECT_EQ(order, (std::vector<int>{2, 1, 3}));
}

TEST(Diagnostics, ThreadIsolationAndDefaultOutput) {
  int parent = 0;
  int worker = 0;
  mqt::ScopedDiagnosticHandler const handler([&](const mqt::Diagnostic&) {
    ++parent;
    return mlir::success();
  });
  testing::internal::CaptureStderr();
  std::thread thread([&] {
    std::ignore = mqt::emitError("unhandled on worker");
    mqt::ScopedDiagnosticHandler const installed([&](const mqt::Diagnostic&) {
      ++worker;
      return mlir::success();
    });
    std::ignore = mqt::emitError("handled on worker");
  });
  thread.join();
  EXPECT_EQ(testing::internal::GetCapturedStderr(),
            "[mqt-core] [error] unhandled on worker\n");
  EXPECT_EQ(parent, 0);
  EXPECT_EQ(worker, 1);
  std::ignore = mqt::emitError("parent");
  EXPECT_EQ(parent, 1);
}

TEST(Diagnostics, ReemissionStartsAtPreviousHandler) {
  std::string message;
  mqt::ScopedDiagnosticHandler const outer(
      [&](const mqt::Diagnostic& diagnostic) {
        message = diagnostic.message;
        return mlir::success();
      });
  mqt::ScopedDiagnosticHandler const inner(
      [&](const mqt::Diagnostic& diagnostic) {
        std::ignore = mqt::emitError("source: " + diagnostic.message);
        return mlir::success();
      });
  std::ignore = mqt::emitError("detail");
  EXPECT_EQ(message, "source: detail");
}
