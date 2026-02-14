/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "TestCaseUtils.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <gtest/gtest.h>
#include <iosfwd>
#include <memory>
#include <mlir/IR/MLIRContext.h>
#include <string>

struct QCTestCase {
  std::string name;
  mqt::test::NamedBuilder<mlir::qc::QCProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<mlir::qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QCTestCase& info);
};

class QCTest : public testing::TestWithParam<QCTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;

  void SetUp() override;
};
