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
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

#include <gtest/gtest.h>
#include <iosfwd>
#include <memory>
#include <mlir/IR/MLIRContext.h>
#include <string>

struct QIRTestCase {
  std::string name;
  mqt::test::NamedBuilder<mlir::qir::QIRProgramBuilder> programBuilder;
  mqt::test::NamedBuilder<mlir::qir::QIRProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QIRTestCase& info);
};

class QIRTest : public testing::TestWithParam<QIRTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;

  void SetUp() override;
};
