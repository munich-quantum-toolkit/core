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

#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

#include <gtest/gtest.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <string>

struct QIRTestCase {
  std::string name;
  llvm::function_ref<void(mlir::qir::QIRProgramBuilder&)> programBuilder;
  llvm::function_ref<void(mlir::qir::QIRProgramBuilder&)> referenceBuilder;
};

class QIRTest : public testing::TestWithParam<QIRTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> program;
  mlir::OwningOpRef<mlir::ModuleOp> reference;

  void SetUp() override;
};

std::string printTestName(const testing::TestParamInfo<QIRTestCase>& info);

inline void emptyQIR([[maybe_unused]] mlir::qir::QIRProgramBuilder& builder) {}
