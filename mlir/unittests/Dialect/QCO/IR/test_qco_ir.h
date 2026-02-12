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

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include <gtest/gtest.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <string>

struct QCOTestCase {
  std::string name;
  llvm::function_ref<void(mlir::qco::QCOProgramBuilder&)> programBuilder;
  llvm::function_ref<void(mlir::qco::QCOProgramBuilder&)> referenceBuilder;
};

class QCOTest : public testing::TestWithParam<QCOTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> program;
  mlir::OwningOpRef<mlir::ModuleOp> reference;

  void SetUp() override;
};

std::string printTestName(const testing::TestParamInfo<QCOTestCase>& info);

inline void emptyQCO([[maybe_unused]] mlir::qco::QCOProgramBuilder& builder) {}
