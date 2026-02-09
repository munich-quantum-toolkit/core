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

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <gtest/gtest.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <string>

struct QCTestCase {
  std::string name;
  llvm::function_ref<void(mlir::qc::QCProgramBuilder&)> programBuilder;
  llvm::function_ref<void(mlir::qc::QCProgramBuilder&)> referenceBuilder;
};

class QCTest : public testing::TestWithParam<QCTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> program;
  mlir::OwningOpRef<mlir::ModuleOp> canonicalizedProgram;
  mlir::OwningOpRef<mlir::ModuleOp> reference;
  mlir::OwningOpRef<mlir::ModuleOp> canonicalizedReference;

  mlir::OwningOpRef<mlir::ModuleOp> emptyQC;

  void SetUp() override;

  void TearDown() override;
};

std::string printTestName(const testing::TestParamInfo<QCTestCase>& info);

inline void emptyQC([[maybe_unused]] mlir::qc::QCProgramBuilder& builder) {}
