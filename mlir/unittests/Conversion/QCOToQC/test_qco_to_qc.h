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
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>

namespace qc = mlir::qc;
namespace qco = mlir::qco;

struct QCOToQCTestCase {
  std::string name;
  llvm::function_ref<void(qco::QCOProgramBuilder&)> programBuilder;
  llvm::function_ref<void(qc::QCProgramBuilder&)> referenceBuilder;
};

class QCOToQCTest : public testing::TestWithParam<QCOToQCTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> program;
  mlir::OwningOpRef<mlir::ModuleOp> reference;

  void SetUp() override;
};

std::string printTestName(const testing::TestParamInfo<QCOToQCTestCase>& info);
