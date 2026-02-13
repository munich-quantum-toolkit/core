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
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>

namespace qc = mlir::qc;
namespace qir = mlir::qir;

struct QCToQIRTestCase {
  std::string name;
  llvm::function_ref<void(qc::QCProgramBuilder&)> programBuilder;
  llvm::function_ref<void(qir::QIRProgramBuilder&)> referenceBuilder;
};

class QCToQIRTest : public testing::TestWithParam<QCToQIRTestCase> {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> program;
  mlir::OwningOpRef<mlir::ModuleOp> reference;

  void SetUp() override;
};

std::string printTestName(const testing::TestParamInfo<QCToQIRTestCase>& info);

inline void emptyQC([[maybe_unused]] qc::QCProgramBuilder& builder) {}

inline void emptyQIR([[maybe_unused]] qir::QIRProgramBuilder& builder) {}
