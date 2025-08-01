/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTDynToMQTOpt/MQTDynToMQTOpt.h" // IWYU pragma: keep
#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h"       // IWYU pragma: keep
#include "mlir/Conversion/MQTOptToMQTDyn/MQTOptToMQTDyn.h" // IWYU pragma: keep
#include "mlir/Conversion/QIRToMQTDyn/QIRToMQTDyn.h"       // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MQTDyn/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h" // IWYU pragma: keep

#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(const int argc, char** argv) {
  mlir::registerAllPasses();
  mqt::ir::opt::registerMQTOptPasses();
  mqt::ir::dyn::registerMQTDynPasses();
  mqt::ir::registerMQTDynToMQTOptPasses();
  mqt::ir::registerMQTOptToMQTDynPasses();
  mqt::ir::registerMQTDynToQIRPasses();
  mqt::ir::registerQIRToMQTDynPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mqt::ir::opt::MQTOptDialect>();
  registry.insert<mqt::ir::dyn::MQTDynDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  return mlir::asMainReturnCode(
      MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
