/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Quantum/IR/QuantumDialect.h"
#include "mlir/Conversion/Catalyst/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h"
#include "mlir/Conversion/Catalyst/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h" // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(const int argc, char** argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);

  // registry.insert<mqt::ir::opt::MQTOptDialect>();
  // registry.insert<catalyst::quantum::QuantumDialect>();

  // mlir::mqt::ir::conversions::registerMQTOptToCatalystQuantum();
  // mlir::mqt::ir::conversions::registerCatalystQuantumToMQTOpt();

  return mlir::asMainReturnCode(
      MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
