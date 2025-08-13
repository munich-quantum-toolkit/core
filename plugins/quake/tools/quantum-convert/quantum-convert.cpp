/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTDynToQuake/MQTDynToQuake.h"
#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <cudaq/Optimizer/Dialect/Quake/QuakeDialect.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(const int argc, char** argv) {
  mlir::registerAllPasses();
  mqt::ir::conversions::registerMQTDynToQuakePasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mqt::ir::dyn::MQTDynDialect>();
  registry.insert<quake::QuakeDialect>();

  return mlir::asMainReturnCode(
      MlirOptMain(argc, argv, "Quantum conversion driver\n", registry));
}
