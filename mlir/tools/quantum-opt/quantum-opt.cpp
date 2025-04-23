/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// At the top of quantum-opt.cpp:
#ifdef ENABLE_CATALYST
#include "mlir/Conversion/Catalyst/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h" // IWYU pragma: keep
#include "mlir/Conversion/Catalyst/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h" // IWYU pragma: keep

#include <Quantum/IR/QuantumDialect.h>
#endif

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MQTDyn/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h" // IWYU pragma: keep

#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

void registerCatalystIfEnabled(mlir::DialectRegistry& registry) {
#ifdef ENABLE_CATALYST
  mlir::mqt::ir::conversions::registerCatalystQuantumToMQTOptPasses();
  mlir::mqt::ir::conversions::registerMQTOptToCatalystQuantumPasses();
  registry.insert<catalyst::quantum::QuantumDialect>();
#endif
}

int main(const int argc, char** argv) {
  mlir::registerAllPasses();
  mqt::ir::opt::registerMQTOptPasses();
  mqt::ir::dyn::registerMQTDynPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mqt::ir::opt::MQTOptDialect>();
  registry.insert<mqt::ir::dyn::MQTDynDialect>();

  registerCatalystIfEnabled(registry);

  return mlir::asMainReturnCode(
      MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
