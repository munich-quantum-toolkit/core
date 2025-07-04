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
#include "mlir/Conversion/QuakeToMQTDyn/QuakeToMQTDyn.h"
#include "mlir/Dialect/MQTOpt/IR/MQTDynDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <llvm/Config/llvm-config.h>
#include <llvm/Support/Compiler.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>
#include <mlir/Tools/Plugins/PassPlugin.h>

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MQTDyn", LLVM_VERSION_STRING,
          [](DialectRegistry* registry) {
            registry->insert<::mqt::ir::dyn::MQTDynDialect>();
            mqt::ir::opt::registerMQTDynPasses();
            mqt::ir::conversions::registerMQTDynToQuake();
            mqt::ir::conversions::registerQuakeToMQTDyn();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MQTDynPasses", LLVM_VERSION_STRING, []() {
            mqt::ir::opt::registerMQTDynPasses();
            mqt::ir::conversions::registerMQTDynToQuake();
            mqt::ir::conversions::registerQuakeToMQTDyn();
          }};
}
