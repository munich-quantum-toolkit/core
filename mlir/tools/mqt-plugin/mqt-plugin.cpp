/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

//===- mqt-plugin.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Quantum/IR/QuantumDialect.h"
#include "mlir/Conversion/Catalyst/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h"
#include "mlir/Conversion/Catalyst/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MQTOpt", LLVM_VERSION_STRING,
          [](DialectRegistry* registry) {
            registry->insert<::mqt::ir::opt::MQTOptDialect>();
            registry->insert<::catalyst::quantum::QuantumDialect>();

            //::mqt::ir::opt::registerMQTOptPasses();
            //::mlir::mqt::ir::conversions::registerMQTOptToQuantumPasses();
            //::mlir::mqt::ir::conversions::registerQuantumToMQTOptPasses();
          }};
}

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION, "MQTOptPasses", LLVM_VERSION_STRING, []() {
        ::mqt::ir::opt::registerMQTOptPasses();
        ::mlir::mqt::ir::conversions::registerMQTOptToCatalystQuantumPasses();
        ::mlir::mqt::ir::conversions::registerCatalystQuantumToMQTOptPasses();
      }};
}
