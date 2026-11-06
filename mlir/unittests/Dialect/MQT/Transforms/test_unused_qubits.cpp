/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "gtest/gtest.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

TEST(RemoveUnusedQubits, PreservesProgramInterfaceAndUsedAllocations) {
  MLIRContext context;
  context.loadDialect<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                      qtensor::QTensorDialect, arith::ArithDialect,
                      func::FuncDialect>();
  for (bool gateSet : {false, true}) {
    auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @qc() {
          %idle = qc.alloc : !qc.qubit
          qc.dealloc %idle : !qc.qubit
          %used = qc.alloc : !qc.qubit
          qc.x %used : !qc.qubit
          qc.dealloc %used : !qc.qubit
          return
        }
        func.func @qco() {
          %idle = qco.alloc : !qco.qubit
          qco.sink %idle : !qco.qubit
          %used = qco.alloc : !qco.qubit
          %out = qco.x %used : !qco.qubit
          qco.sink %out : !qco.qubit
          return
        }
        func.func @tensor() {
          %size = arith.constant 3 : index
          %idle = qtensor.alloc(%size) : tensor<3x!qco.qubit>
          qtensor.dealloc %idle : tensor<3x!qco.qubit>
          return
        }
      }
    )mlir",
                                                &context);
    ASSERT_TRUE(moduleOp);
    if (gateSet) {
      auto environment = parseAttribute(R"mlir(#mqt.target_env<
        compilation_target = #mqt.compilation_target<sites = [],
          connectivity = all_to_all, couplings = [],
          native_operations = unrestricted, operations = [], gate_set = unit>,
        payload_specification = #mqt.payload_spec<
          format = <id = "openqasm", version = "3.0", profile = "", encoding = text>,
          capabilities = [], optional_capabilities_known = false>>
      )mlir",
                                        &context);
      ASSERT_TRUE(environment);
      moduleOp->getOperation()->setAttr(mlir::mqt::TargetEnvAttr::name,
                                        environment);
    }
    PassManager manager(&context);
    manager.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
    unsigned beforeCleanup = 0;
    moduleOp->walk([&](Operation* op) {
      beforeCleanup += isa<qc::AllocOp, qco::AllocOp, qtensor::AllocOp>(op);
    });
    EXPECT_EQ(beforeCleanup, 5U);
    manager.clear();
    manager.addPass(mlir::mqt::createRemoveUnusedQubits());
    ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
    unsigned allocations = 0;
    moduleOp->walk([&](Operation* op) {
      allocations += isa<qc::AllocOp, qco::AllocOp, qtensor::AllocOp>(op);
    });
    EXPECT_EQ(allocations, gateSet ? 5U : 2U);
  }
}
