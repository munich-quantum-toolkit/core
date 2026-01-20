/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

class QCOQuaternionMergeTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder;
  OwningOpRef<ModuleOp> module;

  using RotationGate =
      std::pair<std::function<Value(const std::vector<double>&, Value)>,
                std::vector<double>>;

  std::function<Value(const std::vector<double>&, Value)> rx;
  std::function<Value(const std::vector<double>&, Value)> ry;
  std::function<Value(const std::vector<double>&, Value)> rz;
  std::function<Value(const std::vector<double>&, Value)> u;

  QCOQuaternionMergeTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    // context.loadDialect<math::MathDialect>();

    // Setup Builder
    builder.initialize();

    rx = [this](const std::vector<double>& angles, Value q) {
      return builder.rx(angles[0], q);
    };
    ry = [this](const std::vector<double>& angles, Value q) {
      return builder.ry(angles[0], q);
    };
    rz = [this](const std::vector<double>& angles, Value q) {
      return builder.rz(angles[0], q);
    };
    u = [this](const std::vector<double>& angles, Value q) {
      return builder.u(angles[0], angles[1], angles[2], q);
    };
  }

  // Counts the ammont of operations the current module/circuit contains
  template <typename OpTy> int countOps() {
    int count = 0;
    module->walk([&](OpTy) { ++count; });
    return count;
  }

  // TODO: create docstring
  // testGateMerge takes a list of Rotation gates and uses the builder api to
  // build a small quantum circuit, where a qubit is feed through all rotations
  // in the list.
  LogicalResult testGateMerge(const std::vector<RotationGate>& rotations) {

    auto q = builder.allocQubitRegister(1);

    Value qubit = q[0];

    for (const auto& [gate, angles] : rotations) {
      qubit = gate(angles, qubit);
    }

    module = builder.finalize();
    return runMergePass(module.get());
  }

  LogicalResult runMergePass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(qco::createMergeRotationGates());
    return pm.run(module);
  }
};
