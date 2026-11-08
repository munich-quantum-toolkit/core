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

#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "ExactUnitaryTest.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace mqt::test {
inline mlir::func::FuncOp twirlingMainFunction(mlir::ModuleOp moduleOp) {
  return *moduleOp.getOps<mlir::func::FuncOp>().begin();
}

struct TwirlingEntangler {
  std::string symbol;
  llvm::SmallVector<size_t> sites;
  mlir::qco::Matrix4x4 matrix;
};

struct TwirlingCheckpoint {
  std::vector<TwirlingEntangler> entanglers;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> segments;
};

// These fixtures have one static, straight-line function. Modifier bodies
// belong to their outer unitary; traversing them separately would count
// controls twice.
inline testing::AssertionResult
captureTwirlingCheckpoint(mlir::ModuleOp moduleOp,
                          TwirlingCheckpoint& checkpoint) {
  checkpoint = {};
  if (mlir::failed(mlir::verify(moduleOp)) ||
      mlir::failed(mlir::qco::verifyLinearity(moduleOp))) {
    return testing::AssertionFailure() << "invalid or nonlinear checkpoint";
  }
  auto functions = moduleOp.getOps<mlir::func::FuncOp>();
  if (std::distance(functions.begin(), functions.end()) != 1 ||
      !llvm::hasSingleElement(twirlingMainFunction(moduleOp).getBody())) {
    return testing::AssertionFailure() << "expected one single-block function";
  }
  llvm::DenseMap<mlir::Value, size_t> sites;
  for (auto& op : twirlingMainFunction(moduleOp).front()) {
    if (auto allocation = mlir::dyn_cast<mlir::qco::StaticOp>(op)) {
      sites[allocation.getQubit()] = allocation.getIndex();
      continue;
    }
    auto gate = mlir::dyn_cast<mlir::qco::UnitaryOpInterface>(op);
    if (!gate) {
      if (!mlir::isa<mlir::arith::ConstantOp, mlir::qco::SinkOp,
                     mlir::func::ReturnOp>(op)) {
        return testing::AssertionFailure()
               << "unsupported checkpoint operation: "
               << op.getName().getStringRef().str();
      }
      continue;
    }
    if (gate.getNumQubits() > 2 || !gate.hasCompileTimeKnownUnitaryMatrix()) {
      return testing::AssertionFailure()
             << "expected constant zero-to-two-qubit gates";
    }
    llvm::SmallVector<size_t> operands;
    for (auto input : gate.getInputQubits()) {
      const auto found = sites.find(input);
      if (found == sites.end()) {
        return testing::AssertionFailure() << "missing static wire identity";
      }
      operands.push_back(found->second);
    }
    for (size_t i = 0; i < operands.size(); ++i) {
      sites[gate.getOutputQubit(i)] = operands[i];
    }
    if (gate.isTwoQubit()) {
      mlir::qco::Matrix4x4 matrix;
      if (!gate.getUnitaryMatrix4x4(matrix)) {
        return testing::AssertionFailure() << "unknown entangler matrix";
      }
      checkpoint.entanglers.push_back(
          {gate.getBaseSymbol().str(), operands, matrix});
    }
  }
  if (sites.empty()) {
    return testing::AssertionFailure() << "expected static wires";
  }
  // Retain one local interval at a time, bypassing all other unitaries through
  // their positional wire interface. The existing DD oracle owns matrix math.
  for (size_t selected = 0; selected <= checkpoint.entanglers.size();
       ++selected) {
    mlir::OwningOpRef<mlir::ModuleOp> segment(moduleOp.clone());
    size_t interval = 0;
    for (auto& op :
         llvm::make_early_inc_range(twirlingMainFunction(*segment).front())) {
      auto gate = mlir::dyn_cast<mlir::qco::UnitaryOpInterface>(op);
      if (!gate) {
        continue;
      }
      const bool entangler = gate.isTwoQubit();
      if (entangler || interval != selected) {
        for (size_t i = 0; i < gate.getNumQubits(); ++i) {
          gate.getOutputQubit(i).replaceAllUsesWith(gate.getInputQubit(i));
        }
        op.erase();
      }
      if (entangler) {
        ++interval;
      }
    }
    if (mlir::failed(mlir::verify(*segment)) ||
        mlir::failed(mlir::qco::verifyLinearity(*segment))) {
      return testing::AssertionFailure() << "invalid extracted local segment";
    }
    checkpoint.segments.push_back(std::move(segment));
  }
  return testing::AssertionSuccess();
}

inline testing::AssertionResult
sameEntanglers(const TwirlingCheckpoint& expected,
               const TwirlingCheckpoint& actual) {
  if (expected.entanglers.size() != actual.entanglers.size()) {
    return testing::AssertionFailure() << "entangler count changed";
  }
  for (size_t i = 0; i < expected.entanglers.size(); ++i) {
    const auto& lhs = expected.entanglers[i];
    const auto& rhs = actual.entanglers[i];
    if (lhs.symbol != rhs.symbol || lhs.sites != rhs.sites ||
        !lhs.matrix.isApprox(rhs.matrix)) {
      return testing::AssertionFailure()
             << "entangler kind, ordered wires or matrix changed at " << i;
    }
  }
  return testing::AssertionSuccess();
}

inline void expectLocalSegmentsEquivalent(const TwirlingCheckpoint& expected,
                                          const TwirlingCheckpoint& actual) {
  ASSERT_TRUE(sameEntanglers(expected, actual));
  ASSERT_EQ(expected.segments.size(), actual.segments.size());
  for (size_t i = 0; i < expected.segments.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "local segment " << i);
    size_t numQubits = 0;
    for (auto site : twirlingMainFunction(*expected.segments[i])
                         .getOps<mlir::qco::StaticOp>()) {
      numQubits = std::max(numQubits, static_cast<size_t>(site.getIndex()) + 1);
    }
    expectFullUnitaryEqual(*expected.segments[i], *actual.segments[i],
                           numQubits, mlir::qco::MATRIX_TOLERANCE);
  }
}

} // namespace mqt::test
