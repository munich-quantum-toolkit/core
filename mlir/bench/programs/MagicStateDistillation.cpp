/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/MagicStateDistillation.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"

#include "Programs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace mqt::bench {

using namespace mlir;

namespace {

// Litinski, arXiv:1905.06903v3, Fig. 3, in circuit order. Each word lists
// qubits from top to bottom; qubit 0 retains the distilled T†|+⟩ state.
constexpr std::array<std::string_view, 15> ROTATIONS{
    "IZIII", "IIZII", "IIIZI", "IIIIZ", "IZZZI", "ZZZII", "ZZIZI", "ZIZZI",
    "ZIIZZ", "ZZIIZ", "ZIZIZ", "ZZZZZ", "IIZZZ", "IZIZZ", "IZZIZ",
};

} // namespace

static Value distillMagicStates(qc::QCProgramBuilder& builder,
                                ValueRange workspace,
                                func::FuncOp precedingLevel) {
  auto qubits = workspace.take_front(5);
  for (auto qubit : qubits) {
    builder.reset(qubit);
    builder.h(qubit);
  }
  auto rejected = builder.boolConstant(false);
  for (const auto pauli : ROTATIONS) {
    SmallVector<Value> support;
    for (size_t i = 0; i < pauli.size(); ++i) {
      if (pauli[i] == 'Z') {
        support.push_back(qubits[i]);
      }
    }
    auto parityQubit = support.pop_back_val();
    for (auto control : support) {
      builder.cx(control, parityQubit);
    }
    // Computing parity, applying T, and uncomputing implements exp(-iπP/8)
    // up to a global phase. Higher levels consume a distilled state instead.
    if (!precedingLevel) {
      builder.t(parityQubit);
    } else {
      auto resourceWorkspace = workspace.drop_front(5);
      auto childRejected =
          builder.call(precedingLevel, resourceWorkspace).front();
      rejected = arith::OrIOp::create(builder, rejected, childRejected);
      auto resource = resourceWorkspace.front();
      builder.cx(parityQubit, resource);
      auto outcome = builder.measure(resource);
      // The resource is T†|+⟩, so outcome 0 needs S to turn T† into T;
      // outcome 1 already applies T (up to a global phase).
      auto needsCorrection =
          arith::XOrIOp::create(builder, outcome, builder.boolConstant(true));
      builder.scfIf(needsCorrection, [&] { builder.s(parityQubit); });
    }
    for (auto control : llvm::reverse(support)) {
      builder.cx(control, parityQubit);
    }
  }
  for (auto qubit : qubits.drop_front()) {
    builder.h(qubit);
    rejected = arith::OrIOp::create(builder, rejected, builder.measure(qubit));
  }
  return rejected;
}

SmallVector<Value>
magicStateDistillation(qc::QCProgramBuilder& builder,
                       const MagicStateDistillation& benchmark) {
  const auto levels = benchmark.options().levels;
  auto data =
      builder.allocQubitRegister(static_cast<int64_t>(5 * levels), "magic");
  auto result = builder.allocClassicalBitRegister(2, benchmark.output().name);
  func::FuncOp block;
  for (size_t level = 1; level <= levels; ++level) {
    block = builder.createFunction(
        "distill_15_to_1_level_" + std::to_string(level),
        SmallVector<Type>(5 * level, qc::QubitType::get(builder.getContext())),
        [&](ValueRange workspace) -> SmallVector<Value> {
          return {distillMagicStates(builder, workspace, block)};
        });
  }
  auto rejected = builder.call(block, data.qubits).front();
  auto root = data[0];
  builder.t(root);
  builder.h(root);
  builder.measure(root, result, 0);
  // Reuse the measured root to expose rejection without another qubit.
  builder.reset(root);
  builder.scfIf(rejected, [&] { builder.x(root); });
  builder.measure(root, result, 1);
  return {result};
}

} // namespace mqt::bench
