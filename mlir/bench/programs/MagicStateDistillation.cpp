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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace mqt::bench {

using namespace mlir;

namespace {

/// Decoder for columns 1..15, logical row all ones, then the four coordinate
/// rows. Its inverse maps input bits 0..4 to those five generator rows and
/// bits 5..14 to the physical unit vectors at CORRECTION_QUBITS.
/// Fixed Gaussian elimination of that binary encoding matrix gives these CXs.
constexpr std::array<std::pair<size_t, size_t>, 52> DECODER{
    {
        {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7}, {0, 8},
        {0, 9},  {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {1, 0}, {1, 3},
        {1, 5},  {1, 7},  {1, 9},  {1, 11}, {1, 13}, {2, 0},  {2, 1}, {2, 3},
        {2, 6},  {2, 7},  {2, 10}, {2, 11}, {2, 14}, {3, 4},  {3, 5}, {3, 6},
        {3, 11}, {3, 12}, {3, 13}, {3, 14}, {4, 7},  {7, 4},  {4, 7}, {4, 8},
        {4, 9},  {4, 10}, {4, 11}, {4, 12}, {4, 13}, {4, 14}, {5, 7}, {7, 5},
        {5, 7},  {6, 7},  {7, 6},  {6, 7},
    },
};
constexpr std::array<size_t, 10> CORRECTION_QUBITS{
    4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
};

} /* namespace */

static Value distillMagicStates(qc::QCProgramBuilder& builder,
                                ValueRange qubits) {
  assert(qubits.size() == 15);
  /// Bravyi--Haah, arXiv:1209.2426, Appendix A: project raw magic states
  /// onto Z checks, correct with A(w), apply U, then postselect X checks.
  for (const auto& [control, target] : DECODER) {
    builder.cx(qubits[control], qubits[target]);
  }
  SmallVector<Value, 10> syndrome;
  for (size_t i = 5; i < 15; ++i) {
    syndrome.push_back(builder.measure(qubits[i]));
  }
  for (const auto& [control, target] : llvm::reverse(DECODER)) {
    builder.cx(qubits[control], qubits[target]);
  }
  for (size_t i = 0; i < syndrome.size(); ++i) {
    builder.scfIf(syndrome[i], [&] {
      /// A = T X T-dagger equals S X up to global phase and fixes |T>.
      auto qubit = qubits[CORRECTION_QUBITS[i]];
      builder.x(qubit);
      builder.s(qubit);
    });
  }
  /// The even rows have weight eight and the odd row has weight fifteen:
  /// transversal T-dagger implements logical T, hence U = S-dagger^tensor15.
  for (auto qubit : qubits) {
    builder.sdg(qubit);
  }
  for (const auto& [control, target] : DECODER) {
    builder.cx(qubits[control], qubits[target]);
  }
  auto rejected = builder.boolConstant(false);
  for (size_t i = 1; i < 5; ++i) {
    builder.h(qubits[i]);
    rejected =
        arith::OrIOp::create(builder, rejected, builder.measure(qubits[i]));
  }
  return rejected;
}

SmallVector<Value>
magicStateDistillation(qc::QCProgramBuilder& builder,
                       const MagicStateDistillation& benchmark) {
  int64_t size = 1;
  for (size_t level = 0; level < benchmark.options().levels; ++level) {
    size *= 15;
  }
  auto data = builder.allocQubitRegisterStorage(size, "magic");
  auto result = builder.allocClassicalBitRegister(2, benchmark.output().name);
  auto rejection = builder.allocClassicalBitRegister(1);
  builder.storeClassicalBit(builder.boolConstant(false), rejection, 0);
  auto block = builder.createFunction(
      "distill_15_to_1",
      SmallVector<Type>(15, qc::QubitType::get(builder.getContext())),
      [&](ValueRange qubits) -> SmallVector<Value> {
        return {distillMagicStates(builder, qubits)};
      });
  builder.scfFor(0, size, 1, [&](Value index) {
    auto qubit = builder.loadQubit(data, index);
    builder.h(qubit);
    builder.t(qubit);
  });
  for (int64_t stride = 1; stride < size; stride *= 15) {
    builder.scfFor(0, size, stride * 15, [&](Value first) {
      SmallVector<Value, 15> qubits;
      for (int64_t i = 0; i < 15; ++i) {
        auto index = arith::AddIOp::create(builder, first,
                                           builder.indexConstant(i * stride));
        qubits.push_back(builder.loadQubit(data, index));
      }
      auto rejected = builder.call(block, qubits).front();
      auto sticky = arith::OrIOp::create(
          builder, builder.loadClassicalBit(rejection, 0), rejected);
      builder.storeClassicalBit(sticky, rejection, 0);
    });
  }
  auto root = builder.loadQubit(data, builder.indexConstant(0));
  builder.tdg(root);
  builder.h(root);
  builder.measure(root, result, 0);
  /// QIR lacks writes of computed bits to result slots (qir-spec issue #65).
  /// Reuse the measured root to expose rejection without another qubit.
  builder.reset(root);
  builder.scfIf(builder.loadClassicalBit(rejection, 0),
                [&] { builder.x(root); });
  builder.measure(root, result, 1);
  return {result};
}

} /* namespace mqt::bench */
