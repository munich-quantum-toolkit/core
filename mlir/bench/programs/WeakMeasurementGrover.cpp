/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/WeakMeasurementGrover.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mqt::bench {

using namespace mlir;

static void maskMarkedState(qc::QCProgramBuilder& builder,
                            llvm::ArrayRef<Value> search,
                            std::string_view markedBitstring) {
  for (size_t index = 0; index < search.size(); ++index) {
    if (markedBitstring[search.size() - 1 - index] == '0') {
      builder.x(search[index]);
    }
  }
}

static void markPhase(qc::QCProgramBuilder& builder,
                      llvm::ArrayRef<Value> search,
                      std::string_view markedBitstring) {
  maskMarkedState(builder, search, markedBitstring);
  builder.mcz(search.drop_back(), search.back());
  maskMarkedState(builder, search, markedBitstring);
}

static void groverIteration(qc::QCProgramBuilder& builder,
                            llvm::ArrayRef<Value> search,
                            std::string_view markedBitstring) {
  markPhase(builder, search, markedBitstring);
  for (auto qubit : search) {
    builder.h(qubit);
    builder.x(qubit);
  }
  builder.mcz(search.drop_back(), search.back());
  for (auto qubit : search) {
    builder.x(qubit);
    builder.h(qubit);
  }
}

static void computePredicate(qc::QCProgramBuilder& builder,
                             llvm::ArrayRef<Value> search, Value work,
                             std::string_view markedBitstring) {
  maskMarkedState(builder, search, markedBitstring);
  builder.h(work);
  builder.mcz(search, work);
  builder.h(work);
  maskMarkedState(builder, search, markedBitstring);
}

SmallVector<Value>
weakMeasurementGrover(qc::QCProgramBuilder& builder,
                      const WeakMeasurementGrover& benchmark) {
  const auto& options = benchmark.options();
  const auto width = static_cast<int64_t>(benchmark.qubits());
  auto search = builder.allocQubitRegister(width, "search");
  auto work = builder.allocQubit();
  auto probe = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(width, benchmark.output().name);

  for (auto qubit : search.qubits) {
    builder.h(qubit);
  }

  const auto angle = 2. * std::asin(std::sqrt(*options.measurementStrength));
  llvm::ArrayRef<Value> searchQubits(search.qubits);
  builder.scfWhile(
      [&] {
        groverIteration(builder, searchQubits, options.markedBitstring);

        computePredicate(builder, searchQubits, work, options.markedBitstring);
        // R_kappa = Ry(2 asin(sqrt(kappa))) Z.
        builder.cz(work, probe);
        builder.cry(angle, work, probe);
        computePredicate(builder, searchQubits, work, options.markedBitstring);

        auto detected = builder.measure(probe);
        auto keepSearching = arith::XOrIOp::create(builder, detected,
                                                   builder.boolConstant(true));
        builder.scfCondition(keepSearching);
      },
      [] {});

  for (size_t index = 0; index < search.qubits.size(); ++index) {
    builder.measure(search[index], result, static_cast<int64_t>(index));
  }
  return {result};
}

} // namespace mqt::bench
