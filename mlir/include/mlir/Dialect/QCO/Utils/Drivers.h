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

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <utility>

namespace mlir::qco {
class Qubits {
  /**
   * @brief Specifies the qubit "location" (hardware or program).
   */
  enum class QubitLocation : std::uint8_t { Hardware, Program };

public:
  /**
   * @brief Add qubit with automatically assigned dynamic index.
   */
  [[maybe_unused]] void add(TypedValue<QubitType> q);

  /**
   * @brief Add qubit with static index.
   */
  void add(TypedValue<QubitType> q, std::size_t hw);

  /**
   * @brief Remap the qubit value from prev to next.
   */
  void remap(TypedValue<QubitType> prev, TypedValue<QubitType> next);

  /**
   * @brief Remove the qubit value.
   */
  void remove(TypedValue<QubitType> q);

  /**
   * @returns the qubit value assigned to a program index.
   */
  [[maybe_unused]] TypedValue<QubitType> getProgramQubit(std::size_t index);

  /**
   * @returns the qubit value assigned to a hardware index.
   */
  TypedValue<QubitType> getHardwareQubit(std::size_t index);

private:
  DenseMap<std::size_t, TypedValue<QubitType>> programToValue_;
  DenseMap<std::size_t, TypedValue<QubitType>> hardwareToValue_;
  DenseMap<TypedValue<QubitType>, std::pair<QubitLocation, std::size_t>>
      valueToIndex_;
};

/**
 * @brief Perform top-down non-recursive walk of all operations within a
 * region and apply callback function.
 * @details The signature of the callback function is:
 *
 *     (Operation*, Qubits& q) -> WalkResult
 *
 * where the Qubits object tracks the front of qubit SSA values.
 * @param region The targeted region.
 * @param fn The callback function.
 */
template <typename Fn> void walkUnit(Region& region, Fn&& fn) {
  const auto ffn = std::forward<Fn>(fn);

  Qubits qubits;
  for (Operation& curr : region.getOps()) {
    if (ffn(&curr, qubits).wasInterrupted()) {
      break;
    };

    TypeSwitch<Operation*>(&curr)
        .template Case<StaticOp>([&](StaticOp op) {
          qubits.add(cast<TypedValue<QubitType>>(op.getQubit()), op.getIndex());
        })
        .template Case<AllocOp>([&](AllocOp op) {
          qubits.add(cast<TypedValue<QubitType>>(op.getResult()));
        })
        .template Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
          for (const auto& [prevV, nextV] :
               llvm::zip(op.getInputQubits(), op.getOutputQubits())) {
            const auto prevQ = cast<TypedValue<QubitType>>(prevV);
            const auto nextQ = cast<TypedValue<QubitType>>(nextV);
            qubits.remap(prevQ, nextQ);
          }
        })
        .template Case<ResetOp>([&](ResetOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut());
        })
        .template Case<MeasureOp>([&](MeasureOp op) {
          qubits.remap(op.getQubitIn(), op.getQubitOut());
        })
        .template Case<SinkOp>(
            [&](SinkOp op) { qubits.remove(op.getQubit()); });
  }
}
} // namespace mlir::qco
