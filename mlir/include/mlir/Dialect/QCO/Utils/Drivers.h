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
#include <llvm/Support/Debug.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <utility>

namespace mlir::qco {
class Qubits {
  /**
   * @brief Specifies the qubit "location" (static or dynamic).
   */
  enum class QubitLocation : std::uint8_t { Hardware, Program };

public:
  /**
   * @brief Add qubit with automatically assigned dynamic index.
   */
  [[maybe_unused]] void add(TypedValue<QubitType> q) {
    const auto index = programToValue_.size();
    programToValue_.try_emplace(index, q);
    valueToIndex_.try_emplace(q, std::make_pair(QubitLocation::Program, index));
  }

  /**
   * @brief Add qubit with static index.
   */
  void add(TypedValue<QubitType> q, std::size_t hw) {
    hardwareToValue_.try_emplace(hw, q);
    valueToIndex_.try_emplace(q, std::make_pair(QubitLocation::Hardware, hw));
  }

  /**
   * @brief Remap the qubit value from prev to next.
   */
  void remap(TypedValue<QubitType> prev, TypedValue<QubitType> next) {
    const auto& [location, index] = valueToIndex_.lookup(prev);

    valueToIndex_.erase(prev);
    valueToIndex_.try_emplace(next, std::make_pair(location, index));

    if (location == QubitLocation::Program) {
      programToValue_[index] = next;
      return;
    }

    hardwareToValue_[index] = next;
  }

  /**
   * @brief Remove the qubit value.
   */
  void remove(TypedValue<QubitType> q) {
    assert(valueToIndex_.contains(q));
    const auto& [location, index] = valueToIndex_.lookup(q);

    valueToIndex_.erase(q);

    if (location == QubitLocation::Program) {
      programToValue_.erase(index);
      return;
    }

    hardwareToValue_.erase(index);
  }

  /**
   * @returns the qubit value assigned to a program index.
   */
  [[maybe_unused]] TypedValue<QubitType> getProgramQubit(std::size_t index) {
    assert(programToValue_.contains(index));
    return programToValue_.lookup(index);
  }

  /**
   * @returns the qubit value assigned to a hardware index.
   */
  TypedValue<QubitType> getHardwareQubit(std::size_t index) {
    assert(hardwareToValue_.contains(index));
    return hardwareToValue_.lookup(index);
  }

private:
  DenseMap<std::size_t, TypedValue<QubitType>> programToValue_;
  DenseMap<std::size_t, TypedValue<QubitType>> hardwareToValue_;
  DenseMap<TypedValue<QubitType>, std::pair<QubitLocation, std::size_t>>
      valueToIndex_;
};

template <typename Fn> LogicalResult walkUnit(Region& region, Fn&& fn) {
  const auto ffn = std::forward<Fn>(fn);

  Qubits qubits;
  for (Operation& curr : region.getOps()) {
    if (ffn(&curr, qubits).wasInterrupted()) {
      break;
    };

    TypeSwitch<Operation*>(&curr)
        .template Case<StaticOp>(
            [&](StaticOp op) { qubits.add(op.getQubit(), op.getIndex()); })
        .template Case<AllocOp>([&](AllocOp op) { qubits.add(op.getResult()); })
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
        .template Case<DeallocOp>(
            [&](DeallocOp op) { qubits.remove(op.getQubit()); });
  }

  return success();
}
} // namespace mlir::qco
