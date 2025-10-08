/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {
using namespace mlir;

/**
 * @brief This class maintains the bi-directional mapping between program and
 * hardware qubits.
 *
 * Note that we use the terminology "hardware" and "program" qubits here,
 * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
 * are C++ keywords.
 */
template <class QubitIndex> class [[nodiscard]] Layout {
public:
  explicit Layout(const std::size_t nqubits)
      : qubits_(nqubits), programToHardware_(nqubits) {
    valueToMapping_.reserve(nqubits);
  }

  void add(QubitIndex programIdx, QubitIndex hardwareIdx, Value q) {
    const QubitInfo info{.hardwareIdx = hardwareIdx, .programIdx = programIdx};
    qubits_[info.hardwareIdx] = q;
    programToHardware_[programIdx] = info.hardwareIdx;
    valueToMapping_.try_emplace(q, info);
  }

  /**
   * @brief Look up hardware index for a qubit value.
   * @param q The SSA Value representing the qubit.
   * @return The hardware index where this qubit currently resides.
   */
  [[nodiscard]] QubitIndex lookupHardware(const Value q) const {
    return valueToMapping_.at(q).hardwareIdx;
  }

  /**
   * @brief Look up qubit value for a hardware index.
   * @param hardwareIdx The hardware index.
   * @return The SSA value currently representing the qubit at the hardware
   * location.
   */
  [[nodiscard]] Value lookupHardware(const QubitIndex hardwareIdx) const {
    assert(hardwareIdx < qubits_.size() && "Hardware index out of bounds");
    return qubits_[hardwareIdx];
  }

  /**
   * @brief Look up program index for a qubit value.
   * @param q The SSA Value representing the qubit.
   * @return The program index where this qubit currently resides.
   */
  [[nodiscard]] QubitIndex lookupProgram(const Value q) const {
    return valueToMapping_.at(q).programIdx;
  }

  /**
   * @brief Look up qubit value for a program index.
   * @param programIdx The program index.
   * @return The SSA value currently representing the qubit at the program
   * location.
   */
  [[nodiscard]] Value lookupProgram(const QubitIndex programIdx) const {
    const QubitIndex hardwareIdx = programToHardware_[programIdx];
    return lookupHardware(hardwareIdx);
  }

  /**
   * @brief Replace an old SSA value with a new one.
   */
  void remapQubitValue(const Value in, const Value out) {
    const auto it = valueToMapping_.find(in);
    assert(it != valueToMapping_.end() && "forward: unknown input value");

    const QubitInfo map = it->second;
    qubits_[map.hardwareIdx] = out;

    assert(!valueToMapping_.contains(out) &&
           "forward: output value already mapped");

    valueToMapping_.try_emplace(out, map);
    valueToMapping_.erase(in);
  }

  /**
   * @brief Swap the locations of two program qubits. This is the effect of a
   * SWAP gate.
   */
  void swap(const Value q0, const Value q1) {
    auto ita = valueToMapping_.find(q0);
    auto itb = valueToMapping_.find(q1);
    assert(ita != valueToMapping_.end() && itb != valueToMapping_.end() &&
           "swap: unknown values");
    std::swap(ita->second.programIdx, itb->second.programIdx);
    std::swap(programToHardware_[ita->second.programIdx],
              programToHardware_[itb->second.programIdx]);
  }

  /**
   * @brief Return the current layout.
   */
  ArrayRef<QubitIndex> getCurrentLayout() { return programToHardware_; }

  /**
   * @brief Return the SSA values for hardware indices from 0...nqubits.
   */
  [[nodiscard]] ArrayRef<Value> getHardwareQubits() const { return qubits_; }

private:
  struct QubitInfo {
    QubitIndex hardwareIdx;
    QubitIndex programIdx;
  };

  /**
   * @brief Maps an SSA value to its `QubitInfo`.
   */
  DenseMap<Value, QubitInfo> valueToMapping_;

  /**
   * @brief Maps hardware qubit indices to SSA values.
   */
  SmallVector<Value> qubits_;

  /**
   * @brief Maps a program qubit index to its hardware index.
   */
  SmallVector<QubitIndex> programToHardware_;
};
} // namespace mqt::ir::opt
