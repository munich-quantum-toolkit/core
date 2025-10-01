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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {
using namespace mlir;

template <class QubitIndex> class [[nodiscard]] ThinLayout {
public:
  explicit ThinLayout(const std::size_t nqubits)
      : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}

  /**
   * @brief Insert program:hardware index mapping.
   * @param programIdx The program index.
   * @param hardwareIdx The hardware index.
   */
  void add(QubitIndex programIdx, QubitIndex hardwareIdx) {
    assert(programIdx < programToHardware_.size() &&
           "Program index out of bounds");
    assert(hardwareIdx < hardwareToProgram_.size() &&
           "Hardware index out of bounds");
    programToHardware_[programIdx] = hardwareIdx;
    hardwareToProgram_[hardwareIdx] = programIdx;
  }

  /**
   * @brief Look up program index for a hardware index.
   * @param hardwareIdx The hardware index.
   * @return The program index of the respective hardware index.
   */
  [[nodiscard]] QubitIndex lookupProgram(const QubitIndex hardwareIdx) const {
    assert(hardwareIdx < hardwareToProgram_.size() &&
           "Hardware index out of bounds");
    return hardwareToProgram_[hardwareIdx];
  }

  /**
   * @brief Look up hardware index for a program index.
   * @param programIdx The program index.
   * @return The hardware index of the respective program index.
   */
  [[nodiscard]] QubitIndex lookupHardware(const QubitIndex programIdx) const {
    assert(programIdx < programToHardware_.size() &&
           "Program index out of bounds");
    return programToHardware_[programIdx];
  }

  /**
   * @brief Swap the mapping to hardware indices of two program indices.
   */
  void swap(const QubitIndex programIdx0, const QubitIndex programIdx1) {
    const QubitIndex hardwareIdx0 = programToHardware_[programIdx0];
    const QubitIndex hardwareIdx1 = programToHardware_[programIdx1];

    std::swap(programToHardware_[programIdx0], programToHardware_[programIdx1]);

    hardwareToProgram_[hardwareIdx0] = programIdx1;
    hardwareToProgram_[hardwareIdx1] = programIdx0;
  }

private:
  /**
   * @brief Maps a program qubit index to its hardware index.
   */
  SmallVector<QubitIndex> programToHardware_;

  /**
   * @brief Maps a hardware qubit index to its program index.
   */
  SmallVector<QubitIndex> hardwareToProgram_;
};

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

  /**
   * @brief Insert program:hardware:value mapping.
   * @param programIdx The program index.
   * @param hardwareIdx The hardware index.
   * @param q The SSA value associated with the indices.
   */
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
   * @brief Check whether the layout contains a qubit.
   * @param q The SSA Value representing the qubit.
   * @return True if the layout contains the qubit, false otherwise.
   */
  [[nodiscard]] bool contains(const Value q) const {
    return valueToMapping_.contains(q);
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
