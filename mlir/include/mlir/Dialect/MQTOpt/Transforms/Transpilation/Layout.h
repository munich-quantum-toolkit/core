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

#include "ir/Definitions.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::opt {

/**
 * @brief A qubit layout that maps program and hardware indices without storing
 * Values. Used for efficient memory usage when Value tracking isn't needed.
 *
 * Note that we use the terminology "hardware" and "program" qubits here,
 * because "virtual" (opposed to physical) and "static" (opposed to dynamic)
 * are C++ keywords.
 */
class [[nodiscard]] ThinLayout {
public:
  explicit ThinLayout(const std::size_t nqubits)
      : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}

  /**
   * @brief Insert program:hardware index mapping.
   * @param prog The program index.
   * @param hw The hardware index.
   */
  void add(qc::Qubit prog, qc::Qubit hw) {
    assert(prog < programToHardware_.size() &&
           "add: program index out of bounds");
    assert(hw < hardwareToProgram_.size() &&
           "add: hardware index out of bounds");
    programToHardware_[prog] = hw;
    hardwareToProgram_[hw] = prog;
  }

  /**
   * @brief Look up program index for a hardware index.
   * @param hw The hardware index.
   * @return The program index of the respective hardware index.
   */
  [[nodiscard]] qc::Qubit getProgramIndex(const qc::Qubit hw) const {
    assert(hw < hardwareToProgram_.size() &&
           "getProgramIndex: hardware index out of bounds");
    return hardwareToProgram_[hw];
  }

  /**
   * @brief Look up hardware index for a program index.
   * @param prog The program index.
   * @return The hardware index of the respective program index.
   */
  [[nodiscard]] qc::Qubit getHardwareIndex(const qc::Qubit prog) const {
    assert(prog < programToHardware_.size() &&
           "getHardwareIndex: program index out of bounds");
    return programToHardware_[prog];
  }

  /**
   * @brief Convenience function to lookup multiple hardware indices at once.
   * @param progs The program indices.
   * @return A tuple of hardware indices.
   */
  template <typename... ProgIndices>
    requires(sizeof...(ProgIndices) > 0) &&
            ((std::is_convertible_v<ProgIndices, qc::Qubit>) && ...)
  [[nodiscard]] auto getHardwareIndices(ProgIndices... progs) const {
    return std::tuple{getHardwareIndex(static_cast<qc::Qubit>(progs))...};
  }

  /**
   * @brief Convenience function to lookup multiple program indices at once.
   * @param hws The hardware indices.
   * @return A tuple of program indices.
   */
  template <typename... HwIndices>
    requires(sizeof...(HwIndices) > 0) &&
            ((std::is_convertible_v<HwIndices, qc::Qubit>) && ...)
  [[nodiscard]] auto getProgramIndices(HwIndices... hws) const {
    return std::tuple{getProgramIndex(static_cast<qc::Qubit>(hws))...};
  }

  /**
   * @brief Swap the mapping to hardware indices of two program indices.
   */
  void swap(const qc::Qubit prog0, const qc::Qubit prog1) {
    const qc::Qubit hw0 = programToHardware_[prog0];
    const qc::Qubit hw1 = programToHardware_[prog1];

    std::swap(programToHardware_[prog0], programToHardware_[prog1]);
    std::swap(hardwareToProgram_[hw0], hardwareToProgram_[hw1]);
  }

protected:
  /**
   * @brief Maps a program qubit index to its hardware index.
   */
  mlir::SmallVector<qc::Qubit> programToHardware_;

  /**
   * @brief Maps a hardware qubit index to its program index.
   */
  mlir::SmallVector<qc::Qubit> hardwareToProgram_;
};

/**
 * @brief Enhanced layout that extends ThinLayout with Value tracking
 * capabilities. This is the recommended replacement for the original Layout
 * class.
 */
class [[nodiscard]] Layout : public ThinLayout {
public:
  explicit Layout(const std::size_t nqubits)
      : ThinLayout(nqubits), qubits_(nqubits) {
    valueToMapping_.reserve(nqubits);
  }

  /**
   * @brief Insert program:hardware:value mapping.
   * @param prog The program index.
   * @param hw The hardware index.
   * @param q The SSA value associated with the indices.
   */
  void add(qc::Qubit prog, qc::Qubit hw, mlir::Value q) {
    ThinLayout::add(prog, hw);
    qubits_[hw] = q;
    valueToMapping_.try_emplace(q, prog, hw);
  }

  /**
   * @brief Look up hardware index for a qubit value.
   * @param q The SSA Value representing the qubit.
   * @return The hardware index where this qubit currently resides.
   */
  [[nodiscard]] qc::Qubit lookupHardwareIndex(const mlir::Value q) const {
    const auto it = valueToMapping_.find(q);
    assert(it != valueToMapping_.end() && "lookupHardwareIndex: unknown value");
    return it->second.hw;
  }

  /**
   * @brief Look up qubit value for a hardware index.
   * @param hw The hardware index.
   * @return The SSA value currently representing the qubit at the hardware
   * location.
   */
  [[nodiscard]] mlir::Value lookupHardwareValue(const qc::Qubit hw) const {
    assert(hw < qubits_.size() &&
           "lookupHardwareValue: hardware index out of bounds");
    return qubits_[hw];
  }

  /**
   * @brief Look up program index for a qubit value.
   * @param q The SSA Value representing the qubit.
   * @return The program index where this qubit currently resides.
   */
  [[nodiscard]] qc::Qubit lookupProgramIndex(const mlir::Value q) const {
    const auto it = valueToMapping_.find(q);
    assert(it != valueToMapping_.end() && "lookupProgramIndex: unknown value");
    return it->second.prog;
  }

  /**
   * @brief Look up qubit value for a program index.
   * @param prog The program index.
   * @return The SSA value currently representing the qubit at the program
   * location.
   */
  [[nodiscard]] mlir::Value lookupProgramValue(const qc::Qubit prog) const {
    assert(prog < this->programToHardware_.size() &&
           "lookupProgramValue: program index out of bounds");
    return qubits_[this->programToHardware_[prog]];
  }

  /**
   * @brief Check whether the layout contains a qubit.
   * @param q The SSA Value representing the qubit.
   * @return True if the layout contains the qubit, false otherwise.
   */
  [[nodiscard]] bool contains(const mlir::Value q) const {
    return valueToMapping_.contains(q);
  }

  /**
   * @brief Replace an old SSA value with a new one.
   */
  void remapQubitValue(const mlir::Value in, const mlir::Value out) {
    const auto it = valueToMapping_.find(in);
    assert(it != valueToMapping_.end() &&
           "remapQubitValue: unknown input value");

    const QubitInfo info = it->second;
    qubits_[info.hw] = out;

    assert(!valueToMapping_.contains(out) &&
           "remapQubitValue: output value already mapped");

    valueToMapping_.try_emplace(out, info);
    valueToMapping_.erase(in);
  }

  /**
   * @brief Swap the locations of two program qubits. This is the effect of a
   * SWAP gate.
   */
  void swap(const mlir::Value q0, const mlir::Value q1) {
    auto it0 = valueToMapping_.find(q0);
    auto it1 = valueToMapping_.find(q1);
    assert(it0 != valueToMapping_.end() && it1 != valueToMapping_.end() &&
           "swap: unknown values");

    const qc::Qubit prog0 = it0->second.prog;
    const qc::Qubit prog1 = it1->second.prog;

    std::swap(it0->second.prog, it1->second.prog);

    ThinLayout::swap(prog0, prog1);
  }

  /**
   * @brief Return the current layout.
   */
  mlir::ArrayRef<qc::Qubit> getCurrentLayout() const {
    return this->programToHardware_;
  }

  /**
   * @brief Return the SSA values for hardware indices from 0...nqubits.
   */
  [[nodiscard]] mlir::ArrayRef<mlir::Value> getHardwareQubits() const {
    return qubits_;
  }

private:
  struct QubitInfo {
    qc::Qubit prog;
    qc::Qubit hw;
  };

  /**
   * @brief Maps an SSA value to its `QubitInfo`.
   */
  mlir::DenseMap<mlir::Value, QubitInfo> valueToMapping_;

  /**
   * @brief Maps hardware qubit indices to SSA values.
   */
  mlir::SmallVector<mlir::Value> qubits_;
};
} // namespace mqt::ir::opt
