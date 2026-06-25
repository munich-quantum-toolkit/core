/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Layout.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <random>

namespace mlir::qco {
Layout Layout::random(const size_t nqubits, const size_t seed) {
  SmallVector<size_t> mapping(nqubits);
  std::iota(mapping.begin(), mapping.end(), size_t{0});
  std::ranges::shuffle(mapping, std::mt19937_64{seed});

  Layout layout(nqubits);
  for (const auto [prog, hw] : enumerate(mapping)) {
    layout.add(prog, hw);
  }

  return layout;
}

void Layout::add(const size_t prog, const size_t hw) {
  assert(prog < programToHardware_.size() && "program index out of bounds");
  assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
  programToHardware_[prog] = hw;
  hardwareToProgram_[hw] = prog;
}

size_t Layout::getProgramIndex(const size_t hw) const {
  assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
  return hardwareToProgram_[hw];
}

size_t Layout::getHardwareIndex(const size_t prog) const {
  assert(prog < programToHardware_.size() && "program index out of bounds");
  return programToHardware_[prog];
}

void Layout::swap(const size_t hwA, const size_t hwB) {
  assert(hwA < hardwareToProgram_.size() && "hardware index out of bounds");
  assert(hwB < hardwareToProgram_.size() && "hardware index out of bounds");
  const auto progA = hardwareToProgram_[hwA];
  const auto progB = hardwareToProgram_[hwB];

  std::swap(hardwareToProgram_[hwA], hardwareToProgram_[hwB]);
  std::swap(programToHardware_[progA], programToHardware_[progB]);
}

size_t Layout::nqubits() const { return programToHardware_.size(); }

ArrayRef<size_t> Layout::getProgramToHardware() const {
  return programToHardware_;
}

} // namespace mlir::qco
