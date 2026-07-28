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
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>

namespace mlir::qco {
Layout Layout::random(const size_t nqubits, const size_t seed) {
  auto mapping = llvm::to_vector(llvm::seq(nqubits));
  llvm::shuffle(mapping.begin(), mapping.end(), std::mt19937_64{seed});
  return fromMapping(mapping);
}

Layout Layout::fromMapping(ArrayRef<size_t> mapping) {
  llvm::SmallBitVector seen(mapping.size());
  for (const size_t hw : mapping) {
    if (hw >= mapping.size() || seen.test(hw)) {
      llvm::reportFatalUsageError("mapping must be a permutation");
    }
    seen.set(hw);
  }

  Layout layout(mapping.size());
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
