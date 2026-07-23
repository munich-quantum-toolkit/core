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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

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
  assert(prog < nqubits_ && "program index out of bounds");
  assert(hw < nqubits_ && "hardware index out of bounds");
  assert(!programToHardware_.contains(prog) && "program index already mapped");
  assert(!hardwareToProgram_.contains(hw) && "hardware index already mapped");
  programToHardware_[prog] = hw;
  hardwareToProgram_[hw] = prog;
}

size_t Layout::getProgramIndex(const size_t hw) const {
  const auto it = hardwareToProgram_.find(hw);
  assert(it != hardwareToProgram_.end() && "hardware index not mapped");
  return it->second;
}

size_t Layout::getHardwareIndex(const size_t prog) const {
  const auto it = programToHardware_.find(prog);
  assert(it != programToHardware_.end() && "program index not mapped");
  return it->second;
}

void Layout::swap(const size_t hwA, const size_t hwB) {
  const auto itA = hardwareToProgram_.find(hwA);
  const auto itB = hardwareToProgram_.find(hwB);
  assert(itA != hardwareToProgram_.end() && "hardware index not mapped");
  assert(itB != hardwareToProgram_.end() && "hardware index not mapped");
  const auto progA = itA->second;
  const auto progB = itB->second;
  itA->second = progB;
  itB->second = progA;
  programToHardware_[progA] = hwB;
  programToHardware_[progB] = hwA;
}

size_t Layout::nqubits() const { return nqubits_; }

SmallVector<size_t> Layout::getProgramToHardware() const {
  SmallVector<size_t> result(nqubits_);
  for (size_t prog = 0; prog < nqubits_; ++prog) {
    result[prog] = getHardwareIndex(prog);
  }
  return result;
}

} // namespace mlir::qco
