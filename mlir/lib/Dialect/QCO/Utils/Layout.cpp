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

Layout Layout::random(const size_t nProgramQubits, const size_t nHardwareQubits,
                      const size_t seed) {
  assert(nProgramQubits <= nHardwareQubits &&
         "cannot place more program qubits than hardware qubits");
  auto hwIndices = llvm::to_vector(llvm::seq(nHardwareQubits));
  llvm::shuffle(hwIndices.begin(), hwIndices.end(), std::mt19937_64{seed});

  Layout layout(nProgramQubits, nHardwareQubits);
  for (size_t prog = 0; prog < nProgramQubits; ++prog) {
    layout.add(prog, hwIndices[prog]);
  }
  return layout;
}

Layout Layout::fromMapping(ArrayRef<size_t> mapping) {
  llvm::SmallBitVector seen(mapping.size());
  for (const size_t hw : mapping) {
    if (hw >= mapping.size() || seen.test(hw)) {
      llvm::reportFatalUsageError("mapping must be a permutation");
    }
    seen.set(hw);
  }

  Layout layout(mapping.size(), mapping.size());
  for (const auto [prog, hw] : enumerate(mapping)) {
    layout.add(prog, hw);
  }
  return layout;
}

void Layout::add(const size_t prog, const size_t hw) {
  assert(prog < nProgramQubits_ && "program index out of bounds");
  assert(hw < nHardwareQubits_ && "hardware index out of bounds");
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

bool Layout::hasProgramAt(const size_t hw) const {
  return hardwareToProgram_.contains(hw);
}

void Layout::swap(const size_t hwA, const size_t hwB) {
  assert(hwA < nHardwareQubits_ && "hardware index out of bounds");
  assert(hwB < nHardwareQubits_ && "hardware index out of bounds");
  if (hwA == hwB) {
    return;
  }
  // Read the current value on each side (may be empty), then write each to the
  // other side. Empty is treated as a legitimate value.
  const auto itA = hardwareToProgram_.find(hwA);
  const auto itB = hardwareToProgram_.find(hwB);
  const bool hasA = itA != hardwareToProgram_.end();
  const bool hasB = itB != hardwareToProgram_.end();
  const size_t progA = hasA ? itA->second : 0;
  const size_t progB = hasB ? itB->second : 0;
  hardwareToProgram_.erase(hwA);
  hardwareToProgram_.erase(hwB);
  if (hasA) {
    hardwareToProgram_[hwB] = progA;
    programToHardware_[progA] = hwB;
  }
  if (hasB) {
    hardwareToProgram_[hwA] = progB;
    programToHardware_[progB] = hwA;
  }
}

size_t Layout::nProgramQubits() const { return nProgramQubits_; }

size_t Layout::nHardwareQubits() const { return nHardwareQubits_; }

SmallVector<size_t> Layout::getProgramToHardware() const {
  SmallVector<size_t> result(nProgramQubits_);
  for (size_t prog = 0; prog < nProgramQubits_; ++prog) {
    result[prog] = getHardwareIndex(prog);
  }
  return result;
}

} // namespace mlir::qco
