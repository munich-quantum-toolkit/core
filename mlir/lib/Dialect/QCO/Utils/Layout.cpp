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

#include <cassert>
#include <cstddef>
#include <limits>
#include <random>

namespace mlir::qco {

namespace {
/// Sentinel stored in `programToHardware_` and `hardwareToProgram_` entries
/// that do not currently hold a valid index.
constexpr size_t UNMAPPED = std::numeric_limits<size_t>::max();
} // namespace

Layout::Layout(const size_t nProgramQubits, const size_t nHardwareQubits)
    : programToHardware_(nProgramQubits, UNMAPPED),
      hardwareToProgram_(nHardwareQubits, UNMAPPED) {}

Layout Layout::random(const size_t nProgramQubits, const size_t nHardwareQubits,
                      const size_t seed) {
  assert(nProgramQubits <= nHardwareQubits &&
         "cannot map more program qubits than hardware qubits");
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
  assert(prog < programToHardware_.size() && "program index out of bounds");
  assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
  assert(programToHardware_[prog] == UNMAPPED &&
         "program index already mapped");
  assert(hardwareToProgram_[hw] == UNMAPPED && "hardware index already mapped");
  programToHardware_[prog] = hw;
  hardwareToProgram_[hw] = prog;
}

size_t Layout::getProgramIndex(const size_t hw) const {
  assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
  const auto prog = hardwareToProgram_[hw];
  assert(prog != UNMAPPED && "hardware index not mapped");
  return prog;
}

size_t Layout::getHardwareIndex(const size_t prog) const {
  assert(prog < programToHardware_.size() && "program index out of bounds");
  const auto hw = programToHardware_[prog];
  assert(hw != UNMAPPED && "program index not mapped");
  return hw;
}

bool Layout::hasProgramAt(const size_t hw) const {
  assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
  return hardwareToProgram_[hw] != UNMAPPED;
}

void Layout::swap(const size_t hwA, const size_t hwB) {
  assert(hwA < hardwareToProgram_.size() && "hardware index out of bounds");
  assert(hwB < hardwareToProgram_.size() && "hardware index out of bounds");
  if (hwA == hwB) {
    return;
  }
  const size_t progA = hardwareToProgram_[hwA];
  const size_t progB = hardwareToProgram_[hwB];
  assert(progA != UNMAPPED && "hardware index not mapped");
  assert(progB != UNMAPPED && "hardware index not mapped");
  hardwareToProgram_[hwA] = progB;
  hardwareToProgram_[hwB] = progA;
  programToHardware_[progA] = hwB;
  programToHardware_[progB] = hwA;
}

size_t Layout::nProgramQubits() const { return programToHardware_.size(); }

size_t Layout::nHardwareQubits() const { return hardwareToProgram_.size(); }

ArrayRef<size_t> Layout::getProgramToHardware() const {
#ifndef NDEBUG
  for (const size_t hw : programToHardware_) {
    assert(hw != UNMAPPED && "program qubit not mapped");
  }
#endif
  return programToHardware_;
}

} // namespace mlir::qco
