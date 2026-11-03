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

#include "mqt/Compiler/Programs.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::pennylane {

/// Record PennyLane operations as a QC program without a device session.
class CircuitRecorder {
public:
  CircuitRecorder();
  explicit CircuitRecorder(size_t count);
  ~CircuitRecorder();
  [[nodiscard]] std::vector<intptr_t> allocate(size_t count);
  void release(llvm::ArrayRef<intptr_t> wires);
  [[nodiscard]] const std::vector<intptr_t>& wires() const { return ids_; }
  [[nodiscard]] bool canPrepareBasisState() const {
    return builder_ != nullptr && initialState_;
  }
  [[nodiscard]] llvm::SmallVector<size_t>
  wireIndices(llvm::ArrayRef<intptr_t> wires) const;
  void namedOperation(const std::string& name,
                      const std::vector<double>& parameters,
                      const std::vector<intptr_t>& wires, bool inverse = false,
                      const std::vector<intptr_t>& controls = {},
                      const std::vector<bool>& controlValues = {});
  void matrixOperation(const std::vector<std::complex<double>>& matrix,
                       const std::vector<intptr_t>& wires, bool inverse = false,
                       const std::vector<intptr_t>& controls = {},
                       const std::vector<bool>& controlValues = {});
  void setBasisState(std::span<const int8_t> state,
                     const std::vector<intptr_t>& wires);
  [[nodiscard]] bool hasTouchedWires() const;
  [[nodiscard]] mlir::QCProgram takeProgram();

private:
  void emitOperation(llvm::ArrayRef<size_t> indices, size_t numControls,
                     const std::vector<bool>& controlValues, bool inverse,
                     llvm::function_ref<void(mlir::ValueRange)> primitive);
  std::shared_ptr<mlir::MLIRContext> compiler_;
  std::unique_ptr<mlir::qc::QCProgramBuilder> builder_;
  llvm::SmallVector<mlir::Value> qubits_;
  std::vector<intptr_t> ids_;
  std::vector<bool> touched_;
  intptr_t nextQubit_ = 0;
  bool initialState_ = true;
};

} // namespace mqt::pennylane
