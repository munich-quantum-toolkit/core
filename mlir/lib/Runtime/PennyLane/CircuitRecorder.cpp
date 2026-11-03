/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "CircuitRecorder.h"

#include "mqt/Dialect/MQT/Utils/DenseUnitary.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/Translation/StandardGate.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mqt::pennylane {
static void require(bool valid, const llvm::Twine& message) {
  if (!valid) {
    throw std::runtime_error(message.str());
  }
}
static size_t dimension(size_t width) {
  require(width < std::numeric_limits<size_t>::digits,
          "Dense result dimension does not fit size_t.");
  return size_t{1} << width;
}
namespace {
using Gate = mlir::qc::StandardGate;
struct NamedGate {
  llvm::StringLiteral name;
  Gate gate;
  size_t controls = 0;
};
constexpr std::array GATES{
    NamedGate{.name = "Identity", .gate = Gate::Id},
    NamedGate{.name = "PauliX", .gate = Gate::X},
    NamedGate{.name = "PauliY", .gate = Gate::Y},
    NamedGate{.name = "PauliZ", .gate = Gate::Z},
    NamedGate{.name = "Hadamard", .gate = Gate::H},
    NamedGate{.name = "S", .gate = Gate::S},
    NamedGate{.name = "T", .gate = Gate::T},
    NamedGate{.name = "SX", .gate = Gate::SX},
    NamedGate{.name = "RX", .gate = Gate::RX},
    NamedGate{.name = "RY", .gate = Gate::RY},
    NamedGate{.name = "RZ", .gate = Gate::RZ},
    NamedGate{.name = "PhaseShift", .gate = Gate::P},
    NamedGate{.name = "GlobalPhase", .gate = Gate::GPhase},
    NamedGate{.name = "U1", .gate = Gate::P},
    NamedGate{.name = "U2", .gate = Gate::U2},
    NamedGate{.name = "U3", .gate = Gate::U3},
    NamedGate{.name = "SWAP", .gate = Gate::SWAP},
    NamedGate{.name = "ISWAP", .gate = Gate::ISWAP},
    NamedGate{.name = "ECR", .gate = Gate::ECR},
    NamedGate{.name = "IsingXX", .gate = Gate::RXX},
    NamedGate{.name = "IsingYY", .gate = Gate::RYY},
    NamedGate{.name = "IsingZZ", .gate = Gate::RZZ},
    NamedGate{.name = "CNOT", .gate = Gate::X, .controls = 1},
    NamedGate{.name = "CY", .gate = Gate::Y, .controls = 1},
    NamedGate{.name = "CZ", .gate = Gate::Z, .controls = 1},
    NamedGate{.name = "Toffoli", .gate = Gate::X, .controls = 2},
    NamedGate{.name = "CSWAP", .gate = Gate::SWAP, .controls = 1},
    NamedGate{.name = "ControlledPhaseShift", .gate = Gate::P, .controls = 1},
    NamedGate{.name = "CRX", .gate = Gate::RX, .controls = 1},
    NamedGate{.name = "CRY", .gate = Gate::RY, .controls = 1},
    NamedGate{.name = "CRZ", .gate = Gate::RZ, .controls = 1},
};

} // namespace

CircuitRecorder::CircuitRecorder() = default;
CircuitRecorder::CircuitRecorder(size_t count) {
  [[maybe_unused]] const auto wires = allocate(count);
}
CircuitRecorder::~CircuitRecorder() = default;

llvm::SmallVector<size_t>
CircuitRecorder::wireIndices(llvm::ArrayRef<intptr_t> wires) const {
  require(!ids_.empty(), "No active qubit register.");
  llvm::SmallVector<size_t> result;
  result.reserve(wires.size());
  for (const auto wire : wires) {
    require(wire >= ids_.front() &&
                static_cast<uint64_t>(wire - ids_.front()) < ids_.size(),
            "Invalid or released qubit ID.");
    const auto index = static_cast<size_t>(wire - ids_.front());
    require(std::ranges::find(result, index) == result.end(),
            "Duplicate qubit operand.");
    result.push_back(index);
  }
  return result;
}

std::vector<intptr_t> CircuitRecorder::allocate(size_t count) {
  require(ids_.empty(), "Dynamic qubit allocation is unsupported.");
  require(count > 0 && count <= static_cast<size_t>(INTPTR_MAX) -
                                    static_cast<size_t>(nextQubit_),
          "Qubit allocation exceeds the supported range.");
  if (!compiler_) {
    compiler_ = mlir::createCompilerContext();
  }
  builder_ = std::make_unique<mlir::qc::QCProgramBuilder>(compiler_.get());
  builder_->initialize();
  auto reg = builder_->allocQubitRegister(static_cast<int64_t>(count));
  qubits_ = std::move(reg.qubits);
  touched_.assign(count, false);
  ids_.resize(count);
  std::iota(ids_.begin(), ids_.end(), nextQubit_);
  nextQubit_ += static_cast<intptr_t>(count);
  initialState_ = true;
  return ids_;
}

void CircuitRecorder::release(llvm::ArrayRef<intptr_t> wires) {
  const auto indices = wireIndices(wires);
  require(indices.size() == ids_.size(),
          "Partial qubit release is unsupported.");
  builder_.reset();
  qubits_.clear();
  ids_.clear();
}

void CircuitRecorder::namedOperation(const std::string& name,
                                     const std::vector<double>& parameters,
                                     const std::vector<intptr_t>& wires,
                                     bool inverse,
                                     const std::vector<intptr_t>& controls,
                                     const std::vector<bool>& controlValues) {
  require(builder_ != nullptr,
          "An active register is required before recording gates.");
  require(controls.size() == controlValues.size(),
          "Control wires and values have different lengths.");
  const auto gates = llvm::ArrayRef(GATES);
  const auto* const gate =
      llvm::find_if(gates, [&](const NamedGate& candidate) {
        return candidate.name == name;
      });
  require(gate != gates.end(), "Unsupported gate: " + name);
  const auto& descriptor = mlir::qc::getStandardGateDescriptor(gate->gate);
  require(parameters.size() == descriptor.parameterCount &&
              (gate->gate == Gate::GPhase || gate->gate == Gate::Id ||
               wires.size() == descriptor.targetCount + gate->controls),
          "Invalid parameter or wire count for " + name + '.');
  require(std::ranges::all_of(
              parameters, [](double value) { return std::isfinite(value); }),
          "Non-finite parameter for " + name + '.');
  auto operands = controls;
  operands.insert(operands.end(), wires.begin(), wires.end());
  const auto indices = wireIndices(operands);
  for (const auto index : indices) {
    touched_[index] = true;
  }
  if (gate->gate == Gate::Id) {
    initialState_ = false;
    return;
  }
  auto& builder = *builder_;
  llvm::SmallVector<mlir::Value> angles;
  for (const double angle : parameters) {
    angles.push_back(
        builder.floatConstant(gate->gate == Gate::GPhase ? -angle : angle));
  }
  auto operationIndices = llvm::ArrayRef(indices);
  if (gate->gate == Gate::GPhase) {
    operationIndices = operationIndices.take_front(controls.size());
  }
  emitOperation(
      operationIndices, controls.size() + gate->controls, controlValues,
      inverse, [&](mlir::ValueRange values) {
        require(mlir::succeeded(mlir::qc::emitStandardGate(
                    builder, builder.getLoc(), gate->gate, angles, values)),
                "Cannot emit gate " + name + '.');
      });
}

void CircuitRecorder::emitOperation(
    llvm::ArrayRef<size_t> indices, size_t numControls,
    const std::vector<bool>& controlValues, bool inverse,
    llvm::function_ref<void(mlir::ValueRange)> primitive) {
  auto& builder = *builder_;
  llvm::SmallVector<mlir::Value> controlQubits;
  llvm::SmallVector<mlir::Value> targets;
  for (size_t i = 0; i < indices.size(); ++i) {
    (i < numControls ? controlQubits : targets).push_back(qubits_[indices[i]]);
  }
  const auto emit = [&](mlir::ValueRange values) {
    if (inverse) {
      builder.inv(values, primitive);
    } else {
      primitive(values);
    }
  };
  for (size_t i = 0; i < controlValues.size(); ++i) {
    if (!controlValues[i]) {
      builder.x(controlQubits[i]);
    }
  }
  if (controlQubits.empty()) {
    emit(targets);
  } else {
    builder.ctrl(controlQubits, targets, emit);
  }
  for (size_t i = 0; i < controlValues.size(); ++i) {
    if (!controlValues[i]) {
      builder.x(controlQubits[i]);
    }
  }
  initialState_ = false;
}

void CircuitRecorder::matrixOperation(
    const std::vector<std::complex<double>>& matrix,
    const std::vector<intptr_t>& wires, bool inverse,
    const std::vector<intptr_t>& controls,
    const std::vector<bool>& controlValues) {
  require(builder_ != nullptr,
          "An active register is required before recording gates.");
  require(!wires.empty() && wires.size() <= 2 &&
              controls.size() <= 2 - wires.size(),
          "Dense unitaries support at most two wires, including controls.");
  require(controls.size() == controlValues.size(),
          "Control wires and values have different lengths.");
  const auto size = dimension(wires.size());
  require(matrix.size() == size * size,
          "Unitary matrix dimensions do not match its wires.");
  auto operands = controls;
  operands.insert(operands.end(), wires.begin(), wires.end());
  const auto indices = wireIndices(operands);
  llvm::SmallVector<mlir::Value> targets;
  for (const auto index : llvm::ArrayRef(indices).drop_front(controls.size())) {
    targets.push_back(qubits_[index]);
  }
  auto type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(size), static_cast<int64_t>(size)},
      mlir::ComplexType::get(builder_->getF64Type()));
  auto values = mlir::DenseElementsAttr::get(
      type, llvm::ArrayRef<std::complex<double>>(matrix));
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      compiler_.get(), [&](mlir::Diagnostic& diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  const auto valid = mlir::succeeded(mlir::mqt::verifyDenseUnitaryMatrix(
      builder_->getInsertionBlock()->getParentOp(), values, targets));
  require(valid, "Invalid unitary matrix: " + diagnostics);
  for (const auto index : indices) {
    touched_[index] = true;
  }
  emitOperation(
      indices, controls.size(), controlValues, inverse,
      [&](mlir::ValueRange qubits) { builder_->unitary(qubits, values); });
}

void CircuitRecorder::setBasisState(std::span<const int8_t> state,
                                    const std::vector<intptr_t>& wires) {
  require(canPrepareBasisState(),
          "Basis state preparation must precede all other operations.");
  require(state.size() == wires.size() && !wires.empty(),
          "Basis state length must match its wires.");
  const auto indices = wireIndices(wires);
  for (const auto bit : state) {
    require(bit == 0 || bit == 1, "Basis state entries must be zero or one.");
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (state[i] == 1) {
      touched_[indices[i]] = true;
      builder_->x(qubits_[indices[i]]);
    }
  }
  initialState_ = false;
}

bool CircuitRecorder::hasTouchedWires() const {
  return llvm::is_contained(touched_, true);
}

mlir::QCProgram CircuitRecorder::takeProgram() {
  require(builder_ != nullptr, "No circuit is available for preparation.");
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      compiler_.get(), [&](mlir::Diagnostic& diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        stream << '\n';
        return mlir::success();
      });
  auto& builder = *builder_;
  auto bits = builder.allocClassicalBitRegister(
      static_cast<int64_t>(qubits_.size()), "c");
  for (size_t i = 0; i < qubits_.size(); ++i) {
    if (touched_[i]) {
      builder.measure(qubits_[i], bits, static_cast<int64_t>(i));
    }
  }
  auto moduleOp = builder.finalize({bits});
  builder_.reset();
  auto program = mlir::QCProgram::fromModule(compiler_, std::move(moduleOp));
  require(mlir::succeeded(program), "Invalid recorded circuit: " + diagnostics);
  return std::move(*program);
}

} // namespace mqt::pennylane
