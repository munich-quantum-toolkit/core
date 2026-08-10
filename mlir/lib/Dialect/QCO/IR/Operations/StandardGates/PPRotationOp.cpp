/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::utils;

Value PPRotationOp::getInputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsIn()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value PPRotationOp::getOutputTarget(const size_t i) {
  if (i < getNumTargets()) {
    return getQubitsOut()[i];
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value PPRotationOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getQubitsOut()[i]) {
      return getQubitsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value PPRotationOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getQubitsIn()[i]) {
      return getQubitsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

void PPRotationOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                         ValueRange qubitsIn,
                         const std::variant<double, Value>& theta,
                         ArrayRef<Pauli> pauliProduct) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(qubitsIn.size());
  for (auto qubit : qubitsIn) {
    resultTypes.push_back(qubit.getType());
  }
  auto si8Type = odsBuilder.getIntegerType(8, true);
  SmallVector<Attribute> pauliAttrs;
  pauliAttrs.reserve(pauliProduct.size());
  for (const auto& pauli : pauliProduct) {
    pauliAttrs.push_back(PauliAttr::get(odsBuilder.getContext(), pauli));
  }
  auto pauliWord = odsBuilder.getArrayAttr(pauliAttrs);
  const auto thetaOperand =
      variantToValue(odsBuilder, odsState.location, theta);
  build(odsBuilder, odsState, resultTypes, qubitsIn, thetaOperand, pauliWord);
}

LogicalResult PPRotationOp::verify() {
  size_t numPaulis = getPauliProduct().size();
  if (numPaulis == 0) {
    return emitOpError("pauli_product must be non-empty");
  }
  if (numPaulis != getQubitsIn().size()) {
    return emitOpError("number of elements in pauli_product must match "
                       "number of input qubits");
  }
  return success();
}

Value PPRotationOp::getParameter(const size_t i) {
  if (i >= 1) {
    llvm::reportFatalUsageError("Parameter index out of bounds");
  }
  return getTheta();
}

OperandRange PPRotationOp::getParameters() {
  return getOperation()->getOperands().take_front(1);
}

static void setBitOn(uint64_t& bits, size_t index) { bits |= (1ULL << index); }

/**
 * @brief Get bit representations representing the X and Z components of a Pauli
 * product.
 *
 * @param pauliProduct the Pauli string as an ArrayAttr of PauliAttr
 * @return std::tuple<uint64_t, uint64_t> the X and Z bits
 */
static std::tuple<uint64_t, uint64_t> getPauliXZBits(ArrayAttr pauliProduct) {
  const auto numPaulis = pauliProduct.size();
  assert(numPaulis <= 64 && "Pauli product must have at most 64 qubits");
  uint64_t xBits = 0;
  uint64_t zBits = 0;

  for (size_t i = 0; i < numPaulis; ++i) {
    const auto pauli = cast<PauliAttr>(pauliProduct[i]).getValue();
    switch (pauli) {
    case Pauli::I:
      break;
    case Pauli::X:
      setBitOn(xBits, i);
      break;
    case Pauli::Y:
      setBitOn(xBits, i);
      setBitOn(zBits, i);
      break;
    case Pauli::Z:
      setBitOn(zBits, i);
      break;
    default:
      llvm_unreachable("Invalid Pauli value");
    }
  }

  return {xBits, zBits};
}

DynamicMatrix PPRotationOp::unitaryMatrix(const double theta) {
  const auto numQubits = getQubitsIn().size();
  const auto dim = 1ULL << numQubits;
  auto matrix = DynamicMatrix(dim);

  // Get the X and Z bits for the Pauli product
  auto [xBits, zBits] = getPauliXZBits(getPauliProduct());
  auto yBits = xBits & zBits;
  auto phase = std::popcount(yBits) % 4; // Phase contribution from Y terms

  auto sinAngle = std::sin(theta / 2.0);
  auto cosAngle = std::cos(theta / 2.0);

  for (uint64_t i = 0; i < dim; ++i) {
    // Get sign (Z contributions on this state, parity)
    int actingZGates = std::popcount(i & zBits);
    int sign = (actingZGates % 2 == 0) ? 1 : -1;

    // Get the column based on the X contributions
    uint64_t index = i ^ xBits;

    // Get element value.
    // The pauli matrix has coeff = (-i)^phase, but another (-i) factor comes
    // from the exponentiation, so the final coeff is (-i)^(phase + 1)
    int exp = phase + 1;
    double element = sign * sinAngle;
    std::complex<double> complexElement;
    switch (exp % 4) {
    case 0:
      complexElement = std::complex<double>(element, 0.0);
      break;
    case 1:
      complexElement = std::complex<double>(0.0, -element);
      break;
    case 2:
      complexElement = std::complex<double>(-element, 0.0);
      break;
    case 3:
      complexElement = std::complex<double>(0.0, element);
      break;
    }

    matrix(i, index) = complexElement;
  }

  for (uint64_t i = 0; i < dim; ++i) {
    matrix(i, i) += cosAngle;
  }

  return matrix;
}

std::optional<DynamicMatrix> PPRotationOp::getUnitaryMatrix() {
  if (const auto theta = valueToDouble(getTheta())) {
    return unitaryMatrix(*theta);
  }
  return std::nullopt;
}

bool PPRotationOp::hasCompileTimeKnownUnitaryMatrix() {
  return valueToDouble(getTheta()).has_value();
}
