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

#include <cstddef>
#include <tuple>

using namespace mlir;
using namespace mlir::qco;

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

bool PPRotationOp::isNonClifford() {
  auto piFraction = getRotation();
  return piFraction == 4 || piFraction == -4;
}

bool PPRotationOp::isClifford() { return !isNonClifford(); }

void PPRotationOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                         ValueRange qubitsIn, std::int8_t rotation,
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
  build(odsBuilder, odsState, resultTypes, qubitsIn,
        odsBuilder.getIntegerAttr(si8Type, rotation), pauliWord);
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

/**
 * @brief Computes sin(angle / 2) and cos(angle / 2) for a given rotation value
 * which represents multiples of pi/2.
 *
 * @param rotation the rotation value in multiples of pi/2
 * @return std::tuple<double, double> the sine and cosine of half the angle
 */
static std::tuple<double, double> halfAngleSinCos(int8_t rotation) {
  double invSqrt2 = 1.0 / std::numbers::sqrt2;
  switch (rotation) {
  case -4:
  case 4:
    return {0.0, -1.0};
  case -2:
    return {-1.0, 0.0};
  case -1:
    return {-invSqrt2, invSqrt2};
  case 1:
    return {invSqrt2, invSqrt2};
  case 2:
    return {1.0, 0.0};
  default:
    llvm_unreachable("Invalid rotation value. Must be in the range [-4, 4].");
  }
}

// https://github.com/Qiskit/qiskit/blob/stable/2.5/qiskit/circuit/library/generalized_gates/pauli_product_rotation.py#L179-L195
// https://github.com/Qiskit/qiskit/blob/stable/2.5/qiskit/quantum_info/operators/symplectic/pauli.py#L422-L432
// https://github.com/Qiskit/qiskit/blob/main/qiskit/quantum_info/operators/symplectic/base_pauli.py#L41
DynamicMatrix PPRotationOp::getUnitaryMatrix() {
  const auto numQubits = getQubitsIn().size();
  const auto dim = 1ULL << numQubits;
  auto matrix = DynamicMatrix(dim);

  // Get the X and Z bits for the Pauli product
  auto [xBits, zBits] = getPauliXZBits(getPauliProduct());
  auto yBits = xBits & zBits;
  auto phase = std::popcount(yBits) % 4; // Phase contribution from Y terms

  auto [sinAngle, cosAngle] = halfAngleSinCos(getRotation());

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
