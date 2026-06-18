/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "native_synthesis_test_helpers.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>

using namespace mlir;

namespace mlir::qco::native_synth_test {

TestMatrix TestMatrix::identity(std::size_t dim) {
  TestMatrix result(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    result(i, i) = std::complex<double>{1.0, 0.0};
  }
  return result;
}

TestMatrix TestMatrix::fromMatrix2x2(const Matrix2x2& matrix) {
  TestMatrix result(2);
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t col = 0; col < 2; ++col) {
      result(row, col) = matrix(row, col);
    }
  }
  return result;
}

TestMatrix TestMatrix::fromMatrix4x4(const Matrix4x4& matrix) {
  TestMatrix result(4);
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      result(row, col) = matrix(row, col);
    }
  }
  return result;
}

TestMatrix TestMatrix::operator*(const TestMatrix& rhs) const {
  TestMatrix result(dim_);
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t k = 0; k < dim_; ++k) {
      const std::complex<double> a = (*this)(row, k);
      if (a == std::complex<double>{0.0, 0.0}) {
        continue;
      }
      for (std::size_t col = 0; col < dim_; ++col) {
        result(row, col) += a * rhs(k, col);
      }
    }
  }
  return result;
}

TestMatrix TestMatrix::operator*(std::complex<double> scalar) const {
  TestMatrix result(dim_);
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t col = 0; col < dim_; ++col) {
      result(row, col) = (*this)(row, col) * scalar;
    }
  }
  return result;
}

TestMatrix TestMatrix::adjoint() const {
  TestMatrix result(dim_);
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t col = 0; col < dim_; ++col) {
      result(col, row) = std::conj((*this)(row, col));
    }
  }
  return result;
}

std::complex<double> TestMatrix::trace() const {
  std::complex<double> sum{0.0, 0.0};
  for (std::size_t i = 0; i < dim_; ++i) {
    sum += (*this)(i, i);
  }
  return sum;
}

bool TestMatrix::isApprox(const TestMatrix& other, double tol) const {
  if (dim_ != other.dim_) {
    return false;
  }
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t col = 0; col < dim_; ++col) {
      if (std::abs((*this)(row, col) - other(row, col)) > tol) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] static std::optional<Value>
getUnitaryQubitOperand(qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getOperand(index);
  if (!llvm::isa<qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static std::optional<Value>
getUnitaryQubitResult(qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getResult(index);
  if (!llvm::isa<qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

std::complex<double> phasedAmplitude(const double magnitude,
                                     const double phase) {
  return std::complex<double>(magnitude, 0.0) *
         std::exp(std::complex<double>(0.0, phase));
}

Matrix2x2 u3Matrix(double theta, double phi, double lambda) {
  return decomposition::uMatrix(theta, phi, lambda);
}

bool isUnitary(const Matrix2x2& matrix, const double atol) {
  return (matrix * matrix.adjoint()).isIdentity(atol) &&
         (matrix.adjoint() * matrix).isIdentity(atol);
}

std::optional<double> evaluateConstF64(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto cst = value.getDefiningOp<arith::ConstantFloatOp>()) {
    if (auto attr = llvm::dyn_cast<FloatAttr>(cst.getValue())) {
      return attr.getValueAsDouble();
    }
    return std::nullopt;
  }
  if (auto neg = value.getDefiningOp<arith::NegFOp>()) {
    if (auto v = evaluateConstF64(neg.getOperand())) {
      return -*v;
    }
    return std::nullopt;
  }
  if (auto add = value.getDefiningOp<arith::AddFOp>()) {
    auto lhs = evaluateConstF64(add.getLhs());
    auto rhs = evaluateConstF64(add.getRhs());
    if (lhs && rhs) {
      return *lhs + *rhs;
    }
    return std::nullopt;
  }
  if (auto sub = value.getDefiningOp<arith::SubFOp>()) {
    auto lhs = evaluateConstF64(sub.getLhs());
    auto rhs = evaluateConstF64(sub.getRhs());
    if (lhs && rhs) {
      return *lhs - *rhs;
    }
    return std::nullopt;
  }
  if (auto mul = value.getDefiningOp<arith::MulFOp>()) {
    auto lhs = evaluateConstF64(mul.getLhs());
    auto rhs = evaluateConstF64(mul.getRhs());
    if (lhs && rhs) {
      return *lhs * *rhs;
    }
    return std::nullopt;
  }
  if (auto div = value.getDefiningOp<arith::DivFOp>()) {
    auto lhs = evaluateConstF64(div.getLhs());
    auto rhs = evaluateConstF64(div.getRhs());
    if (lhs && rhs) {
      return *lhs / *rhs;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// Extract the 2x2 unitary matrix associated with a single-qubit op.
bool extractSingleQubitMatrix(qco::UnitaryOpInterface op, Matrix2x2& out) {
  if (llvm::isa<qco::RZOp>(op.getOperation())) {
    auto* raw = op.getOperation();
    if (raw->getNumOperands() < 2) {
      return false;
    }
    auto theta = evaluateConstF64(raw->getOperand(1));
    if (!theta) {
      return false;
    }
    out = qco::decomposition::rzMatrix(*theta);
    return true;
  }
  if (llvm::isa<qco::RXOp>(op.getOperation())) {
    auto* raw = op.getOperation();
    if (raw->getNumOperands() < 2) {
      return false;
    }
    auto theta = evaluateConstF64(raw->getOperand(1));
    if (!theta) {
      return false;
    }
    out = qco::decomposition::rxMatrix(*theta);
    return true;
  }
  if (llvm::isa<qco::RYOp>(op.getOperation())) {
    auto* raw = op.getOperation();
    if (raw->getNumOperands() < 2) {
      return false;
    }
    auto theta = evaluateConstF64(raw->getOperand(1));
    if (!theta) {
      return false;
    }
    out = qco::decomposition::ryMatrix(*theta);
    return true;
  }
  if (llvm::isa<qco::UOp>(op.getOperation())) {
    auto* raw = op.getOperation();
    if (raw->getNumOperands() < 4) {
      return false;
    }
    auto theta = evaluateConstF64(raw->getOperand(1));
    auto phi = evaluateConstF64(raw->getOperand(2));
    auto lambda = evaluateConstF64(raw->getOperand(3));
    if (!theta || !phi || !lambda) {
      return false;
    }
    out = u3Matrix(*theta, *phi, *lambda);
    return true;
  }
  if (llvm::isa<qco::POp>(op.getOperation())) {
    auto* raw = op.getOperation();
    if (raw->getNumOperands() < 2) {
      return false;
    }
    auto lambda = evaluateConstF64(raw->getOperand(1));
    if (!lambda) {
      return false;
    }
    out = qco::decomposition::pMatrix(*lambda);
    return true;
  }
  if (llvm::isa<qco::ROp>(op.getOperation())) {
    auto* raw = op.getOperation();
    if (raw->getNumOperands() < 3) {
      return false;
    }
    auto theta = evaluateConstF64(raw->getOperand(1));
    auto phi = evaluateConstF64(raw->getOperand(2));
    if (!theta || !phi) {
      return false;
    }
    const auto thetaSin = std::sin(*theta / 2.0);
    const auto m01 =
        phasedAmplitude(thetaSin, -*phi - (std::numbers::pi / 2.0));
    const auto m10 = phasedAmplitude(thetaSin, *phi - (std::numbers::pi / 2.0));
    const std::complex<double> thetaCos = std::cos(*theta / 2.0);
    out = Matrix2x2::fromElements(thetaCos, m01, m10, thetaCos);
    return true;
  }
  if (Matrix2x2 raw; op.getUnitaryMatrix2x2(raw)) {
    out = raw;
    return true;
  }
  qco::DynamicMatrix dynamic;
  if (!op.getUnitaryMatrixDynamic(dynamic) || dynamic.rows() != 2 ||
      dynamic.cols() != 2) {
    return false;
  }
  out = Matrix2x2::fromElements(dynamic(0, 0), dynamic(0, 1), dynamic(1, 0),
                                dynamic(1, 1));
  return true;
}

/// 4×4 unitary for a two-qubit op (same layout as ``getUnitaryMatrix4x4``).
bool extractTwoQubitMatrix(qco::UnitaryOpInterface op, Matrix4x4& out) {
  if (auto ctrl = llvm::dyn_cast<qco::CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<qco::ZOp>(body)) {
      out = Matrix4x4::identity();
      out(3, 3) = -1.0;
      return true;
    }
    if (llvm::isa<qco::XOp>(body)) {
      out = Matrix4x4::fromElements(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                                    0);
      return true;
    }
    return false;
  }
  if (Matrix4x4 raw; op.getUnitaryMatrix4x4(raw)) {
    out = raw;
    return true;
  }
  qco::DynamicMatrix dynamic;
  if (!op.getUnitaryMatrixDynamic(dynamic) || dynamic.rows() != 4 ||
      dynamic.cols() != 4) {
    return false;
  }
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      out(row, col) = dynamic(static_cast<std::int64_t>(row),
                              static_cast<std::int64_t>(col));
    }
  }
  return true;
}

std::optional<Matrix4x4>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }
  Matrix4x4 unitary = Matrix4x4::identity();
  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t nextQubitId = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto alloc = llvm::dyn_cast<qco::AllocOp>(&rawOp)) {
          if (nextQubitId >= 2) {
            return std::nullopt;
          }
          qubitIds.try_emplace(alloc.getResult(), nextQubitId++);
        }
      }
    }
  }

  auto getQubitId = [&](Value qubit) -> std::optional<std::size_t> {
    auto it = qubitIds.find(qubit);
    if (it == qubitIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        auto op = llvm::dyn_cast<qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (llvm::isa<qco::BarrierOp, qco::GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          const auto qIn = getUnitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = decomposition::expandToTwoQubits(
                        oneQ, static_cast<decomposition::QubitId>(*qid)) *
                    unitary;
          const auto qOut = getUnitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = getUnitaryQubitOperand(op, 0);
          const auto q1In = getUnitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          auto q0id = getQubitId(*q0In);
          auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          // Reorder the gate's (operand0, operand1) layout into the canonical
          // (qubit 0, qubit 1) order used by `unitary`.
          const llvm::SmallVector<decomposition::QubitId, 2> ids{
              static_cast<decomposition::QubitId>(*q0id),
              static_cast<decomposition::QubitId>(*q1id)};
          unitary =
              decomposition::fixTwoQubitMatrixQubitOrder(twoQ, ids) * unitary;
          const auto q0Out = getUnitaryQubitResult(op, 0);
          const auto q1Out = getUnitaryQubitResult(op, 1);
          if (!q0Out || !q1Out) {
            return std::nullopt;
          }
          qubitIds[*q0Out] = *q0id;
          qubitIds[*q1Out] = *q1id;
          continue;
        }
      }
    }
  }

  if (nextQubitId != 2) {
    return std::nullopt;
  }
  return unitary;
}

/// Kronecker-embed ``matrix`` on wire ``q`` into a ``2^N``-dim unitary (same
/// index bit order as QCO 4×4 matrices: wire 0 is the high bit).
TestMatrix expandOneQToN(const Matrix2x2& matrix, std::size_t q,
                         std::size_t numQubits) {
  const std::size_t dim = 1ULL << numQubits;
  TestMatrix full(dim);
  const auto bit = numQubits - 1 - q;
  const std::size_t mask = 1ULL << bit;
  for (std::size_t col = 0; col < dim; ++col) {
    const std::size_t sIn = (col >> bit) & 1ULL;
    const std::size_t rest = col & ~mask;
    for (std::size_t sOut = 0; sOut < 2; ++sOut) {
      const std::size_t row = rest | (sOut << bit);
      full(row, col) = matrix(sOut, sIn);
    }
  }
  return full;
}

/// Embed ``matrix`` on wires ``q0``, ``q1`` into a ``2^N``-dim unitary.
TestMatrix expandTwoQToN(const Matrix4x4& matrix, std::size_t q0,
                         std::size_t q1, std::size_t numQubits) {
  const std::size_t dim = 1ULL << numQubits;
  TestMatrix full(dim);
  const auto bit0 = numQubits - 1 - q0;
  const auto bit1 = numQubits - 1 - q1;
  const std::size_t mask0 = 1ULL << bit0;
  const std::size_t mask1 = 1ULL << bit1;
  const std::size_t maskBoth = mask0 | mask1;
  for (std::size_t col = 0; col < dim; ++col) {
    const std::size_t s0In = (col >> bit0) & 1ULL;
    const std::size_t s1In = (col >> bit1) & 1ULL;
    // 2-bit index for the pair matches QCO 4×4 row/column layout.
    const std::size_t smallIn = (s0In << 1) | s1In;
    const std::size_t rest = col & ~maskBoth;
    for (std::size_t smallOut = 0; smallOut < 4; ++smallOut) {
      const std::size_t s0Out = (smallOut >> 1) & 1ULL;
      const std::size_t s1Out = smallOut & 1ULL;
      const std::size_t row = rest | (s0Out << bit0) | (s1Out << bit1);
      full(row, col) = matrix(smallOut, smallIn);
    }
  }
  return full;
}

/// Full ``2^N`` unitary from a QCO module (``alloc`` / ``static``, 1q/2q
/// unitaries, ``ctrl`` with X/Z body). ``std::nullopt`` on unsupported ops or
/// if ``N`` exceeds ``maxQubits``.
std::optional<TestMatrix>
computeNQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp,
                               std::size_t maxQubits) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }

  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t numQubits = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto alloc = llvm::dyn_cast<qco::AllocOp>(&rawOp)) {
          if (numQubits >= maxQubits) {
            return std::nullopt;
          }
          qubitIds.try_emplace(alloc.getResult(), numQubits++);
        } else if (auto staticOp = llvm::dyn_cast<qco::StaticOp>(&rawOp)) {
          const auto idx = static_cast<std::size_t>(staticOp.getIndex());
          if (idx >= maxQubits) {
            return std::nullopt;
          }
          qubitIds.try_emplace(staticOp.getResult(), idx);
          numQubits = std::max(numQubits, idx + 1);
        }
      }
    }
  }

  if (numQubits == 0) {
    return std::nullopt;
  }

  TestMatrix unitary = TestMatrix::identity(1ULL << numQubits);

  auto getQubitId = [&](Value qubit) -> std::optional<std::size_t> {
    auto it = qubitIds.find(qubit);
    if (it == qubitIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        auto op = llvm::dyn_cast<qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (llvm::isa<qco::BarrierOp, qco::GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          const auto qIn = getUnitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = expandOneQToN(oneQ, *qid, numQubits) * unitary;
          const auto qOut = getUnitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = getUnitaryQubitOperand(op, 0);
          const auto q1In = getUnitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          auto q0id = getQubitId(*q0In);
          auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          unitary = expandTwoQToN(twoQ, *q0id, *q1id, numQubits) * unitary;
          const auto q0Out = getUnitaryQubitResult(op, 0);
          const auto q1Out = getUnitaryQubitResult(op, 1);
          if (!q0Out || !q1Out) {
            return std::nullopt;
          }
          qubitIds[*q0Out] = *q0id;
          qubitIds[*q1Out] = *q1id;
          continue;
        }
      }
    }
  }

  return unitary;
}

} // namespace mlir::qco::native_synth_test
