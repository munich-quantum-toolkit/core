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

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <cmath>

using namespace mlir;

namespace mlir::qco::native_synth_test {

std::complex<double> phasedAmplitude(const double magnitude,
                                     const double phase) {
  return std::complex<double>(magnitude, 0.0) *
         std::exp(std::complex<double>(0.0, phase));
}

Eigen::Matrix2cd u3Matrix(double theta, double phi, double lambda) {
  using Complex = std::complex<double>;
  const Complex i(0.0, 1.0);
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  const Complex eiphi = std::exp(i * phi);
  const Complex eilambda = std::exp(i * lambda);
  const Complex eiphilambda = std::exp(i * (phi + lambda));

  Eigen::Matrix2cd mat;
  mat << c, -eilambda * s, eiphi * s, eiphilambda * c;
  return mat;
}

bool isUnitary(const Eigen::Matrix2cd& m, const double atol) {
  const auto identity = Eigen::Matrix2cd::Identity();
  return (m * m.adjoint()).isApprox(identity, atol) &&
         (m.adjoint() * m).isApprox(identity, atol);
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
bool extractSingleQubitMatrix(qco::UnitaryOpInterface op,
                              Eigen::Matrix2cd& out) {
  if (auto rz = dyn_cast<qco::RZOp>(op.getOperation())) {
    auto theta = evaluateConstF64(rz.getTheta());
    if (!theta) {
      return false;
    }
    out = qco::decomposition::rzMatrix(*theta);
    return true;
  }
  if (auto rx = dyn_cast<qco::RXOp>(op.getOperation())) {
    auto theta = evaluateConstF64(rx.getTheta());
    if (!theta) {
      return false;
    }
    out = qco::decomposition::rxMatrix(*theta);
    return true;
  }
  if (auto ry = dyn_cast<qco::RYOp>(op.getOperation())) {
    auto theta = evaluateConstF64(ry.getTheta());
    if (!theta) {
      return false;
    }
    out = qco::decomposition::ryMatrix(*theta);
    return true;
  }
  if (auto u = dyn_cast<qco::UOp>(op.getOperation())) {
    auto theta = evaluateConstF64(u.getTheta());
    auto phi = evaluateConstF64(u.getPhi());
    auto lambda = evaluateConstF64(u.getLambda());
    if (!theta || !phi || !lambda) {
      return false;
    }
    out = u3Matrix(*theta, *phi, *lambda);
    return true;
  }
  if (auto p = dyn_cast<qco::POp>(op.getOperation())) {
    auto lambda = evaluateConstF64(p.getTheta());
    if (!lambda) {
      return false;
    }
    out = qco::decomposition::pMatrix(*lambda);
    return true;
  }
  if (auto r = dyn_cast<qco::ROp>(op.getOperation())) {
    auto theta = evaluateConstF64(r.getTheta());
    auto phi = evaluateConstF64(r.getPhi());
    if (!theta || !phi) {
      return false;
    }
    const auto thetaSin = std::sin(*theta / 2.0);
    const auto m01 = phasedAmplitude(thetaSin, -*phi - (llvm::numbers::pi / 2));
    const auto m10 = phasedAmplitude(thetaSin, *phi - (llvm::numbers::pi / 2));
    const std::complex<double> thetaCos = std::cos(*theta / 2.0);
    out = Eigen::Matrix2cd{{thetaCos, m01}, {m10, thetaCos}};
    return true;
  }
  if (op.getUnitaryMatrix2x2(out)) {
    return true;
  }
  auto dynamic = op.getUnitaryMatrix<Eigen::MatrixXcd>();
  if (!dynamic || dynamic->rows() != 2 || dynamic->cols() != 2) {
    return false;
  }
  out = dynamic->template block<2, 2>(0, 0);
  return true;
}

/// 4×4 unitary for a two-qubit op (same layout as ``getUnitaryMatrix4x4``).
bool extractTwoQubitMatrix(qco::UnitaryOpInterface op, Eigen::Matrix4cd& out) {
  if (auto ctrl = dyn_cast<qco::CtrlOp>(op.getOperation())) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary().getOperation();
    if (isa<qco::ZOp>(body)) {
      out = Eigen::Matrix4cd::Identity();
      out(3, 3) = -1.0;
      return true;
    }
    if (isa<qco::XOp>(body)) {
      out << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;
      return true;
    }
    return false;
  }
  if (op.getUnitaryMatrix4x4(out)) {
    return true;
  }
  auto dynamic = op.getUnitaryMatrix<Eigen::MatrixXcd>();
  if (!dynamic || dynamic->rows() != 4 || dynamic->cols() != 4) {
    return false;
  }
  out = *dynamic;
  return true;
}

std::optional<Eigen::Matrix4cd>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }
  Eigen::Matrix4cd unitary = Eigen::Matrix4cd::Identity();
  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t nextQubitId = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto alloc = dyn_cast<qco::AllocOp>(&rawOp)) {
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
        auto op = dyn_cast<qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (isa<qco::BarrierOp, qco::GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          auto qid = getQubitId(op.getInputQubit(0));
          if (!qid) {
            return std::nullopt;
          }
          Eigen::Matrix2cd oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = qco::decomposition::expandToTwoQubits(oneQ, *qid) * unitary;
          qubitIds[op.getOutputQubit(0)] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          auto q0id = getQubitId(op.getInputQubit(0));
          auto q1id = getQubitId(op.getInputQubit(1));
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Eigen::Matrix4cd twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          unitary =
              expandTwoQToN(twoQ, *q0id, *q1id, /*numQubits=*/2) * unitary;
          qubitIds[op.getOutputQubit(0)] = *q0id;
          qubitIds[op.getOutputQubit(1)] = *q1id;
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

/// Kronecker-embed ``m`` on wire ``q`` into a ``2^N``-dim unitary (same index
/// bit order as QCO 4×4 matrices: wire 0 is the high bit).
Eigen::MatrixXcd expandOneQToN(const Eigen::Matrix2cd& m, std::size_t q,
                               std::size_t numQubits) {
  const auto dim = static_cast<Eigen::Index>(1ULL << numQubits);
  Eigen::MatrixXcd full = Eigen::MatrixXcd::Zero(dim, dim);
  const auto bit = numQubits - 1 - q;
  const std::size_t mask = 1ULL << bit;
  for (Eigen::Index col = 0; col < dim; ++col) {
    const auto colIdx = static_cast<std::size_t>(col);
    const std::size_t sIn = (colIdx >> bit) & 1ULL;
    const std::size_t rest = colIdx & ~mask;
    for (std::size_t sOut = 0; sOut < 2; ++sOut) {
      const auto row = static_cast<Eigen::Index>(rest | (sOut << bit));
      full(row, col) =
          m(static_cast<Eigen::Index>(sOut), static_cast<Eigen::Index>(sIn));
    }
  }
  return full;
}

/// Embed ``m`` on wires ``q0``, ``q1`` into a ``2^N``-dim unitary.
Eigen::MatrixXcd expandTwoQToN(const Eigen::Matrix4cd& m, std::size_t q0,
                               std::size_t q1, std::size_t numQubits) {
  const auto dim = static_cast<Eigen::Index>(1ULL << numQubits);
  Eigen::MatrixXcd full = Eigen::MatrixXcd::Zero(dim, dim);
  const auto bit0 = numQubits - 1 - q0;
  const auto bit1 = numQubits - 1 - q1;
  const std::size_t mask0 = 1ULL << bit0;
  const std::size_t mask1 = 1ULL << bit1;
  const std::size_t maskBoth = mask0 | mask1;
  for (Eigen::Index col = 0; col < dim; ++col) {
    const auto colIdx = static_cast<std::size_t>(col);
    const std::size_t s0In = (colIdx >> bit0) & 1ULL;
    const std::size_t s1In = (colIdx >> bit1) & 1ULL;
    // 2-bit index for the pair matches QCO 4×4 row/column layout.
    const std::size_t smallIn = (s0In << 1) | s1In;
    const std::size_t rest = colIdx & ~maskBoth;
    for (std::size_t smallOut = 0; smallOut < 4; ++smallOut) {
      const std::size_t s0Out = (smallOut >> 1) & 1ULL;
      const std::size_t s1Out = smallOut & 1ULL;
      const auto row =
          static_cast<Eigen::Index>(rest | (s0Out << bit0) | (s1Out << bit1));
      full(row, col) = m(static_cast<Eigen::Index>(smallOut),
                         static_cast<Eigen::Index>(smallIn));
    }
  }
  return full;
}

/// Full ``2^N`` unitary from a QCO module (``alloc`` / ``static``, 1q/2q
/// unitaries, ``ctrl`` with X/Z body). ``std::nullopt`` on unsupported ops or
/// if ``N`` exceeds ``maxQubits``.
std::optional<Eigen::MatrixXcd>
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
        if (auto alloc = dyn_cast<qco::AllocOp>(&rawOp)) {
          if (numQubits >= maxQubits) {
            return std::nullopt;
          }
          qubitIds.try_emplace(alloc.getResult(), numQubits++);
        } else if (auto staticOp = dyn_cast<qco::StaticOp>(&rawOp)) {
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

  const auto dim = static_cast<Eigen::Index>(1ULL << numQubits);
  Eigen::MatrixXcd unitary = Eigen::MatrixXcd::Identity(dim, dim);

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
        auto op = dyn_cast<qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (isa<qco::BarrierOp, qco::GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          auto qid = getQubitId(op.getInputQubit(0));
          if (!qid) {
            return std::nullopt;
          }
          Eigen::Matrix2cd oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = expandOneQToN(oneQ, *qid, numQubits) * unitary;
          qubitIds[op.getOutputQubit(0)] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          auto q0id = getQubitId(op.getInputQubit(0));
          auto q1id = getQubitId(op.getInputQubit(1));
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Eigen::Matrix4cd twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          unitary = expandTwoQToN(twoQ, *q0id, *q1id, numQubits) * unitary;
          qubitIds[op.getOutputQubit(0)] = *q0id;
          qubitIds[op.getOutputQubit(1)] = *q1id;
          continue;
        }
      }
    }
  }

  return unitary;
}

} // namespace mlir::qco::native_synth_test
