/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Helpers.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {
/**
 * @brief This pattern TODO.
 */
struct EulerDecompositionPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit EulerDecompositionPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    if (!helpers::isSingleQubitOperation(op)) {
      return mlir::failure();
    }

    auto series = getSingleQubitSeries(op);
    if (series.size() <= 3) {
      // TODO: find better way to prevent endless optimization loop
      return mlir::failure();
    }

    dd::GateMatrix unitaryMatrix = dd::opToSingleQubitGateMatrix(qc::I);
    for (auto&& gate : series) {
      if (auto gateMatrix = helpers::getUnitaryMatrix(gate)) {
        unitaryMatrix = helpers::multiply(unitaryMatrix, *gateMatrix);
      }
    }

    auto [decomposedGateSchematic, globalPhase] =
        calculateRotationGates(unitaryMatrix);

    // apply global phase
    createOneParameterGate<GPhaseOp>(rewriter, op->getLoc(), globalPhase, {});

    auto newGates = createMlirGates(rewriter, decomposedGateSchematic,
                                    op.getInQubits().front());
    if (!newGates.empty()) {
      // attach new gates by replacing the uses of the last gate of the series
      rewriter.replaceAllOpUsesWith(series.back(), newGates.back());
    } else {
      // gate series is equal to identity; remove it entirely
      rewriter.replaceAllOpUsesWith(series.back(), op->getOperands());
    }

    // delete in reverse order since last use has been replaced and for the
    // others the only use will be deleted before the operation
    for (auto&& gate : llvm::reverse(series)) {
      rewriter.eraseOp(gate);
    }

    return mlir::success();
  }

  [[nodiscard]] static llvm::SmallVector<UnitaryInterface>
  getSingleQubitSeries(UnitaryInterface op) {
    llvm::SmallVector<UnitaryInterface> result = {op};
    while (op->hasOneUse()) {
      op = getNextOperation(op);
      if (op && helpers::isSingleQubitOperation(op)) {
        result.push_back(op);
      } else {
        break;
      }
    }
    return result;
  }

  [[nodiscard]] static UnitaryInterface getNextOperation(UnitaryInterface op) {
    // since there is only one output qubit in single qubit gates, there should
    // only be one user
    assert(op->hasOneUse());
    auto&& users = op->getUsers();
    return llvm::dyn_cast<UnitaryInterface>(*users.begin());
  }

  /**
   * @brief Creates a new rotation gate with no controls.
   *
   * @tparam OpType The type of the operation to be created.
   * @param op The first instance of the rotation gate.
   * @param rewriter The pattern rewriter.
   * @return A new rotation gate.
   */
  template <typename OpType>
  static OpType createOneParameterGate(mlir::PatternRewriter& rewriter,
                                       mlir::Location location,
                                       qc::fp parameter,
                                       mlir::ValueRange inQubits) {
    auto parameterValue = rewriter.create<mlir::arith::ConstantOp>(
        location, rewriter.getF64Type(), rewriter.getF64FloatAttr(parameter));

    return rewriter.create<OpType>(
        location, inQubits.getType(), mlir::TypeRange{}, mlir::TypeRange{},
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
        mlir::ValueRange{parameterValue}, inQubits, mlir::ValueRange{},
        mlir::ValueRange{});
  }

  [[nodiscard]] static llvm::SmallVector<UnitaryInterface, 3> createMlirGates(
      mlir::PatternRewriter& rewriter,
      const llvm::SmallVector<std::pair<qc::OpType, qc::fp>, 3>& schematic,
      mlir::Value inQubit) {
    llvm::SmallVector<UnitaryInterface, 3> result;
    for (auto [type, angle] : schematic) {
      if (type == qc::RZ) {
        auto newRz = createOneParameterGate<RZOp>(rewriter, inQubit.getLoc(),
                                                  angle, {inQubit});
        result.push_back(newRz);
      } else if (type == qc::RY) {
        auto newRy = createOneParameterGate<RYOp>(rewriter, inQubit.getLoc(),
                                                  angle, {inQubit});
        result.push_back(newRy);
      } else {
        throw std::logic_error{"Unable to create MLIR gate in Euler "
                               "Decomposition (unsupported gate)"};
      }
      inQubit = result.back().getOutQubits().front();
    }
    return result;
  }

  /**
   * @note Adapted from circuit_kak() in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
   *
   *       This code is licensed under the Apache License, Version 2.0. You may
   *       obtain a copy of this license in the LICENSE.txt file in the root
   *       directory of this source tree or at
   *       http://www.apache.org/licenses/LICENSE-2.0.
   *
   *       Any modifications or derivative works of this code must retain this
   *       copyright notice, and modified files need to carry a notice
   *       indicating that they have been altered from the originals.
   */
  [[nodiscard]] static std::pair<
      llvm::SmallVector<std::pair<qc::OpType, qc::fp>, 3>, qc::fp>
  calculateRotationGates(dd::GateMatrix unitaryMatrix) {
    constexpr qc::fp angleZeroEpsilon = 1e-12;

    auto remEuclid = [](qc::fp a, qc::fp b) {
      auto r = std::fmod(a, b);
      return (r < 0.0) ? r + std::abs(b) : r;
    };
    // Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to
    // -π
    auto mod2pi = [&](qc::fp angle) -> qc::fp {
      // remEuclid() isn't exactly the same as Python's % operator, but
      // because the RHS here is a constant and positive it is effectively
      // equivalent for this case
      auto wrapped = remEuclid(angle + qc::PI, 2. * qc::PI) - qc::PI;
      if (std::abs(wrapped - qc::PI) < angleZeroEpsilon) {
        return -qc::PI;
      }
      return wrapped;
    };

    auto [theta, phi, lambda, phase] = paramsZyzInner(unitaryMatrix);
    qc::fp globalPhase = phase - ((phi + lambda) / 2.);

    llvm::SmallVector<std::pair<qc::OpType, qc::fp>, 3> gates;
    if (std::abs(theta) < angleZeroEpsilon) {
      lambda += phi;
      lambda = mod2pi(lambda);
      if (std::abs(lambda) > angleZeroEpsilon) {
        gates.push_back({qc::RZ, lambda});
        globalPhase += lambda / 2.0;
      }
      return {gates, globalPhase};
    }

    if (std::abs(theta - qc::PI) < angleZeroEpsilon) {
      globalPhase += phi;
      lambda -= phi;
      phi = 0.0;
    }
    if (std::abs(mod2pi(lambda + qc::PI)) < angleZeroEpsilon ||
        std::abs(mod2pi(phi + qc::PI)) < angleZeroEpsilon) {
      lambda += qc::PI;
      theta = -theta;
      phi += qc::PI;
    }
    lambda = mod2pi(lambda);
    if (std::abs(lambda) > angleZeroEpsilon) {
      globalPhase += lambda / 2.0;
      gates.push_back({qc::RZ, lambda});
    }
    gates.push_back({qc::RY, theta});
    phi = mod2pi(phi);
    if (std::abs(phi) > angleZeroEpsilon) {
      globalPhase += phi / 2.0;
      gates.push_back({qc::RZ, phi});
    }
    return {gates, globalPhase};
  }

  /**
   * @note Adapted from circuit_kak() in the IBM Qiskit framework.
   *       (C) Copyright IBM 2022
   *
   *       This code is licensed under the Apache License, Version 2.0. You may
   *       obtain a copy of this license in the LICENSE.txt file in the root
   *       directory of this source tree or at
   *       http://www.apache.org/licenses/LICENSE-2.0.
   *
   *       Any modifications or derivative works of this code must retain this
   *       copyright notice, and modified files need to carry a notice
   *       indicating that they have been altered from the originals.
   */
  [[nodiscard]] static std::array<qc::fp, 4>
  paramsZyzInner(dd::GateMatrix unitaryMatrix) {
    auto getIndex = [](auto x, auto y) { return (y * 2) + x; };
    auto determinant = [getIndex](auto&& matrix) {
      return (matrix.at(getIndex(0, 0)) * matrix.at(getIndex(1, 1))) -
             (matrix.at(getIndex(1, 0)) * matrix.at(getIndex(0, 1)));
    };

    auto detArg = std::arg(determinant(unitaryMatrix));
    auto phase = 0.5 * detArg;
    auto theta = 2. * std::atan2(std::abs(unitaryMatrix.at(getIndex(1, 0))),
                                 std::abs(unitaryMatrix.at(getIndex(0, 0))));
    auto ang1 = std::arg(unitaryMatrix.at(getIndex(1, 1)));
    auto ang2 = std::arg(unitaryMatrix.at(getIndex(1, 0)));
    auto phi = ang1 + ang2 - detArg;
    auto lam = ang1 - ang2;
    return {theta, phi, lam, phase};
  }
};

/**
 * @brief Populates the given pattern set with patterns for gate elimination.
 *
 * @param patterns The pattern set to populate.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<NegCtrlDecompositionPattern>(patterns.getContext());
  patterns.add<EulerDecompositionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
