/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Passes/Decomposition/BasisDecomposer.h"
#include "mlir/Passes/Decomposition/EulerBasis.h"
#include "mlir/Passes/Decomposition/EulerDecomposition.h"
#include "mlir/Passes/Decomposition/Gate.h"
#include "mlir/Passes/Decomposition/GateSequence.h"
#include "mlir/Passes/Decomposition/Helpers.h"
#include "mlir/Passes/Decomposition/UnitaryMatrices.h"
#include "mlir/Passes/Decomposition/WeylDecomposition.h"
#include "mlir/Passes/Passes.h"

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <utility>

namespace mlir::qco {

/**
 * @brief This pattern attempts to collect as many operations as possible into a
 *        4x4 unitary matrix and then decompose it into rotation and given basis
 *        gates.
 */
struct GateDecompositionPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryOpInterface> {
  using EulerBasis = decomposition::EulerBasis;
  using Gate = decomposition::Gate;

  /**
   * Initialize pattern with a set of basis gates and euler bases.
   * The best combination of (basis gate, euler basis) will be evaluated for
   * each decomposition.
   */
  explicit GateDecompositionPattern(mlir::MLIRContext* context,
                                    llvm::SmallVector<Gate> basisGate,
                                    llvm::SmallVector<EulerBasis> eulerBasis)
      : OpInterfaceRewritePattern(context),
        decomposerBasisGate{std::move(basisGate)},
        decomposerEulerBases{std::move(eulerBasis)} {
    for (auto&& basisGate : decomposerBasisGate) {
      basisDecomposers.push_back(decomposition::TwoQubitBasisDecomposer::create(
          basisGate, DEFAULT_FIDELITY));
    }
  }

  mlir::LogicalResult
  matchAndRewrite(UnitaryOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    auto series = TwoQubitSeries::getTwoQubitSeries(op);

    if (series.gates.size() < 3) {
      // too short
      return mlir::failure();
    }

    std::optional<decomposition::TwoQubitGateSequence> bestSequence;

    if (series.isSingleQubitSeries()) {
      // only a single-qubit series;
      // single-qubit euler decomposition is more efficient
      const matrix2x2 unitaryMatrix = series.getSingleQubitUnitaryMatrix();
      for (auto&& eulerBasis : decomposerEulerBases) {
        auto sequence = decomposition::EulerDecomposition::generateCircuit(
            eulerBasis, unitaryMatrix, true, std::nullopt);
        if (!bestSequence ||
            sequence.complexity() < bestSequence->complexity()) {
          bestSequence = sequence;
        }
      }
    } else {
      // two-qubit series; perform two-qubit basis decomposition
      const matrix4x4 unitaryMatrix = series.getUnitaryMatrix();
      const auto targetDecomposition =
          decomposition::TwoQubitWeylDecomposition::create(unitaryMatrix,
                                                           DEFAULT_FIDELITY);

      for (const auto& decomposer : basisDecomposers) {
        auto sequence = decomposer.twoQubitDecompose(
            targetDecomposition, decomposerEulerBases, DEFAULT_FIDELITY, false,
            std::nullopt);
        if (sequence) {
          if (!bestSequence ||
              sequence->complexity() < bestSequence->complexity()) {
            bestSequence = sequence;
          }
        }
      }
    }
    if (!bestSequence) {
      return mlir::failure();
    }
    // only accept new sequence if it shortens existing series by more than two
    // gates; this prevents an oscillation with phase gates
    if (bestSequence->complexity() + 2 >= series.complexity) {
      return mlir::failure();
    }

    applySeries(rewriter, series, *bestSequence);

    return mlir::success();
  }

protected:
  /**
   * Factor by which two matrices are considered to be the same when simplifying
   * during a decomposition.
   */
  static constexpr auto DEFAULT_FIDELITY = 1.0 - 1e-15;
  static constexpr auto SANITY_CHECK_PRECISION =
      decomposition::SANITY_CHECK_PRECISION;

  using QubitId = decomposition::QubitId;
  struct TwoQubitSeries {
    /**
     * Complexity of series using getComplexity() for each gate.
     */
    std::size_t complexity{0};
    /**
     * Qubits that are the input for the series.
     * First qubit will always be set, second qubit may be equal to
     * mlir::Value{} if the series consists of only single-qubit gates.
     *
     * All
     */
    std::array<mlir::Value, 2> inQubits{};
    /**
     * Qubits that are the input for the series.
     * First qubit will always be set, second qubit may be equal to
     * mlir::Value{} if the series consists of only single-qubit gates.
     */
    std::array<mlir::Value, 2> outQubits{};

    struct MlirGate {
      UnitaryOpInterface op;
      llvm::SmallVector<QubitId, 2> qubitIds;
    };
    llvm::SmallVector<MlirGate, 8> gates;

    [[nodiscard]] static TwoQubitSeries
    getTwoQubitSeries(UnitaryOpInterface op) {
      if (isBarrier(op)) {
        return {};
      }
      TwoQubitSeries result(op);

      auto getUser = [](mlir::Value qubit,
                        auto&& filter) -> std::optional<UnitaryOpInterface> {
        if (qubit) {
          auto userIt = qubit.getUsers().begin();
          // qubit may have more than one use if it is in a ctrl block (one use
          // for gate, one use for ctrl); we want to use the ctrl operation
          // since it is relevant for the total unitary matrix of the circuit
          assert(qubit.hasOneUse() || qubit.hasNUses(2));
          if (!qubit.hasOneUse()) {
            // TODO: use wire iterator for proper handling
            while (!mlir::dyn_cast<CtrlOp>(*userIt)) {
              ++userIt;
            }
          }
          auto user = mlir::dyn_cast<UnitaryOpInterface>(*userIt);
          if (user && filter(user)) {
            return user;
          }
        }
        return std::nullopt;
      };

      bool foundGate = true;
      while (foundGate) {
        foundGate = false;
        // collect all available single-qubit operations
        for (std::size_t i = 0; i < result.outQubits.size(); ++i) {
          while (auto user = getUser(result.outQubits[i],
                                     &helpers::isSingleQubitOperation)) {
            foundGate = result.appendSingleQubitGate(*user);
          }
        }

        for (std::size_t i = 0; i < result.outQubits.size(); ++i) {
          if (auto user =
                  getUser(result.outQubits[i], &helpers::isTwoQubitOperation)) {
            foundGate = result.appendTwoQubitGate(*user);
            break; // go back to single-qubit collection
          }
        }
      }
      return result;
    }

    [[nodiscard]] matrix2x2 getSingleQubitUnitaryMatrix() {
      auto unitaryMatrix = decomposition::IDENTITY_GATE;
      for (auto&& gate : gates) {
        // auto gateMatrix = gate.op.getFastUnitaryMatrix<matrix2x2>();
        auto gateMatrix = gate.op.getUnitaryMatrix();
        unitaryMatrix = gateMatrix * unitaryMatrix;
      }

      assert(helpers::isUnitaryMatrix(unitaryMatrix));
      return unitaryMatrix;
    }

    [[nodiscard]] matrix4x4 getUnitaryMatrix() {
      matrix4x4 unitaryMatrix = helpers::kroneckerProduct(
          decomposition::IDENTITY_GATE, decomposition::IDENTITY_GATE);
      for (auto&& gate : gates) {
        auto gateMatrix = gate.op.getUnitaryMatrix();
        if (gate.op.isSingleQubit()) {
          assert(gate.qubitIds.size() == 1);
          // TODO: use helpers::kroneckerProduct or Eigen::kroneckerProduct?
          if (gate.qubitIds[0] == 0) {
            gateMatrix = Eigen::kroneckerProduct(decomposition::IDENTITY_GATE,
                                                 gateMatrix);
          } else if (gate.qubitIds[0] == 1) {
            gateMatrix = Eigen::kroneckerProduct(gateMatrix,
                                                 decomposition::IDENTITY_GATE);
          }
        }
        // auto gateMatrix = gate.op.getFastUnitaryMatrix<matrix4x4>();
        unitaryMatrix = gateMatrix * unitaryMatrix;
      }

      assert(helpers::isUnitaryMatrix(unitaryMatrix));
      return unitaryMatrix;
    }

    [[nodiscard]] bool isSingleQubitSeries() const {
      return llvm::is_contained(inQubits, mlir::Value{}) ||
             llvm::is_contained(outQubits, mlir::Value{});
    }

  private:
    /**
     * Initialize empty TwoQubitSeries instance.
     * New operations can *NOT* be added when calling this constructor overload.
     */
    TwoQubitSeries() = default;
    /**
     * Initialize TwoQubitSeries instance with given first operation.
     */
    explicit TwoQubitSeries(UnitaryOpInterface initialOperation) {
      if (helpers::isSingleQubitOperation(initialOperation)) {
        inQubits = {initialOperation.getInputQubit(0), mlir::Value{}};
        outQubits = {initialOperation.getOutputQubit(0), mlir::Value{}};
        gates.push_back({.op = initialOperation, .qubitIds = {0}});
      } else if (helpers::isTwoQubitOperation(initialOperation)) {
        inQubits = {initialOperation.getInputQubit(0),
                    initialOperation.getInputQubit(1)};
        outQubits = {initialOperation.getOutputQubit(0),
                     initialOperation.getOutputQubit(1)};
        gates.push_back({.op = initialOperation, .qubitIds = {0, 1}});
      }
      complexity += helpers::getComplexity(helpers::getQcType(initialOperation),
                                           initialOperation.getNumQubits());
    }

    /**
     * @return true if series continues, otherwise false
     *         (will always return true)
     */
    bool appendSingleQubitGate(UnitaryOpInterface nextGate) {
      if (isBarrier(nextGate)) {
        return false;
      }
      auto operand = nextGate.getInputQubit(0);
      // NOLINTNEXTLINE(readability-qualified-auto)
      auto it = llvm::find(outQubits, operand);
      if (it == outQubits.end()) {
        throw std::logic_error{"Operand of single-qubit op and user of "
                               "qubit is not in current outQubits"};
      }
      QubitId qubitId = std::distance(outQubits.begin(), it);
      *it = nextGate->getResult(0);

      gates.push_back({.op = nextGate, .qubitIds = {qubitId}});
      complexity += helpers::getComplexity(helpers::getQcType(nextGate), 1);
      return true;
    }

    /**
     * @return true if series continues, otherwise false
     */
    bool appendTwoQubitGate(UnitaryOpInterface nextGate) {
      auto&& firstOperand = nextGate.getInputQubit(0);
      auto&& secondOperand = nextGate.getInputQubit(1);
      auto firstQubitIt = // NOLINT(readability-qualified-auto)
          llvm::find(outQubits, firstOperand);
      auto secondQubitIt = // NOLINT(readability-qualified-auto)
          llvm::find(outQubits, secondOperand);
      if (firstQubitIt == outQubits.end() || secondQubitIt == outQubits.end()) {
        // another qubit is involved, series is finished (except there only
        // has been one qubit so far)
        auto it = // NOLINT(readability-qualified-auto)
            llvm::find(outQubits, mlir::Value{});
        if (it == outQubits.end()) {
          return false;
        }
        // TODO: this only works because parameters are at end of operands;
        // use future getInputQubits() instead
        auto&& opInQubits = nextGate->getOperands();
        // iterator in the operation input of "old" qubit that already has
        // previous single-qubit gates in this series
        auto it2 = llvm::find(opInQubits, firstQubitIt != outQubits.end()
                                              ? *firstQubitIt
                                              : *secondQubitIt);
        // new qubit ID based on position in outQubits
        const QubitId newInQubitId = std::distance(outQubits.begin(), it);
        // position in operation input; since there are only two qubits, it must
        // be the "not old" one
        const QubitId newOpInQubitId =
            1 - std::distance(opInQubits.begin(), it2);

        // update inQubit and update dangling iterator, then proceed as usual
        inQubits[newInQubitId] = opInQubits[newOpInQubitId];
        firstQubitIt = (firstQubitIt != outQubits.end()) ? firstQubitIt : it;
        secondQubitIt = (secondQubitIt != outQubits.end()) ? secondQubitIt : it;

        // before proceeding as usual, see if backtracking on the "new" qubit is
        // possible to collect other single-qubit operations
        backtrackSingleQubitSeries(newInQubitId);
      }
      if (isBarrier(nextGate)) {
        // a barrier operation should not be crossed for a decomposition
        return false;
      }
      const QubitId firstQubitId =
          std::distance(outQubits.begin(), firstQubitIt);
      const QubitId secondQubitId =
          std::distance(outQubits.begin(), secondQubitIt);
      *firstQubitIt = nextGate->getResult(0);
      *secondQubitIt = nextGate->getResult(1);

      gates.push_back(
          {.op = nextGate, .qubitIds = {firstQubitId, secondQubitId}});
      complexity += helpers::getComplexity(helpers::getQcType(nextGate), 2);
      return true;
    }

    /**
     * Traverse single-qubit series back from a given qubit.
     * This is used when a series starts with single-qubit gates and then
     * encounters a two-qubit gate. The second qubit involved in the two-qubit
     * gate could have previous single-qubit operations that can be incorporated
     * in the series.
     */
    void backtrackSingleQubitSeries(QubitId qubitId) {
      auto prependSingleQubitGate = [&](UnitaryOpInterface op) {
        inQubits[qubitId] = op.getInputQubit(0);
        gates.insert(gates.begin(), {.op = op, .qubitIds = {qubitId}});
        // outQubits do not need to be updated because the final out qubit is
        // already fixed
      };
      while (auto* op = inQubits[qubitId].getDefiningOp()) {
        auto unitaryOp = mlir::dyn_cast<UnitaryOpInterface>(op);
        if (unitaryOp && helpers::isSingleQubitOperation(unitaryOp) &&
            !isBarrier(unitaryOp)) {
          prependSingleQubitGate(unitaryOp);
        } else {
          break;
        }
      }
    }

    [[nodiscard]] static bool isBarrier(UnitaryOpInterface op) {
      return llvm::isa_and_nonnull<BarrierOp>(*op);
    }
  };

  template <typename OpType>
  static OpType createGate(mlir::PatternRewriter& rewriter,
                           mlir::Location location, mlir::ValueRange inQubits,
                           const llvm::SmallVector<fp, 3>& parameters) {
    mlir::SmallVector<mlir::Value, 2> parameterValues;
    for (auto&& parameter : parameters) {
      auto parameterValue = rewriter.create<mlir::arith::ConstantOp>(
          location, rewriter.getF64Type(), rewriter.getF64FloatAttr(parameter));
      parameterValues.push_back(parameterValue);
    }

    return rewriter.create<OpType>(location, inQubits, parameterValues);
  }

  template <typename OpType>
  static CtrlOp
  createControlledGate(mlir::PatternRewriter& rewriter, mlir::Location location,
                       mlir::ValueRange inQubits, mlir::ValueRange ctrlQubits,
                       const llvm::SmallVector<fp, 3>& parameters) {
    auto op = createGate<OpType>(rewriter, location, inQubits, parameters);
    auto opOutQubits = op->getResults();
    return rewriter.create<CtrlOp>(location, ctrlQubits, opOutQubits);
  }

  static void applySeries(mlir::PatternRewriter& rewriter,
                          TwoQubitSeries& series,
                          const decomposition::TwoQubitGateSequence& sequence) {
    auto& lastSeriesOp = series.gates.back().op;
    auto location = lastSeriesOp->getLoc();
    rewriter.setInsertionPointAfter(lastSeriesOp);

    auto inQubits = series.inQubits;
    auto updateInQubits =
        [&inQubits](const llvm::SmallVector<QubitId, 2>& qubitIds,
                    auto&& newGate) {
          if (qubitIds.size() == 2) {
            inQubits[qubitIds[0]] = newGate.getOutputQubit(0);
            inQubits[qubitIds[1]] = newGate.getOutputQubit(1);
          } else if (qubitIds.size() == 1) {
            inQubits[qubitIds[0]] = newGate.getOutputQubit(0);
          } else {
            throw std::logic_error{"Invalid number of qubit IDs!"};
          }
        };

    if (sequence.hasGlobalPhase()) {
      createGate<GPhaseOp>(rewriter, location, {}, {sequence.globalPhase});
    }

    matrix4x4 unitaryMatrix = helpers::kroneckerProduct(
        decomposition::IDENTITY_GATE, decomposition::IDENTITY_GATE);
    for (auto&& gate : sequence.gates) {
      auto gateMatrix = decomposition::getTwoQubitMatrix(gate);
      unitaryMatrix = gateMatrix * unitaryMatrix;

      // TODO: need to add each basis gate we want to use
      if (gate.type == qc::X) {
        mlir::SmallVector<mlir::Value, 1> inCtrlQubits;
        if (gate.qubitId.size() > 1) {
          // controls come last
          inCtrlQubits.push_back(inQubits[gate.qubitId[1]]);
        }
        auto newGate = createControlledGate<XOp>(rewriter, location,
                                                 {inQubits[gate.qubitId[0]]},
                                                 inCtrlQubits, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RX) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RXOp>(rewriter, location, qubits, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RY) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RYOp>(rewriter, location, qubits, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RZ) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RZOp>(rewriter, location, qubits, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else {
        throw std::runtime_error{"Unknown gate type!"};
      }
    }
    assert((unitaryMatrix * std::exp(IM * sequence.globalPhase))
               .isApprox(series.getUnitaryMatrix(), SANITY_CHECK_PRECISION));

    if (series.isSingleQubitSeries()) {
      rewriter.replaceAllUsesWith(series.outQubits[0], inQubits[0]);
    } else {
      rewriter.replaceAllUsesWith(series.outQubits, inQubits);
    }
    for (auto&& gate : llvm::reverse(series.gates)) {
      rewriter.eraseOp(gate.op);
    }
  }

private:
  llvm::SmallVector<Gate> decomposerBasisGate;
  llvm::SmallVector<decomposition::TwoQubitBasisDecomposer, 0> basisDecomposers;
  llvm::SmallVector<EulerBasis> decomposerEulerBases;
};

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  llvm::SmallVector<GateDecompositionPattern::Gate> basisGates;
  llvm::SmallVector<GateDecompositionPattern::EulerBasis> eulerBases;
  basisGates.push_back({.type = qc::X, .parameter = {}, .qubitId = {0, 1}});
  basisGates.push_back({.type = qc::X, .parameter = {}, .qubitId = {1, 0}});
  eulerBases.push_back(GateDecompositionPattern::EulerBasis::ZYZ);
  eulerBases.push_back(GateDecompositionPattern::EulerBasis::XYX);
  eulerBases.push_back(GateDecompositionPattern::EulerBasis::ZXZ);
  patterns.add<GateDecompositionPattern>(patterns.getContext(), basisGates,
                                         eulerBases);
}

} // namespace mlir::qco
