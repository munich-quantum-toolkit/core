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
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/BasisDecomposer.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/EulerBasis.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/EulerDecomposition.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/GateSequence.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/MQTOpt/Transforms/Decomposition/WeylDecomposition.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

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

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to collect as many operations as possible into a
 *        4x4 unitary matrix and then decompose it into rotation and given basis
 *        gates.
 */
struct GateDecompositionPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {
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
  matchAndRewrite(UnitaryInterface op,
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
      UnitaryInterface op;
      llvm::SmallVector<QubitId, 2> qubitIds;
    };
    llvm::SmallVector<MlirGate, 8> gates;

    [[nodiscard]] static TwoQubitSeries getTwoQubitSeries(UnitaryInterface op) {
      if (isBarrier(op)) {
        return {};
      }
      TwoQubitSeries result(op);

      auto getUser = [](mlir::Value qubit,
                        auto&& filter) -> std::optional<UnitaryInterface> {
        if (qubit) {
          assert(qubit.hasOneUse());
          auto user =
              mlir::dyn_cast<UnitaryInterface>(*qubit.getUsers().begin());
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

    [[nodiscard]] matrix2x2 getSingleQubitUnitaryMatrix() const {
      auto unitaryMatrix = decomposition::IDENTITY_GATE;
      for (auto&& gate : gates) {
        auto gateMatrix = decomposition::getSingleQubitMatrix(
            {.type = helpers::getQcType(gate.op),
             .parameter = helpers::getParameters(gate.op),
             .qubitId = gate.qubitIds});
        unitaryMatrix = gateMatrix * unitaryMatrix;
      }

      assert(helpers::isUnitaryMatrix(unitaryMatrix));
      return unitaryMatrix;
    }

    [[nodiscard]] matrix4x4 getUnitaryMatrix() const {
      matrix4x4 unitaryMatrix = helpers::kroneckerProduct(
          decomposition::IDENTITY_GATE, decomposition::IDENTITY_GATE);
      for (auto&& gate : gates) {
        auto gateMatrix = decomposition::getTwoQubitMatrix(
            {.type = helpers::getQcType(gate.op),
             .parameter = helpers::getParameters(gate.op),
             .qubitId = gate.qubitIds});
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
    explicit TwoQubitSeries(UnitaryInterface initialOperation) {
      auto&& in = initialOperation.getAllInQubits();
      auto&& out = initialOperation->getResults();
      if (helpers::isSingleQubitOperation(initialOperation)) {
        inQubits = {in[0], mlir::Value{}};
        outQubits = {out[0], mlir::Value{}};
        gates.push_back({.op = initialOperation, .qubitIds = {0}});
      } else if (helpers::isTwoQubitOperation(initialOperation)) {
        inQubits = {in[0], in[1]};
        outQubits = {out[0], out[1]};
        gates.push_back({.op = initialOperation, .qubitIds = {0, 1}});
      }
      complexity += helpers::getComplexity(helpers::getQcType(initialOperation),
                                           in.size());
    }

    /**
     * @return true if series continues, otherwise false
     *         (will always return true)
     */
    bool appendSingleQubitGate(UnitaryInterface nextGate) {
      if (isBarrier(nextGate)) {
        return false;
      }
      auto operand = nextGate.getAllInQubits()[0];
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
    bool appendTwoQubitGate(UnitaryInterface nextGate) {
      auto opInQubits = nextGate.getAllInQubits();
      auto&& firstOperand = opInQubits[0];
      auto&& secondOperand = opInQubits[1];
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
      auto prependSingleQubitGate = [&](UnitaryInterface op) {
        inQubits[qubitId] = op.getAllInQubits()[0];
        gates.insert(gates.begin(), {.op = op, .qubitIds = {qubitId}});
        // outQubits do not need to be updated because the final out qubit is
        // already fixed
      };
      while (auto* op = inQubits[qubitId].getDefiningOp()) {
        auto unitaryOp = mlir::dyn_cast<UnitaryInterface>(op);
        if (unitaryOp && helpers::isSingleQubitOperation(unitaryOp) &&
            !isBarrier(unitaryOp)) {
          prependSingleQubitGate(unitaryOp);
        } else {
          break;
        }
      }
    }

    [[nodiscard]] static bool isBarrier(UnitaryInterface op) {
      return llvm::isa_and_nonnull<BarrierOp>(*op);
    }
  };

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
                                       mlir::Location location, fp parameter,
                                       mlir::ValueRange inQubits) {
    auto parameterValue = rewriter.create<mlir::arith::ConstantOp>(
        location, rewriter.getF64Type(), rewriter.getF64FloatAttr(parameter));

    return rewriter.create<OpType>(
        location, inQubits.getType(), mlir::TypeRange{}, mlir::TypeRange{},
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
        mlir::ValueRange{parameterValue}, inQubits, mlir::ValueRange{},
        mlir::ValueRange{});
  }

  template <typename OpType>
  static OpType createGate(mlir::PatternRewriter& rewriter,
                           mlir::Location location, mlir::ValueRange inQubits,
                           mlir::ValueRange ctrlQubits,
                           const llvm::SmallVector<fp, 3>& parameters) {
    mlir::SmallVector<mlir::Value, 2> parameterValues;
    for (auto&& parameter : parameters) {
      auto parameterValue = rewriter.create<mlir::arith::ConstantOp>(
          location, rewriter.getF64Type(), rewriter.getF64FloatAttr(parameter));
      parameterValues.push_back(parameterValue);
    }

    return rewriter.create<OpType>(
        location, inQubits.getType(), ctrlQubits.getType(), mlir::TypeRange{},
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{}, parameterValues,
        inQubits, ctrlQubits, mlir::ValueRange{});
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
          // TODO: need to handle controls differently?
          auto results = newGate.getAllOutQubits();
          if (qubitIds.size() == 2) {
            inQubits[qubitIds[0]] = results[0];
            inQubits[qubitIds[1]] = results[1];
          } else if (qubitIds.size() == 1) {
            inQubits[qubitIds[0]] = results[0];
          } else {
            throw std::logic_error{"Invalid number of qubit IDs!"};
          }
        };

    if (sequence.hasGlobalPhase()) {
      createOneParameterGate<GPhaseOp>(rewriter, location, sequence.globalPhase,
                                       {});
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
          // controls come first
          inCtrlQubits.push_back(inQubits[gate.qubitId[1]]);
        }
        auto newGate =
            createGate<XOp>(rewriter, location, {inQubits[gate.qubitId[0]]},
                            inCtrlQubits, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RX) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RXOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RY) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RYOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RZ) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RZOp>(rewriter, location, qubits, {}, gate.parameter);
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

} // namespace mqt::ir::opt
