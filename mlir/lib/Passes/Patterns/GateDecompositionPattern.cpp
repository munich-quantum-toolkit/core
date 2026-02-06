/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
   *
   * @param context MLIR context in which the pattern is applied
   * @param basisGate Set of two-qubit gates that should be used in the
   *                  decomposition. All two-qubit interactions will be
   *                  represented by one of the gates in this set
   * @param eulerBasis Set of euler bases that should be used for the
   *                   decomposition of local single-qubit modifications. For
   *                   each necessary single-qubit operation, the optimal basis
   *                   will be chosen from this set
   * @param singleQubitOnly If true, only perform single-qubit decompositions
   *                        and no two-qubit decompositions
   * @param forceApplication If true, always apply best decomposition, even if
   *                         it is longer/more complex than the previous
   *                         circuit. To prevent recursion, this will not apply
   *                         a decomposition if the (sub)circuit only contains
   *                         gates available as basis gates or euler bases
   */
  explicit GateDecompositionPattern(mlir::MLIRContext* context,
                                    llvm::SmallVector<Gate> basisGate,
                                    llvm::SmallVector<EulerBasis> eulerBasis,
                                    bool singleQubitOnly, bool forceApplication)
      : OpInterfaceRewritePattern(context),
        decomposerBasisGates{std::move(basisGate)},
        decomposerEulerBases{std::move(eulerBasis)},
        singleQubitOnly{singleQubitOnly}, forceApplication{forceApplication} {
    for (auto&& basisGate : decomposerBasisGates) {
      basisDecomposers.push_back(decomposition::TwoQubitBasisDecomposer::create(
          basisGate, DEFAULT_FIDELITY));
    }
  }

  mlir::LogicalResult
  matchAndRewrite(UnitaryOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    if (op->getParentOfType<CtrlOp>()) {
      // application of pattern might not work on gates inside a control
      // modifier because rotation gates need to create new constants which are
      // not allowed inside a control body; also, the foreign gate detection
      // does not work and e.g. a CNOT will not be recognized as such and thus
      // will be further decomposed into a RX gate inside the control body which
      // is most likely undesired
      return mlir::failure();
    }

    auto collectSeries = [](UnitaryOpInterface op, bool singleQubitOnly) {
      if (singleQubitOnly) {
        return TwoQubitSeries::getSingleQubitSeries(op);
      }
      return TwoQubitSeries::getTwoQubitSeries(op);
    };
    auto series = collectSeries(op, singleQubitOnly);

    auto&& [singleQubitGates, twoQubitGates] = getDecompositionGates();
    auto containsForeignGates =
        !series.containsOnlyGates(singleQubitGates, twoQubitGates);

    if (series.gates.empty() || (series.gates.size() < 3 &&
                                 !(forceApplication && containsForeignGates))) {
      // empty or too short and only contains valid gates anyway
      return mlir::failure();
    }

    std::optional<decomposition::TwoQubitGateSequence> bestSequence;

    if (series.isSingleQubitSeries()) {
      // only a single-qubit series;
      // single-qubit euler decomposition is more efficient
      const auto unitaryMatrix = series.getSingleQubitUnitaryMatrix();
      if (!unitaryMatrix) {
        // cannot process decomposition without the matrix of the series
        return mlir::failure();
      }
      for (auto&& eulerBasis : decomposerEulerBases) {
        auto sequence = decomposition::EulerDecomposition::generateCircuit(
            eulerBasis, *unitaryMatrix, true, std::nullopt);
        if (!bestSequence ||
            sequence.complexity() < bestSequence->complexity()) {
          bestSequence = sequence;
        }
      }
    } else {
      // two-qubit series; perform two-qubit basis decomposition
      const auto unitaryMatrix = series.getUnitaryMatrix();
      if (!unitaryMatrix) {
        // cannot process decomposition without the matrix of the series
        return mlir::failure();
      }
      const auto targetDecomposition =
          decomposition::TwoQubitWeylDecomposition::create(*unitaryMatrix,
                                                           DEFAULT_FIDELITY);

      for (const auto& decomposer : basisDecomposers) {
        auto sequence = decomposer.twoQubitDecompose(
            targetDecomposition, decomposerEulerBases, DEFAULT_FIDELITY, false,
            std::nullopt);
        if (sequence) {
          // decomposition successful
          if (!bestSequence ||
              sequence->complexity() < bestSequence->complexity()) {
            // this decomposition is better than any successful decomposition
            // before
            bestSequence = sequence;
          }
        }
      }
    }

    llvm::errs() << "Found series (" << series.complexity << "): ";
    for (auto&& gate : series.gates) {
      llvm::errs() << gate.op->getName().stripDialect().str() << ", ";
    }

    if (!bestSequence) {
      return mlir::failure();
    }
    llvm::errs() << "\nDecomposition (" << bestSequence->complexity() << "): ";
    for (auto&& gate : bestSequence->gates) {
      llvm::errs() << qc::toString(gate.type) << ", ";
    }
    llvm::errs() << "\n";
    // only accept new sequence if it shortens existing series by more than two
    // gates; this prevents an oscillation with phase gates
    if (bestSequence->complexity() + 2 >= series.complexity &&
        !(forceApplication && containsForeignGates)) {
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
     */
    std::array<mlir::Value, 2> inQubits{};
    /**
     * Qubits that are the output for the series.
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
    getSingleQubitSeries(UnitaryOpInterface op) {
      if (isBarrier(op) || !op.isSingleQubit()) {
        return {};
      }
      TwoQubitSeries result(op);

      while (auto user = getUser(result.outQubits[0],
                                 &UnitaryOpInterface::isSingleQubit)) {
        if (!result.appendSingleQubitGate(*user)) {
          break;
        }
      }

      assert(result.isSingleQubitSeries());
      return result;
    }

    [[nodiscard]] static TwoQubitSeries
    getTwoQubitSeries(UnitaryOpInterface op) {
      if (isBarrier(op)) {
        return {};
      }
      TwoQubitSeries result(op);

      bool foundGate = true;
      while (foundGate) {
        foundGate = false;
        // collect all available single-qubit operations
        for (std::size_t i = 0; i < result.outQubits.size(); ++i) {
          while (auto user = getUser(result.outQubits[i],
                                     &UnitaryOpInterface::isSingleQubit)) {
            foundGate = result.appendSingleQubitGate(*user);
            if (!foundGate) {
              // result.outQubits was not updated, prevent endless loop
              break;
            }
          }
        }

        for (std::size_t i = 0; i < result.outQubits.size(); ++i) {
          if (auto user = getUser(result.outQubits[i],
                                  &UnitaryOpInterface::isTwoQubit)) {
            foundGate = result.appendTwoQubitGate(*user);
            break; // go back to single-qubit collection
          }
        }
      }
      return result;
    }

    [[nodiscard]] std::optional<Eigen::Matrix2cd>
    getSingleQubitUnitaryMatrix() {
      Eigen::Matrix2cd unitaryMatrix = Eigen::Matrix2cd::Identity();
      for (auto&& gate : gates) {
        if (auto gateMatrix = gate.op.getUnitaryMatrix<Eigen::Matrix2cd>()) {
          unitaryMatrix = *gateMatrix * unitaryMatrix;
        } else {
          return std::nullopt;
        }
      }

      assert(helpers::isUnitaryMatrix(unitaryMatrix));
      return unitaryMatrix;
    }

    [[nodiscard]] std::optional<Eigen::Matrix4cd> getUnitaryMatrix() {
      Eigen::Matrix4cd unitaryMatrix = Eigen::Matrix4cd::Identity();
      Eigen::Matrix4cd gateMatrix;
      for (auto&& gate : gates) {
        if (gate.op.isSingleQubit()) {
          assert(gate.qubitIds.size() == 1);
          auto matrix = gate.op.getUnitaryMatrix<Eigen::Matrix2cd>();
          if (!matrix) {
            return std::nullopt;
          }
          gateMatrix =
              decomposition::expandToTwoQubits(*matrix, gate.qubitIds[0]);
        } else if (gate.op.isTwoQubit()) {
          if (auto matrix = gate.op.getUnitaryMatrix<Eigen::Matrix4cd>()) {
            gateMatrix = decomposition::fixTwoQubitMatrixQubitOrder(
                *matrix, gate.qubitIds);
          } else {
            return std::nullopt;
          }
        }
        unitaryMatrix = gateMatrix * unitaryMatrix;
      }

      assert(helpers::isUnitaryMatrix(unitaryMatrix));
      return unitaryMatrix;
    }

    [[nodiscard]] bool isSingleQubitSeries() const {
      return llvm::is_contained(inQubits, mlir::Value{}) ||
             llvm::is_contained(outQubits, mlir::Value{});
    }

    [[nodiscard]] bool
    containsOnlyGates(const llvm::SetVector<qc::OpType>& singleQubitGates,
                      const llvm::SetVector<qc::OpType>& twoQubitGates) {
      return llvm::all_of(gates, [&](auto&& gate) {
        auto&& gateType = helpers::getQcType(gate.op);
        return (gate.qubitIds.size() == 1 &&
                singleQubitGates.contains(gateType)) ||
               (gate.qubitIds.size() == 2 && twoQubitGates.contains(gateType));
      });
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
      if (initialOperation.isSingleQubit()) {
        inQubits = {initialOperation.getInputQubit(0), mlir::Value{}};
        outQubits = {initialOperation.getOutputQubit(0), mlir::Value{}};
        gates.push_back({.op = initialOperation, .qubitIds = {0}});
      } else if (initialOperation.isTwoQubit()) {
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
      if (isBarrier(nextGate)) {
        // a barrier operation should not be crossed for a decomposition;
        // ignore possitility to backtrack (if this is the first two-qubit gate)
        // since two single-qubit decompositions are less expensive than one
        // two-qubit decomposition
        return false;
      }
      auto&& firstOperand = nextGate.getInputQubit(0);
      auto&& secondOperand = nextGate.getInputQubit(1);
      assert(firstOperand != secondOperand);
      auto firstQubitIt = // NOLINT(readability-qualified-auto)
          llvm::find(outQubits, firstOperand);
      auto secondQubitIt = // NOLINT(readability-qualified-auto)
          llvm::find(outQubits, secondOperand);
      assert(firstQubitIt != secondQubitIt);
      if (firstQubitIt == outQubits.end() || secondQubitIt == outQubits.end()) {
        // another qubit is involved, series is finished (except there only
        // has been one qubit so far)
        auto it = // NOLINT(readability-qualified-auto)
            llvm::find(outQubits, mlir::Value{});
        if (it == outQubits.end()) {
          return false;
        }
        auto&& opInQubits = nextGate.getInputQubits();
        // iterator in the operation input of "old" qubit that already has
        // previous single-qubit gates in this series
        auto it2 = llvm::find(opInQubits, firstQubitIt != outQubits.end()
                                              ? *firstQubitIt
                                              : *secondQubitIt);
        assert(it2 != opInQubits.end());
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
        complexity += helpers::getComplexity(helpers::getQcType(op), 1);
        // outQubits do not need to be updated because the final out qubit is
        // already fixed
      };
      while (auto* op = inQubits[qubitId].getDefiningOp()) {
        auto unitaryOp = mlir::dyn_cast<UnitaryOpInterface>(op);
        if (unitaryOp && unitaryOp.isSingleQubit() && !isBarrier(unitaryOp)) {
          prependSingleQubitGate(unitaryOp);
        } else {
          break;
        }
      }
    }

    [[nodiscard]] static bool isBarrier(UnitaryOpInterface op) {
      return llvm::isa_and_nonnull<BarrierOp>(*op);
    }

    /**
     *
     */
    template <typename Func>
    static std::optional<UnitaryOpInterface> getUser(mlir::Value qubit,
                                                     Func&& filter) {
      if (qubit) {
        auto users = qubit.getUsers();
        auto userIt = users.begin();
        assert(qubit.hasOneUse());
        auto user = mlir::dyn_cast<UnitaryOpInterface>(*userIt);
        if (user && std::invoke(std::forward<Func>(filter), user)) {
          return user;
        }
      }
      return std::nullopt;
    };
  };

  template <typename OpType, typename... Args>
  static OpType createGate(mlir::PatternRewriter& rewriter,
                           mlir::Location location,
                           Args&&... inQubitsAndParams) {
    return rewriter.create<OpType>(location,
                                   std::forward<Args>(inQubitsAndParams)...);
  }

  template <typename OpType, typename... Args>
  static CtrlOp createControlledGate(mlir::PatternRewriter& rewriter,
                                     mlir::Location location,
                                     mlir::ValueRange ctrlQubits,
                                     Args&&... inQubitsAndParams) {
    llvm::SmallVector<mlir::Value, 3> inQubits;
    auto collectInQubits = [&inQubits](auto&& x) {
      if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>,
                                   mlir::Value>) {
        // if argument is a qubit, add it to list; otherwise, do nothing
        inQubits.push_back(std::forward<decltype(x)>(x));
      }
    };
    (collectInQubits(inQubitsAndParams), ...);
    return rewriter.create<CtrlOp>(
        location, ctrlQubits, mlir::ValueRange{inQubits},
        createGate<OpType>(rewriter, location,
                           std::forward<Args>(inQubitsAndParams)...));
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
      createGate<GPhaseOp>(rewriter, location, sequence.globalPhase);
    }

    Eigen::Matrix4cd unitaryMatrix = Eigen::Matrix4cd::Identity();
    for (auto&& gate : sequence.gates) {
      // TODO: need to add each basis gate we want to use
      if (gate.type == qc::X && gate.qubitId.size() > 1) {
        // X gate involving more than one qubit is a CX gate:
        // qubit position 0 is target, 1 is control
        auto newGate = createControlledGate<XOp>(rewriter, location,
                                                 {inQubits[gate.qubitId[1]]},
                                                 inQubits[gate.qubitId[0]]);
        unitaryMatrix = decomposition::fixTwoQubitMatrixQubitOrder(
                            newGate.getUnitaryMatrix().value(), gate.qubitId) *
                        unitaryMatrix;
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RX) {
        assert(gate.qubitId.size() == 1);
        auto newGate = createGate<RXOp>(
            rewriter, location, inQubits[gate.qubitId[0]], gate.parameter[0]);
        unitaryMatrix =
            decomposition::expandToTwoQubits(newGate.getUnitaryMatrix().value(),
                                             gate.qubitId[0]) *
            unitaryMatrix;
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RY) {
        assert(gate.qubitId.size() == 1);
        auto newGate = createGate<RYOp>(
            rewriter, location, inQubits[gate.qubitId[0]], gate.parameter[0]);
        unitaryMatrix =
            decomposition::expandToTwoQubits(newGate.getUnitaryMatrix().value(),
                                             gate.qubitId[0]) *
            unitaryMatrix;
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RZ) {
        assert(gate.qubitId.size() == 1);
        auto newGate = createGate<RZOp>(
            rewriter, location, inQubits[gate.qubitId[0]], gate.parameter[0]);
        unitaryMatrix =
            decomposition::expandToTwoQubits(newGate.getUnitaryMatrix().value(),
                                             gate.qubitId[0]) *
            unitaryMatrix;
        updateInQubits(gate.qubitId, newGate);
      } else {
        throw std::runtime_error{"Unknown gate type!"};
      }
    }
    assert((unitaryMatrix * std::exp(C_IM * sequence.globalPhase))
               .isApprox(
                   series.getUnitaryMatrix().value_or(Eigen::Matrix4cd::Zero()),
                   SANITY_CHECK_PRECISION));

    if (series.isSingleQubitSeries()) {
      rewriter.replaceAllUsesWith(series.outQubits[0], inQubits[0]);
    } else {
      rewriter.replaceAllUsesWith(series.outQubits, inQubits);
    }
    for (auto&& gate : llvm::reverse(series.gates)) {
      rewriter.eraseOp(gate.op);
    }
  }

  [[nodiscard]] std::array<llvm::SetVector<qc::OpType>, 2>
  getDecompositionGates() const {
    llvm::SetVector<qc::OpType> eulerBasesGates;
    llvm::SetVector<qc::OpType> basisGates;
    for (auto&& eulerBasis : decomposerEulerBases) {
      eulerBasesGates.insert_range(
          decomposition::getGateTypesForEulerBasis(eulerBasis));
    }
    for (auto&& basisGate : decomposerBasisGates) {
      basisGates.insert(basisGate.type);
    }
    return {eulerBasesGates, basisGates};
  }

private:
  llvm::SmallVector<Gate> decomposerBasisGates;
  llvm::SmallVector<decomposition::TwoQubitBasisDecomposer, 0> basisDecomposers;
  llvm::SmallVector<EulerBasis> decomposerEulerBases;
  bool singleQubitOnly;
  bool forceApplication;
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
                                         eulerBases, false, false);
}

} // namespace mlir::qco
