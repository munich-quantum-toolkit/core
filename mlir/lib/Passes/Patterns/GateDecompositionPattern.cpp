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
#include <cstddef>
#include <iterator>
#include <limits>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <type_traits>
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
   * @param basisGates Set of two-qubit gates that should be used in the
   *                   decomposition. All two-qubit interactions will be
   *                   represented by one of the gates in this set
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
  explicit GateDecompositionPattern(
      mlir::MLIRContext* context, llvm::SmallVector<Gate> basisGates,
      llvm::SmallVector<EulerBasis> eulerBasis, bool singleQubitOnly,
      bool forceApplication, llvm::Statistic& twoQubitCreationTime,
      llvm::Statistic& numberOfTwoQubitCreations,
      llvm::Statistic& successfulSingleQubitDecompositions,
      llvm::Statistic& totalSingleQubitDecompositions,
      llvm::Statistic& successfulTwoQubitDecompositions,
      llvm::Statistic& totalTwoQubitDecompositions,
      llvm::Statistic& totalCircuitCollections,
      llvm::Statistic& totalTouchedGates,
      llvm::Statistic& subCircuitComplexityChange,
      llvm::Statistic& timeInCircuitCollection,
      llvm::Statistic& timeInSingleQubitDecomposition,
      llvm::Statistic& timeInTwoQubitDecomposition)
      : OpInterfaceRewritePattern(context),
        decomposerBasisGates{std::move(basisGates)},
        decomposerEulerBases{std::move(eulerBasis)},
        singleQubitOnly{singleQubitOnly}, forceApplication{forceApplication},
        twoQubitCreationTime{twoQubitCreationTime},
        numberOfTwoQubitCreations{numberOfTwoQubitCreations},
        successfulSingleQubitDecompositions{
            successfulSingleQubitDecompositions},
        totalSingleQubitDecompositions{totalSingleQubitDecompositions},
        successfulTwoQubitDecompositions{successfulTwoQubitDecompositions},
        totalTwoQubitDecompositions{totalTwoQubitDecompositions},
        totalCircuitCollections{totalCircuitCollections},
        totalTouchedGates{totalTouchedGates},
        subCircuitComplexityChange{subCircuitComplexityChange},
        timeInCircuitCollection{timeInCircuitCollection},
        timeInSingleQubitDecomposition{timeInSingleQubitDecomposition},
        timeInTwoQubitDecomposition{timeInTwoQubitDecomposition} {
    ++numberOfTwoQubitCreations;
    auto startTime = std::chrono::steady_clock::now();
    for (auto&& basisGate : decomposerBasisGates) {
      basisDecomposers.push_back(decomposition::TwoQubitBasisDecomposer::create(
          basisGate, DEFAULT_FIDELITY));
    }
    auto endTime = std::chrono::steady_clock::now();
    twoQubitCreationTime +=
        std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                              startTime)
            .count();
    auto&& [singleQubitGates, twoQubitGates] = getDecompositionGates();
    availableSingleQubitGates = std::move(singleQubitGates);
    availableTwoQubitGates = std::move(twoQubitGates);
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

    auto collectSeries = [this](UnitaryOpInterface op, bool singleQubitOnly) {
      ++totalCircuitCollections;
      if (singleQubitOnly) {
        return TwoQubitSeries::getSingleQubitSeries(op);
      }
      return TwoQubitSeries::getTwoQubitSeries(op);
    };
    auto startTime = std::chrono::steady_clock::now();
    auto series = collectSeries(op, singleQubitOnly);
    auto endTime = std::chrono::steady_clock::now();
    timeInCircuitCollection +=
        std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                              startTime)
            .count();
    // not really accurate since it neglects the "past the series" gates that
    // terminated the series
    totalTouchedGates += series.gates.size();

    auto containsForeignGates = !series.containsOnlyGates(
        availableSingleQubitGates, availableTwoQubitGates);

    if (series.gates.empty() || (series.gates.size() < 3 &&
                                 !(forceApplication && containsForeignGates))) {
      // empty or too short and only contains valid gates anyway
      return mlir::failure();
    }

    // sequence and cached complexity to avoid repeated recomputations
    auto bestSequence = std::make_pair(decomposition::TwoQubitGateSequence{},
                                       std::numeric_limits<std::size_t>::max());

    if (series.isSingleQubitSeries()) {
      // only a single-qubit series;
      // single-qubit euler decomposition is more efficient
      const auto unitaryMatrix = series.getSingleQubitUnitaryMatrix();
      if (!unitaryMatrix) {
        // cannot process decomposition without the matrix of the series
        return mlir::failure();
      }
      // only count the multiple decompositions as "one" since the number of
      // euler bases is constant
      ++totalSingleQubitDecompositions;
      startTime = std::chrono::steady_clock::now();
      for (auto&& eulerBasis : decomposerEulerBases) {
        auto sequence = decomposition::EulerDecomposition::generateCircuit(
            eulerBasis, *unitaryMatrix, true, std::nullopt);

        auto newComplexity = sequence.complexity();
        if (newComplexity < bestSequence.second) {
          bestSequence = std::make_pair(std::move(sequence), newComplexity);
        }
      }
      endTime = std::chrono::steady_clock::now();
      timeInSingleQubitDecomposition +=
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime)
              .count();
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

      // only count the multiple decompositions as "one" since the number of
      // euler bases is constant
      ++totalTwoQubitDecompositions;
      startTime = std::chrono::steady_clock::now();
      for (const auto& decomposer : basisDecomposers) {
        auto sequence = decomposer.twoQubitDecompose(
            targetDecomposition, decomposerEulerBases, DEFAULT_FIDELITY, true,
            std::nullopt);
        if (sequence) {
          // decomposition successful
          auto newComplexity = sequence->complexity();
          if (newComplexity < bestSequence.second) {
            // this decomposition is better than any successful decomposition
            // before
            bestSequence = std::make_pair(*sequence, newComplexity);
          }
        }
      }
      endTime = std::chrono::steady_clock::now();
      timeInTwoQubitDecomposition +=
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime)
              .count();
    }

    if (bestSequence.second == std::numeric_limits<std::size_t>::max()) {
      // unable to decompose series
      return mlir::failure();
    }
    if (bestSequence.second >= series.complexity &&
        !(forceApplication && containsForeignGates)) {
      // decomposition is longer/more complex than input series; result will
      // always be used (even if more complex) if forceApplication is set and
      // the input series contained at least one gate unavailable for the
      // decomposition
      return mlir::failure();
    }

    if (series.isSingleQubitSeries()) {
      ++successfulSingleQubitDecompositions;
    } else {
      ++successfulTwoQubitDecompositions;
    }
    subCircuitComplexityChange += series.complexity - bestSequence.second;

    applySeries(rewriter, series, bestSequence.first);

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
  /**
   * Qubit series of MLIR operations involving up to two qubits.
   */
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
        } else {
          llvm::reportFatalInternalError(
              "Gate in series has neither one nor two qubits - decomposition "
              "not possible!");
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
                      const llvm::SetVector<qc::OpType>& twoQubitGates) const {
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
     * If the operation is not valid for a one- or two-qubit series,
     * leave in/out qubits uninitialized and the gate list empty.
     */
    explicit TwoQubitSeries(UnitaryOpInterface initialOperation) {
      if (isBarrier(initialOperation)) {
        // not a valid single- or two-qubit series
        // (barrier cannot be decomposed)
        return;
      }
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
      } else {
        // not a valid single- or two-qubit series (more than two qubits)
        return;
      }
      complexity += helpers::getComplexity(helpers::getQcType(initialOperation),
                                           initialOperation.getNumQubits());
    }

    /**
     * Add a single-qubit operation to the series.
     *
     * @param nextGate Gate to be added, must have exactly one qubit
     *
     * @return true if series continues, otherwise false
     */
    bool appendSingleQubitGate(UnitaryOpInterface nextGate) {
      if (isBarrier(nextGate)) {
        return false;
      }
      auto operand = nextGate.getInputQubit(0);
      // NOLINTNEXTLINE(readability-qualified-auto)
      auto it = llvm::find(outQubits, operand);
      if (it == outQubits.end()) {
        llvm::reportFatalInternalError("Operand of single-qubit op and user of "
                                       "qubit is not in current outQubits");
      }
      QubitId qubitId = std::distance(outQubits.begin(), it);
      *it = nextGate->getResult(0);

      gates.push_back({.op = nextGate, .qubitIds = {qubitId}});
      complexity += helpers::getComplexity(helpers::getQcType(nextGate), 1);
      return true;
    }

    /**
     * Add a two-qubit operation to the series.
     *
     * @param nextGate Gate to be added, must have exactly two qubits
     *
     * @return true if series continues, otherwise false
     */
    bool appendTwoQubitGate(UnitaryOpInterface nextGate) {
      if (isBarrier(nextGate)) {
        // a barrier operation should not be crossed for a decomposition;
        // ignore possibility to backtrack (if this is the first two-qubit gate)
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
          // series already has two qubits, thus it is finished because of this
          // new qubit
          return false;
        }
        auto&& opInQubits = nextGate.getInputQubits();
        // iterator in the operation input of nextGate to "old" qubit that
        // already has previous single-qubit gates in this series
        auto it2 = llvm::find(opInQubits, firstQubitIt != outQubits.end()
                                              ? *firstQubitIt
                                              : *secondQubitIt);
        // operation is a user of the "old" qubit since it was found this way;
        // thus should always succeed
        assert(it2 != opInQubits.end());
        // new qubit ID based on position in outQubits
        const QubitId newInQubitId = std::distance(outQubits.begin(), it);
        // position in operation input; since there are only two qubits, the
        // other in qubit must be the "not old" one
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
      llvm::SmallVector<MlirGate, 4> backtrackedGates;
      auto prependSingleQubitGate = [&](UnitaryOpInterface op) {
        inQubits[qubitId] = op.getInputQubit(0);
        backtrackedGates.push_back({.op = op, .qubitIds = {qubitId}});
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

      gates.insert(gates.begin(),
                   std::make_move_iterator(backtrackedGates.rbegin()),
                   std::make_move_iterator(backtrackedGates.rend()));
    }

    [[nodiscard]] static bool isBarrier(UnitaryOpInterface op) {
      return llvm::isa_and_present<BarrierOp>(op);
    }

    /**
     * Get user (should only be one due to dialect's one-use policy) of given
     * qubit. If the filter returns false for this user, std::nullopt will be
     * returned instead.
     */
    template <typename Func>
    static std::optional<UnitaryOpInterface> getUser(mlir::Value qubit,
                                                     Func&& filter) {
      if (qubit) {
        auto users = qubit.getUsers();
        auto userIt = users.begin();
        if (!qubit.hasOneUse()) {
          llvm::reportFatalUsageError("Qubit has more than one use - unable to "
                                      "collect gate series for decomposition!");
        }
        auto user = mlir::dyn_cast<UnitaryOpInterface>(*userIt);
        if (user && std::invoke(std::forward<Func>(filter), user)) {
          return user;
        }
      }
      return std::nullopt;
    };
  };

  /**
   * Create controlled version of given gate operation type.
   *
   * @param rewriter Rewriter instance to apply modifications
   * @param location Location for the created operations
   * @param ctrlQubits Qubits that serve as controls
   * @param inQubitsAndParams Qubits and parameters for inner gate
   *                          (as required by the builder of the gate);
   *                          all qubits must be of type mlir::Value
   *                          and all parameters must have another type
   */
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
        rewriter.create<OpType>(location,
                                std::forward<Args>(inQubitsAndParams)...));
  }

  /**
   * Replace given series by given sequence.
   * This is done using the rewriter to create the MLIR operations described by
   * the sequence between the input and output qubits of the series and then
   * deleting all gates of the series.
   */
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
            llvm::reportFatalInternalError(
                "Invalid number of qubit IDs while trying to apply "
                "decomposition result to MLIR!");
          }
        };

    if (sequence.hasGlobalPhase()) {
      rewriter.create<GPhaseOp>(location, sequence.globalPhase);
    }

#ifndef NDEBUG
    Eigen::Matrix4cd unitaryMatrix = Eigen::Matrix4cd::Identity();
#endif // NDEBUG

    auto addSingleQubitRotationGate = [&](auto&& gate) {
      assert(gate.qubitId.size() == 1);
      UnitaryOpInterface newGate;
      if (gate.type == qc::RX) {
        newGate = rewriter.create<RXOp>(location, inQubits[gate.qubitId[0]],
                                        gate.parameter[0]);
      } else if (gate.type == qc::RY) {
        newGate = rewriter.create<RYOp>(location, inQubits[gate.qubitId[0]],
                                        gate.parameter[0]);
      } else if (gate.type == qc::RZ) {
        newGate = rewriter.create<RZOp>(location, inQubits[gate.qubitId[0]],
                                        gate.parameter[0]);
      } else {
        llvm::reportFatalInternalError(
            "Unknown single-qubit rotation gate while applying decomposition!");
      }
#ifndef NDEBUG
      unitaryMatrix = decomposition::expandToTwoQubits(
                          newGate.getUnitaryMatrix<Eigen::Matrix2cd>().value(),
                          gate.qubitId[0]) *
                      unitaryMatrix;
#endif // NDEBUG
      return newGate;
    };

    for (auto&& gate : sequence.gates) {
      // these if branches should handle all gates in availableSingleQubitGates
      // and availableTwoQubitGates; additional gates will need to be added when
      // using new euler bases or basis gates
      if (gate.type == qc::X && gate.qubitId.size() > 1) {
        // X gate involving more than one qubit is a CX gate:
        // qubit position 0 is target, 1 is control
        auto newGate = createControlledGate<XOp>(rewriter, location,
                                                 {inQubits[gate.qubitId[1]]},
                                                 inQubits[gate.qubitId[0]]);
#ifndef NDEBUG
        unitaryMatrix = decomposition::fixTwoQubitMatrixQubitOrder(
                            newGate.getUnitaryMatrix().value(), gate.qubitId) *
                        unitaryMatrix;
#endif // NDEBUG
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RX || gate.type == qc::RY ||
                 gate.type == qc::RZ) {
        auto newGate = addSingleQubitRotationGate(gate);
        updateInQubits(gate.qubitId, newGate);
      } else {
        llvm::reportFatalInternalError("Unsupported gate type in decomposition "
                                       "while applying result to MLIR!");
      }
    }
    assert((unitaryMatrix * helpers::globalPhaseFactor(sequence.globalPhase))
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

  /**
   * Get all gates that are potentially in the circuit after the decomposition.
   * This is based on the euler bases and basis gates passed to the constructor.
   *
   * @return Array with the following two elements:
   *          * All possible single-qubit gate types
   *          * All possible two-qubit gate types
   */
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
  // available basis gates
  llvm::SmallVector<Gate> decomposerBasisGates;
  // available euler bases
  llvm::SmallVector<EulerBasis> decomposerEulerBases;

  // cached basis decomposers; one for each basis gate
  llvm::SmallVector<decomposition::TwoQubitBasisDecomposer, 0> basisDecomposers;

  // cached result of getDecompositionGates()
  llvm::SetVector<qc::OpType> availableSingleQubitGates;
  llvm::SetVector<qc::OpType> availableTwoQubitGates;

  // configuration of pattern
  bool singleQubitOnly;
  bool forceApplication;

  llvm::Statistic& twoQubitCreationTime;
  llvm::Statistic& numberOfTwoQubitCreations;
  llvm::Statistic& successfulSingleQubitDecompositions;
  llvm::Statistic& totalSingleQubitDecompositions;
  llvm::Statistic& successfulTwoQubitDecompositions;
  llvm::Statistic& totalTwoQubitDecompositions;
  llvm::Statistic& totalCircuitCollections;
  llvm::Statistic& totalTouchedGates;
  llvm::Statistic& subCircuitComplexityChange;
  llvm::Statistic& timeInCircuitCollection;
  llvm::Statistic& timeInSingleQubitDecomposition;
  llvm::Statistic& timeInTwoQubitDecomposition;
};

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(
    mlir::RewritePatternSet& patterns, llvm::Statistic& twoQubitCreationTime,
    llvm::Statistic& numberOfTwoQubitCreations,
    llvm::Statistic& successfulSingleQubitDecompositions,
    llvm::Statistic& totalSingleQubitDecompositions,
    llvm::Statistic& successfulTwoQubitDecompositions,
    llvm::Statistic& totalTwoQubitDecompositions,
    llvm::Statistic& totalCircuitCollections,
    llvm::Statistic& totalTouchedGates,
    llvm::Statistic& subCircuitComplexityChange,
    llvm::Statistic& timeInCircuitCollection,
    llvm::Statistic& timeInSingleQubitDecomposition,
    llvm::Statistic& timeInTwoQubitDecomposition) {
  llvm::SmallVector<GateDecompositionPattern::Gate> basisGates;
  llvm::SmallVector<GateDecompositionPattern::EulerBasis> eulerBases;
  basisGates.push_back({.type = qc::X, .parameter = {}, .qubitId = {0, 1}});
  basisGates.push_back({.type = qc::X, .parameter = {}, .qubitId = {1, 0}});
  eulerBases.push_back(GateDecompositionPattern::EulerBasis::ZYZ);
  eulerBases.push_back(GateDecompositionPattern::EulerBasis::XYX);
  eulerBases.push_back(GateDecompositionPattern::EulerBasis::ZXZ);
  patterns.add<GateDecompositionPattern>(
      patterns.getContext(), basisGates, eulerBases, false, true,
      twoQubitCreationTime, numberOfTwoQubitCreations,
      successfulSingleQubitDecompositions, totalSingleQubitDecompositions,
      successfulTwoQubitDecompositions, totalTwoQubitDecompositions,
      totalCircuitCollections, totalTouchedGates, subCircuitComplexityChange,
      timeInCircuitCollection, timeInSingleQubitDecomposition,
      timeInTwoQubitDecomposition);
}

} // namespace mlir::qco
