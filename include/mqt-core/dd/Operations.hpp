/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <cassert>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dd {

/**
 * @brief Get the decision diagram representation of an operation based on its
 * constituent parts.
 *
 * @note This function is only intended for internal use and should not be
 * called directly.
 *
 * @param dd The DD package to use
 * @param type The operation type
 * @param params The operation parameters
 * @param controls The operation controls
 * @param targets The operation targets
 * @return The decision diagram representation of the operation
 */
MatrixDD getStandardOperationDD(Package& dd, qc::OpType type,
                                const std::vector<fp>& params,
                                const qc::Controls& controls,
                                const std::vector<qc::Qubit>& targets);

/**
 * @brief Get the decision diagram representation of a @ref
 * qc::StandardOperation.
 *
 * @note This function is only intended for internal use and should not be
 * called directly.
 *
 * @param op The operation to get the DD for
 * @param dd The DD package to use
 * @param controls The operation controls
 * @param targets The operation targets
 * @param inverse Whether to get the inverse of the operation
 * @return The decision diagram representation of the operation
 */
MatrixDD getStandardOperationDD(const qc::StandardOperation& op, Package& dd,
                                const qc::Controls& controls,
                                const std::vector<qc::Qubit>& targets,
                                bool inverse);

/**
 * @brief Get the decision diagram representation of an operation.
 *
 * @param op The operation to get the DD for
 * @param dd The DD package to use
 * @param permutation The permutation to apply to the operation's qubits. An
 * empty permutation marks the identity permutation.
 * @param inverse Whether to get the inverse of the operation
 * @return The decision diagram representation of the operation
 */
MatrixDD getDD(const qc::Operation& op, Package& dd,
               const qc::Permutation& permutation = {}, bool inverse = false);

/**
 * @brief Get the decision diagram representation of the inverse of an
 * operation.
 *
 * @see getDD
 *
 * @param op The operation to get the inverse DD for
 * @param dd The DD package to use
 * @param permutation The permutation to apply to the operation's qubits. An
 * empty permutation marks the identity permutation.
 * @return The decision diagram representation of the inverse of the operation
 */
MatrixDD getInverseDD(const qc::Operation& op, Package& dd,
                      const qc::Permutation& permutation = {});

/**
 * @brief Apply a unitary operation to a given vector DD.
 *
 * @details This is a convenience function that realizes @p op times @p in and
 * correctly accounts for the permutation of the operation's qubits and
 * automatically manages root-set membership.
 *
 * @param op The operation to apply
 * @param in The input DD
 * @param dd The DD package to use
 * @param permutation The permutation to apply to the operation's qubits. An
 * empty permutation marks the identity permutation.
 * @return The output DD
 */
VectorDD applyUnitaryOperation(const qc::Operation& op, const VectorDD& in,
                               Package& dd,
                               const qc::Permutation& permutation = {});

/**
 * @brief Apply a unitary operation to a given matrix DD.
 *
 * @details This is a convenience function that realizes @p op times @p in and
 * correctly accounts for the permutation of the operation's qubits and
 * automatically manages root-set membership.
 *
 * @param op The operation to apply
 * @param in The input DD
 * @param dd The DD package to use
 * @param permutation The permutation to apply to the operation's qubits. An
 * empty permutation marks the identity permutation.
 * @param applyFromLeft Whether to apply the operation from the left (true)
 * or from the right (false).
 * @return The output DD
 */
MatrixDD applyUnitaryOperation(const qc::Operation& op, const MatrixDD& in,
                               Package& dd,
                               const qc::Permutation& permutation = {},
                               bool applyFromLeft = true);

/**
 * @brief Apply a measurement operation to a given DD.
 *
 * @details This is a convenience function that realizes the measurement @p op
 * on @p in and stores the measurement results in @p measurements. The result is
 * determined based on the RNG @p rng. The function correctly accounts for the
 * permutation of the operation's qubits and automatically manages root-set
 * membership.
 *
 * @param op The measurement operation to apply
 * @param in The input DD
 * @param dd The DD package to use
 * @param rng The random number generator to use
 * @param measurements The vector to store the measurement results in
 * @param permutation The permutation to apply to the operation's qubits. An
 * empty permutation marks the identity permutation.
 * @return The output DD
 */
VectorDD applyMeasurement(const qc::NonUnitaryOperation& op, VectorDD in,
                          Package& dd, std::mt19937_64& rng,
                          std::vector<bool>& measurements,
                          const qc::Permutation& permutation = {});

/**
 * @brief Apply a reset operation to a given DD.
 *
 * @details This is a convenience function that realizes the reset @p op on @p
 * in. To this end, it measures the qubit and applies an X operation if the
 * measurement result is one. The result is determined based on the RNG @p rng.
 * The function correctly accounts for the permutation of the operation's
 * qubits and automatically manages root-set membership.
 *
 * @param op The reset operation to apply
 * @param in The input DD
 * @param dd The DD package to use
 * @param rng The random number generator to use
 * @param permutation The permutation to apply to the operation's qubits. An
 * empty permutation marks the identity permutation.
 * @return The output DD
 */
VectorDD applyReset(const qc::NonUnitaryOperation& op, VectorDD in, Package& dd,
                    std::mt19937_64& rng,
                    const qc::Permutation& permutation = {});

/**
 * @brief Apply a classic controlled operation to a given DD.
 *
 * @details This is a convenience function that realizes the classic controlled
 * operation @p op on @p in. It applies the underlying operation if the actual
 * value stored in the measurement results matches the expected value according
 * to the comparison kind. The function correctly accounts for the permutation
 * of the operation's qubits and automatically manages root-set membership.
 *
 * @param op The classic controlled operation to apply
 * @param in The input DD
 * @param dd The DD package to use
 * @param measurements The vector of measurement results
 * @param permutation The permutation to apply to the operation's qubits. An
 * empty permutation marks the identity permutation.
 * @return The output DD
 */
VectorDD
applyClassicControlledOperation(const qc::ClassicControlledOperation& op,
                                const VectorDD& in, Package& dd,
                                const std::vector<bool>& measurements,
                                const qc::Permutation& permutation = {});

/**
 * @brief Check whether @p op is virtually executable.
 *
 * @param op The operation in question.
 * @return Whether @p op is virtually executable.
 */
bool isExecutableVirtually(const qc::Operation& op) noexcept;

/**
 * @brief Apply virtual operation @p op.
 *
 * @param op The virtual operation to apply.
 * @param permutation If suitable, the to be updated permutation.
 */
void applyVirtualOperation(const qc::Operation& op,
                           qc::Permutation& permutation) noexcept;

/**
 * @brief Apply global phase to a given DD.
 *
 * @param in The input DD
 * @param phase The phase to apply
 * @param dd The DD package to use
 * @return The output DD
 */
VectorDD applyGlobalPhase(VectorDD& in, const fp& phase, Package& dd);

/**
 * @brief Change the permutation of a given DD.
 *
 * @details This function changes the permutation of the given DD @p on from
 * @p from to @p to by applying SWAP gates. The @p from permutation must be at
 * least as large as the @p to permutation.
 *
 * @tparam DDType The type of the DD
 * @param on The DD to change the permutation of
 * @param from The current permutation
 * @param to The target permutation
 * @param dd The DD package to use
 * @param regular Whether to apply the permutation from the left (true) or from
 * the right (false)
 */
template <class DDType>
void changePermutation(DDType& on, qc::Permutation& from,
                       const qc::Permutation& to, Package& dd,
                       const bool regular = true) {
  assert(from.size() >= to.size());
  if (on.isZeroTerminal()) {
    return;
  }

  // iterate over (k,v) pairs of second permutation
  for (const auto& [i, goal] : to) {
    // search for key in the first map
    auto it = from.find(i);
    if (it == from.end()) {
      throw std::runtime_error(
          "[changePermutation] Key " + std::to_string(it->first) +
          " was not found in first permutation. This should never happen.");
    }
    auto current = it->second;

    // permutations agree for this key value
    if (current == goal) {
      continue;
    }

    // search for goal value in first permutation
    qc::Qubit j = 0;
    for (const auto& [key, value] : from) {
      if (value == goal) {
        j = key;
        break;
      }
    }

    // swap i and j
    auto saved = on;
    const auto swapDD = dd.makeTwoQubitGateDD(opToTwoQubitGateMatrix(qc::SWAP),
                                              from.at(i), from.at(j));
    if constexpr (std::is_same_v<DDType, VectorDD>) {
      on = dd.multiply(swapDD, on);
    } else {
      // the regular flag only has an effect on matrix DDs
      if (regular) {
        on = dd.multiply(swapDD, on);
      } else {
        on = dd.multiply(on, swapDD);
      }
    }

    dd.incRef(on);
    dd.decRef(saved);
    dd.garbageCollect();

    // update permutation
    from.at(i) = goal;
    from.at(j) = current;
  }
}

} // namespace dd
