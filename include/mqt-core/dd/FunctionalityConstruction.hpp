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

#include "dd/Package_fwd.hpp"

namespace qc {
class QuantumComputation;
}

namespace dd {
/**
 * @brief Sequentially build a decision diagram representation for the
 * functionality of a purely-quantum @ref qc::QuantumComputation.
 *
 * @details For a circuit \f$G\f$ with \f$|G|\f$ gates
 * \f$g_0, g_1, \ldots, g_{|G|-1}\f$, the functionality of \f$G\f$ is defined as
 * the unitary matrix \f$U\f$ such that
 * \f[
 * U = U_{|G|-1}) \cdot U_{|G|-2} \cdot \ldots \cdot U_1 \cdot U_0,
 * \f]
 * where \f$U_i\f$ is the unitary matrix corresponding to gate \f$g_i\f$.
 * For an \f$n\f$-qubit quantum computation, \f$U\f$ is a \f$2^n \times 2^n\f$
 * matrix.
 *
 * By representing every single operation in the circuit as a decision diagram
 * instead of a unitary matrix and performing the matrix multiplication directly
 * using decision diagrams, a representation of the functionality of a quantum
 * computation can oftentimes be computed more efficiently in terms of memory
 * and runtime.
 *
 * This function effectively computes
 * \f[
 * DD(U) = DD(g_{|G|-1}) \otimes DD(g_{|G|-2}) \otimes \ldots \otimes DD(g_0)
 * \f]
 * by sequentially applying the decision diagrams of the gates in the circuit to
 * the current decision diagram representing the functionality of the quantum
 * computation.
 *
 * @param qc The quantum computation to construct the functionality for
 * @param dd The DD package to use for the construction
 * @return The matrix diagram representing the functionality of the quantum
 * computation
 */
MatrixDD buildFunctionality(const qc::QuantumComputation& qc, Package& dd);

/**
 * @brief Recursively build a decision diagram representation for the
 * functionality of a purely-quantum @ref qc::QuantumComputation.
 *
 * @see buildFunctionality
 * @details Instead of sequentially applying the decision diagrams of the gates
 * in the circuit, this function builds a binary computation tree out of the
 * decision diagrams of the gates in the circuit.
 * This results in a recursive pairwise grouping that can be more memory and
 * runtime efficient compared to the sequential approach.
 * @see https://arxiv.org/abs/2103.08281
 *
 * @param qc The quantum computation to construct the functionality for
 * @param dd The DD package to use for the construction
 * @return The matrix diagram representing the functionality of the quantum
 * computation
 */
MatrixDD buildFunctionalityRecursive(const qc::QuantumComputation& qc,
                                     Package& dd);

} // namespace dd
