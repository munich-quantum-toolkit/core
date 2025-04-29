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

#include "ZXDefinitions.hpp"
#include "ZXDiagram.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/Operation.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace zx {

/**
 * @brief Static class to construct ZX-diagrams from a Quantumcomputation
 */
class FunctionalityConstruction {
  using op_it = qc::QuantumComputation::const_iterator;

public:
  /**
   * @brief Builds a ZX-diagram from a QuantumComputation
   *
   * @param qc Pointer to QuantumComputation to build the ZX-diagram from
   * @return ZXDiagram
   */
  static ZXDiagram buildFunctionality(const qc::QuantumComputation* qc);

  /**
   * @brief Check whether a given QuantumComputation can be transformed to a
   * ZXDiagram.
   * @details Not all instructions supported by the QuantumComputation are
   * supported by the ZXDiagram (e.g. arbitrarily-controlled multi-qubit gates).
   * @param qc Pointer to QuantumComputation to check
   * @return true if the QuantumComputation can be transformed to a ZXDiagram,
   * false otherwise
   */
  static bool transformableToZX(const qc::QuantumComputation* qc);

  /**
   * @brief Check whether a given Operation can be transformed to a ZXDiagram.
   * @details Not all Operations have a corresponding representation in the
   * ZX-calculus.
   * @param op Pointer to Operation to check
   * @return true if the Operation can be transformed to a ZXDiagram, false
   * otherwise
   */
  static bool transformableToZX(const qc::Operation* op);

protected:
  static bool checkSwap(const op_it& it, const op_it& end, Qubit ctrl,
                        Qubit target, const qc::Permutation& p);
  static void addZSpider(ZXDiagram& diag, zx::Qubit qubit,
                         std::vector<Vertex>& qubits,
                         const PiExpression& phase = PiExpression(),
                         EdgeType type = EdgeType::Simple);
  static void addXSpider(ZXDiagram& diag, Qubit qubit,
                         std::vector<Vertex>& qubits,
                         const PiExpression& phase = PiExpression(),
                         EdgeType type = EdgeType::Simple);
  static void
  addRz(ZXDiagram& diag, const PiExpression& phase, Qubit target,
        std::vector<Vertex>& qubits,
        const std::optional<double>& unconvertedPhase = std::nullopt);
  static void addRx(ZXDiagram& diag, const PiExpression& phase, Qubit target,
                    std::vector<Vertex>& qubits);
  static void
  addRy(ZXDiagram& diag, const PiExpression& phase, Qubit target,
        std::vector<Vertex>& qubits,
        const std::optional<double>& unconvertedPhase = std::nullopt);
  static void addCnot(ZXDiagram& diag, Qubit ctrl, Qubit target,
                      std::vector<Vertex>& qubits,
                      EdgeType type = EdgeType::Simple);
  static void addCphase(ZXDiagram& diag, const PiExpression& phase, Qubit ctrl,
                        Qubit target, std::vector<Vertex>& qubits);
  static void addSwap(ZXDiagram& diag, Qubit target, Qubit target2,
                      std::vector<Vertex>& qubits);
  static void
  addRzz(ZXDiagram& diag, const PiExpression& phase, Qubit target,
         Qubit target2, std::vector<Vertex>& qubits,
         const std::optional<double>& unconvertedPhase = std::nullopt);
  static void
  addRxx(ZXDiagram& diag, const PiExpression& phase, Qubit target,
         Qubit target2, std::vector<Vertex>& qubits,
         const std::optional<double>& unconvertedPhase = std::nullopt);
  static void
  addRzx(ZXDiagram& diag, const PiExpression& phase, Qubit target,
         Qubit target2, std::vector<Vertex>& qubits,
         const std::optional<double>& unconvertedPhase = std::nullopt);
  static void addDcx(ZXDiagram& diag, Qubit qubit1, Qubit qubit2,
                     std::vector<Vertex>& qubits);
  static void
  addXXplusYY(ZXDiagram& diag, const PiExpression& theta,
              const PiExpression& beta, Qubit qubit0, Qubit qubit1,
              std::vector<Vertex>& qubits,
              const std::optional<double>& unconvertedBeta = std::nullopt);
  static void
  addXXminusYY(ZXDiagram& diag, const PiExpression& theta,
               const PiExpression& beta, Qubit qubit0, Qubit qubit1,
               std::vector<Vertex>& qubits,
               const std::optional<double>& unconvertedBeta = std::nullopt);
  static void addCcx(ZXDiagram& diag, Qubit ctrl0, Qubit ctrl1, Qubit target,
                     std::vector<Vertex>& qubits);
  static op_it parseOp(ZXDiagram& diag, op_it it, op_it end,
                       std::vector<Vertex>& qubits, const qc::Permutation& p);
  static op_it parseCompoundOp(ZXDiagram& diag, op_it it, op_it end,
                               std::vector<Vertex>& qubits,
                               const qc::Permutation& initialLayout);

  static PiExpression toPiExpr(const qc::SymbolOrNumber& param);
  static PiExpression parseParam(const qc::Operation* op, std::size_t i);
};
} // namespace zx
