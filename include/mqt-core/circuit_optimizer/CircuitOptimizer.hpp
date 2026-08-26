/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file CircuitOptimizer.hpp
 * @brief Circuit-optimizer interface.
 */

#pragma once

namespace qc {

class QuantumComputation;

/**
 * @brief Shared circuit transformations used across MQT projects.
 */
class CircuitOptimizer {
public:
  /**
   * @brief Fuse adjacent single-qubit gates.
   * @param qc The circuit to transform.
   */
  static void singleQubitGateFusion(QuantumComputation& qc);

  /**
   * @brief Remove measurements and barriers that form the end of a circuit.
   * @param qc The circuit to transform.
   */
  static void removeFinalMeasurements(QuantumComputation& qc);

  /**
   * @brief Flatten compound operations.
   * @param qc The circuit to transform.
   * @param customGatesOnly Whether to flatten only custom gates.
   */
  static void flattenOperations(QuantumComputation& qc,
                                bool customGatesOnly = false);

private:
  CircuitOptimizer() = delete;
};
} // namespace qc
