/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file OpenQASMSerializer.hpp
 * @brief OpenQASM serialization for the circuit intermediate representation.
 */

#pragma once

#include "ir/Definitions.hpp"
#include "ir/Register.hpp"

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>

namespace qc {

class Operation;
class QuantumComputation;

using QubitIndexToRegisterMap =
    std::unordered_map<Qubit, std::pair<QuantumRegister, std::string>>;
using BitIndexToRegisterMap =
    std::unordered_map<Bit, std::pair<ClassicalRegister, std::string>>;

/**
 * @brief Serializes circuit IR computations and operations to OpenQASM.
 */
class OpenQASMSerializer final {
public:
  /**
   * @brief Callback for serializing otherwise unsupported leaf operations.
   * @return Whether the operation was serialized.
   */
  using CustomOperationSerializer = std::function<bool(
      std::ostream&, const Operation&, const QubitIndexToRegisterMap&,
      const BitIndexToRegisterMap&, std::size_t)>;

  explicit OpenQASMSerializer(std::ostream& output,
                              Format format = Format::OpenQASM3,
                              CustomOperationSerializer customSerializer = {})
      : output(output), format(format),
        customOperationSerializer(std::move(customSerializer)) {}

  /**
   * @brief Serialize a complete quantum computation.
   * @param computation The computation to serialize.
   */
  void serialize(const QuantumComputation& computation) const;

  /**
   * @brief Serialize a single operation without a circuit header.
   * @param operation The operation to serialize.
   * @param qubitMap Map from qubit indices to their register and operand name.
   * @param bitMap Map from bit indices to their register and operand name.
   * @param indent Nesting level, using two spaces per level.
   */
  void serialize(const Operation& operation,
                 const QubitIndexToRegisterMap& qubitMap,
                 const BitIndexToRegisterMap& bitMap,
                 std::size_t indent = 0U) const;

private:
  std::ostream& output;
  Format format;
  CustomOperationSerializer customOperationSerializer;
};

} // namespace qc
