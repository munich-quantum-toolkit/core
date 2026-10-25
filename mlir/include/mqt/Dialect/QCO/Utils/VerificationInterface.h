/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QCO/Utils/Verification.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace mlir::qco {

/// One program's physical realization of the shared verification interface.
///
/// inputWires and outputWires map logical positions to resolved trace wires.
/// Source slots occupy [0, numInputs); private workspace follows. Both vectors
/// are complete permutations of [0, numQubits), including idle layout sites
/// absent from the trace. zeroInputs is indexed by logical input position.
/// Output tokens retain trace wire coordinates, to be decoded with outputWires.
struct VerificationInterface {
  size_t numInputs = 0;
  size_t numQubits = 0;
  std::vector<dd::Qubit> inputWires;
  std::vector<dd::Qubit> outputWires;
  std::vector<bool> zeroInputs;
  /// Logical output positions that must be proved zero before omission.
  std::vector<dd::Qubit> cleanOutputs;
  std::vector<std::string> outputLabels;
  std::vector<DDOutputBit> outputs;
};

/// Resolve layout and readout identities without rewriting either program.
///
/// expectedSourceCount comes from the reference interface, never from an
/// independently compacted second program. Both it and contract.numInputs are
/// checked before mapping. Labels are resultN:name[bit], or resultN[bit] when
/// unnamed. Explicit selections preserve the caller's label order.
[[nodiscard]] FailureOr<VerificationInterface> resolveVerificationInterface(
    func::FuncOp func, const DDResolvedExecution& execution,
    const VerificationContract& contract = {},
    std::optional<size_t> expectedSourceCount = std::nullopt);

} // namespace mlir::qco
