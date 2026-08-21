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

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <variant>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::benchmark {

using namespace mlir;

/// Resets the @p size qubits of the register @p reg.
void resetRegister(qc::QCProgramBuilder& b, Value reg, int64_t size);

/// Measures the @p size qubits of the register @p reg into the classical
/// register @p bits, one qubit per bit.
void measureRegister(qc::QCProgramBuilder& b, Value reg, int64_t size,
                     Value bits);
/**
 * @brief Runs @p body over the range [@p lower, @p upper) with a scaled angle
 *
 * @details A chain of phase rotations halves or doubles its angle from one
 * qubit to the next. The angle starts at @p start and is multiplied by
 * @p factor after every step, so the loop never raises two to a power that
 * depends on the induction variable. A start angle that an enclosing loop
 * carries is passed as a value. @p body receives the angle of the step and the
 * induction variable.
 */
void phaseRotationLoop(qc::QCProgramBuilder& b, Value lower, Value upper,
                       const std::variant<double, Value>& start, double factor,
                       const function_ref<void(Value, Value)>& body);

} // namespace mqt::benchmark
