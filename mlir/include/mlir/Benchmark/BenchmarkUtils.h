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

#include <llvm/ADT/ArrayRef.h>
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
 * @details The angle starts at @p start and is multiplied by @p factor after
 * every step, which is how a chain of phase rotations spreads over a register.
 * A start angle that an enclosing loop carries is passed as a value. @p body
 * receives the angle of the step and the induction variable.
 */
void scfForWithAngle(qc::QCProgramBuilder& b, Value lower, Value upper,
                     const std::variant<double, Value>& start, double factor,
                     const function_ref<void(Value, Value)>& body);

/**
 * @brief Adds a classical value to @p reg in the Fourier basis
 *
 * @details The value enters through @p base, which carries pi times the value.
 * Every qubit takes half the angle of the one before it, so the addition is a
 * single layer of phase gates. An empty @p controls adds unconditionally.
 */
void phaseAdd(qc::QCProgramBuilder& b, Value reg, int64_t size, Value base,
              ArrayRef<Value> controls);

/**
 * @brief Adds a classical value to @p acc modulo a modulus
 *
 * @details Follows Beauregard's construction. The adder subtracts the modulus,
 * reads the top qubit to learn whether the sum underflowed, and adds the
 * modulus back when it did. The second half compares the result against the
 * added value so that @p anc returns to zero, which is what lets the adder sit
 * inside a larger circuit.
 */
void modularAdd(qc::QCProgramBuilder& b, Value acc, int64_t size, Value addend,
                Value modulus, ArrayRef<Value> controls, Value anc);

/**
 * @brief Multiplies the register @p x into @p acc modulo a modulus
 *
 * @details Follows Beauregard's construction. Multiplier bit `i` adds
 * `(2^i * a) mod N`, so the round carries that value and doubles it modulo
 * @p modulus for the next one. A negative @p sign subtracts the product, which
 * is how the accumulator is returned to zero.
 */
void modularMultiply(qc::QCProgramBuilder& b, Value ctrl, Value x, Value acc,
                     Value anc, int64_t bits, Value first, int64_t modulus,
                     double sign);

/**
 * @brief Applies a quantum Fourier transform to @p reg
 *
 * @details A negative @p sign applies the inverse transform, which runs the
 * same rotations in the opposite order.
 */
void fourierTransform(qc::QCProgramBuilder& b, Value reg, int64_t size,
                      double sign);

} // namespace mqt::benchmark
