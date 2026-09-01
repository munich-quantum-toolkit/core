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

#include "dd/Package_fwd.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <map>
#include <random>
#include <string>

namespace mlir::qco {

/// Concrete values for QCO DD entry arguments.
///
/// Values must be integer, index, or `f64` attributes. An index value sets the
/// size of a dynamic one-dimensional QTensor argument.
using DDArgumentBindings = DenseMap<Value, Attribute>;

/// Build a matrix DD for a unitary QCO function.
///
/// The function must have one block. The interpreter supports concrete QCO and
/// SCF structured control, non-recursive calls, common scalar math,
/// one-dimensional memrefs, and QTensor bookkeeping. `qco.static` values, or
/// qubit arguments when no static values exist, set the wire map. Entry-block
/// `qco.alloc` operations add subsequent wires. Measurements, resets, symbolic
/// control, and other runtime allocation are not supported.
///
/// The containing module must pass MLIR verification and
/// `qco::verifyLinearity`.
///
/// @param func QCO function to interpret.
/// @param dd DD package. The function grows it when needed.
/// @param argumentBindings Scalar values and dynamic QTensor argument sizes.
/// @return Matrix DD, or failure for an unsupported program.
FailureOr<dd::MatrixDD> buildFunctionality(
    func::FuncOp func, dd::Package& dd,
    const DDArgumentBindings& argumentBindings = DDArgumentBindings());

/// Simulate a single-block QCO function.
///
/// In addition to the operations supported by `buildFunctionality`, simulation
/// supports measurements, resets, CBit registers, and runtime qubit and QTensor
/// allocation. QCO and SCF structured control requires concrete values. A
/// shared 10000-step limit bounds loops and calls. `qco.sink` and
/// `qtensor.dealloc` mark lifetimes but do not remove DD wires.
///
/// The containing module must pass MLIR verification and
/// `qco::verifyLinearity`. The function consumes one reference to `in`, also on
/// failure.
///
/// @param func QCO function to simulate.
/// @param in Input state. It must contain every function qubit.
/// @param dd DD package. The function grows it when needed.
/// @param rng Random-number generator for measurements and resets.
/// @param argumentBindings Scalar values and dynamic QTensor argument sizes.
/// @return Output state DD, or failure for an unsupported program.
FailureOr<dd::VectorDD>
simulate(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
         std::mt19937_64& rng,
         const DDArgumentBindings& argumentBindings = DDArgumentBindings());

/// Simulate a zero-state QCO function without collapsing terminal
/// measurements.
///
/// The function must have one block. Measurement results can be unused or can
/// set returned CBit registers. No later operation can use a measured wire.
/// Called functions cannot measure or reset qubits.
FailureOr<dd::VectorDD> simulateStatevector(func::FuncOp func, dd::Package& dd);

/// Sample a single-block QCO function from the zero state.
///
/// Returned CBit registers set the outcome in return order and from high to low
/// bit. Without CBit results, the function samples all DD wires. Terminal
/// measurements use one simulation for all shots. Programs that use a measured
/// value or wire, or reset a qubit, run once per shot.
///
/// The containing module must pass MLIR verification and
/// `qco::verifyLinearity`.
///
/// @param func QCO function to sample.
/// @param dd DD package. The function grows it when needed.
/// @param shots Number of samples.
/// @param rng Random-number generator.
/// @param argumentBindings Scalar values and dynamic QTensor argument sizes.
/// @return Outcome counts, or failure for an unsupported program.
FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, dd::Package& dd, size_t shots, std::mt19937_64& rng,
       const DDArgumentBindings& argumentBindings = DDArgumentBindings());
} // namespace mlir::qco
