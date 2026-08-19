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

/**
 * @brief Concrete values for symbolic QCO DD inputs.
 *
 * Integer and `f64` attributes bind scalar function arguments. An integer
 * attribute bound to a dynamic one-dimensional qtensor argument gives its
 * runtime extent. Bindings for other values are rejected.
 */
using DDBindings = DenseMap<Value, Attribute>;

/**
 * @brief Sequentially build a matrix DD for a static unitary QCO `func.func`.
 *
 * @details Walks the entry block of @p func, maps `qco.static` SSA values to
 * wire indices (or, if none are present, qubit-typed function arguments as
 * wires `0..n-1`), assigns entry-block `qco.alloc` operations subsequent
 * wires, and applies unitary operations via decision-diagram multiplication.
 *
 * Supported programs:
 * - Standard single-, two-, and three-qubit gates with constant or bound
 *   parameters (sparse DD path)
 * - `ctrl` with a sole standard-gate body (same sparse path)
 * - Other `UnitaryOpInterface` ops with a compile-time known matrix (`inv`,
 *   compound `ctrl`, ...), including `gphase` and `barrier`
 * - QTensor bookkeeping over existing input wires
 * - Concrete QCO and SCF control flow and non-recursive single-block calls
 * - Concrete integer, index, and `f64` arithmetic and one-dimensional memrefs
 *   of those scalar types
 * - `qco.static` establishes the wire map (or qubit-typed `func` args if none),
 *   followed by entry-block `qco.alloc`; `sink` is ignored; returned qubits
 *   and qtensors must preserve canonical wire order
 *
 * Known one-, two-, and three-qubit matrices are constructed directly as DD
 * gates. Larger compile-time unitaries are embedded directly into a DD over
 * their target wires, so idle register qubits do not enlarge the local matrix.
 * Measurements, resets, unbound parameters, and non-concrete control flow are
 * not supported.
 *
 * @pre The containing module has passed MLIR verification and
 * `qco::verifyLinearity`.
 *
 * @param func The QCO function to construct the functionality for
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @param bindings Concrete values for symbolic function arguments
 * @return The matrix DD on success, or failure for unsupported programs
 */
FailureOr<dd::MatrixDD>
buildFunctionality(func::FuncOp func, dd::Package& dd,
                   const DDBindings& bindings = DDBindings());

/**
 * @brief Simulate a QCO `func.func` that may contain measurements, resets, and
 * concrete control-flow.
 *
 * @details Supports the unitary op set of @ref buildFunctionality, plus
 * `qco.measure` / `qco.reset` (collapsing via @p rng) and `qco.if` /
 * `qco.index_switch` when the branch selector is a concrete classical SSA value
 * (`arith.constant`, a prior measurement, integer and `f64`
 * arithmetic, comparisons, casts, shifts, and `arith.select`). Dynamic quantum
 * allocation, qtensors, memrefs, CBit registers, loops, regions, and calls are
 * supported. QTensor sizes and indices must be concrete; dynamic qtensor
 * arguments require an extent in @p bindings. A shared 10000-step budget bounds
 * loop iterations across nested loops and calls. Multi-block function bodies
 * remain unsupported. Allocated wires stay in the returned state after
 * deallocation.
 * Consumes one reference to @p in regardless of success or failure.
 *
 * @pre The containing module has passed MLIR verification and
 * `qco::verifyLinearity`.
 *
 * @param func The QCO function to simulate
 * @param in The input state, which must span at least the function's qubits;
 * higher wires are preserved; one reference is consumed
 * @param dd The DD package to use
 * @param rng RNG used for collapsing measurements and resets
 * @param bindings Concrete values for symbolic function arguments
 * @return The output statevector DD on success, or failure for unsupported
 *         programs
 */
FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng,
                                 const DDBindings& bindings = DDBindings());

/**
 * @brief Sample measurement outcomes from a QCO `func.func`.
 *
 * @details Starts from the all-zero state. If the entry function returns CBit
 * registers, their initialized cells form the outcome in return order and from
 * bit `N-1` to bit `0` within each register. Mixed CBit/non-CBit results are
 * rejected. Programs without CBit results fall back to final computational-
 * basis sampling via `Package::measureAll` (qubit `n-1` … `0`). Terminal entry-
 * block measurements that only produce returned CBit cells are sampled from
 * one DD evolution; resets and execution-dependent measurements are executed
 * once per shot. Dynamically allocated wires are included in fallback basis
 * outcomes even after deallocation.
 *
 * @pre The containing module has passed MLIR verification and
 * `qco::verifyLinearity`.
 *
 * @param func The QCO function to sample
 * @param dd The DD package to use
 * @param shots Number of shots
 * @param rng RNG for collapsing measurements and non-collapsing sampling
 * @param bindings Concrete values for symbolic function arguments
 * @return Histogram of outcome strings on success, or failure for unsupported
 *         programs
 */
FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, dd::Package& dd, size_t shots, std::mt19937_64& rng,
       const DDBindings& bindings = DDBindings());
} // namespace mlir::qco
