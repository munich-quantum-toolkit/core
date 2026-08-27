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

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <map>
#include <random>
#include <string>

namespace mlir::qco {

/**
 * @brief Sequentially build a matrix DD for a static unitary QCO `func.func`.
 *
 * @details Walks the entry block of @p func, maps `qco.static` SSA values to
 * wire indices (or, if none are present, qubit-typed function arguments as
 * wires `0..n-1`), assigns entry-block `qco.alloc` operations subsequent
 * wires, and applies unitary operations via decision-diagram multiplication.
 *
 * Supported programs:
 * - Standard single-, two-, and three-qubit gates with compile-time constant
 *   parameters (sparse DD path)
 * - `ctrl` with a sole standard-gate body (same sparse path)
 * - Other `UnitaryOpInterface` ops with a compile-time known matrix (`inv`,
 *   compound `ctrl`, ...), including `gphase` and `barrier`
 * - `qco.static` establishes the wire map (or qubit-typed `func` args if none),
 *   followed by entry-block `qco.alloc`; `sink` is ignored; `arith.constant`
 *   is ignored for matrix construction; `func.return` accepts qubit results
 *   only in canonical wire order
 *
 * Known one-, two-, and three-qubit matrices are constructed directly as DD
 * gates. Larger compile-time unitaries are embedded directly into a DD over
 * their target wires, so idle register qubits do not enlarge the local matrix.
 * Measurements, resets, symbolic parameters, and control-flow ops are not
 * supported.
 *
 * @param func The QCO function to construct the functionality for
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @return The matrix DD on success, or failure for unsupported programs
 */
FailureOr<dd::MatrixDD> buildFunctionality(func::FuncOp func, dd::Package& dd);

/**
 * @brief Simulate a QCO `func.func` on a given input state without stochastic
 * collapse.
 *
 * @details Same supported unitary op set as @ref buildFunctionality, plus
 * concrete classical control-flow (`qco.if` / `qco.index_switch` with
 * compile-time or previously recorded classical selectors) and CBit registers
 * (`cbit.alloc` / `cbit.store` / `cbit.load`). Mid-circuit `measure` /
 * `reset` require the RNG overload below. Concrete-bound `scf.for` loops and
 * non-recursive single-block `func.call` are supported independently of RNG.
 * Only qubit-typed linear values are supported (no qtensors). Nested regions
 * are walked; `scf.while` and multi-block function bodies remain unsupported.
 * Consumes one reference to @p in regardless of whether simulation succeeds or
 * fails.
 *
 * @param func The QCO function to simulate
 * @param in The input state, which must span at least the function's qubits;
 * higher wires are preserved; one reference is consumed
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @return The output statevector DD on success, or failure for unsupported
 *         programs
 */
FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd);

/**
 * @brief Simulate a QCO `func.func` that may contain measurements, resets, and
 * concrete control-flow.
 *
 * @details Supports the unitary op set of @ref buildFunctionality, plus
 * `qco.measure` / `qco.reset` (collapsing via @p rng) and `qco.if` /
 * `qco.index_switch` when the branch selector is a concrete classical SSA value
 * (`arith.constant` integer/index, a prior measurement, a `cbit.load`,
 * `arith.index_castui`, `arith.cmpi`, `arith.select`,
 * `arith.addi` / `subi` / `muli`, or `andi` / `ori` / `xori` / `shli` /
 * `shrui` on those values). The simulation tracks CBit initialization, loads,
 * and stores. Deterministic control-flow without measure/reset also works on
 * the non-RNG overload. Only qubit-typed linear values are supported (no
 * qtensors). Nested regions are walked; direct `scf.for` execution with
 * concrete positive steps and non-recursive single-block `func.call` are
 * supported. A shared 10000-step budget bounds loop iterations across nested
 * loops and calls; `scf.while` and multi-block function bodies remain
 * unsupported.
 * Consumes one reference to @p in regardless of whether simulation succeeds or
 * fails.
 *
 * @param func The QCO function to simulate
 * @param in The input state, which must span at least the function's qubits;
 * higher wires are preserved; one reference is consumed
 * @param dd The DD package to use
 * @param rng RNG used for collapsing measurements and resets
 * @return The output statevector DD on success, or failure for unsupported
 *         programs
 */
FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng);

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
 * once per shot.
 *
 * @param func The QCO function to sample
 * @param dd The DD package to use
 * @param shots Number of shots
 * @param rng RNG for collapsing measurements and non-collapsing sampling
 * @return Histogram of outcome strings on success, or failure for unsupported
 *         programs
 */
FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, dd::Package& dd, size_t shots, std::mt19937_64& rng);

} // namespace mlir::qco
