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
 * wires `0..n-1`), and applies unitary operations via decision-diagram
 * multiplication.
 *
 * Supported programs:
 * - Standard single-, two-, and three-qubit gates with compile-time constant
 *   parameters (sparse DD path)
 * - `ctrl` with a sole standard-gate body (same sparse path)
 * - Other `UnitaryOpInterface` ops with a compile-time known matrix (`inv`,
 *   compound `ctrl`, ...), including `gphase` and `barrier`
 * - `qco.static` establishes the wire map (or qubit-typed `func` args if none);
 *   `sink` is ignored; `arith.constant` is ignored for matrix construction;
 *   `func.return` accepts qubit results only in canonical wire order
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
 * @param in The input state, represented as a vector DD; one reference is
 * consumed
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
 * (`arith.constant` `i1`/`index`, a prior measurement, a `cbit.load`,
 * `arith.index_castui` from `i1` to `index`, `arith.cmpi`, `arith.select`,
 * `arith.addi` / `subi` / `muli`, or `andi` / `ori` / `xori` / `shli` /
 * `shrui` on those values). The simulation tracks CBit initialization, loads,
 * and stores. Deterministic control-flow without measure/reset also works on
 * the non-RNG overload. Only qubit-typed linear values are supported (no
 * qtensors). Nested regions are walked; `scf.for` with a concrete positive
 * step and at most 10000 trips and non-recursive single-block `func.call` are
 * supported; `scf.while` and multi-block function bodies remain unsupported.
 * Consumes one reference to @p in regardless of whether simulation succeeds or
 * fails.
 *
 * @param func The QCO function to simulate
 * @param in The input state; one reference is consumed
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
 * @details Starts from the all-zero state and draws @p shots bitstrings via
 * `Package::measureAll` (qubit `n-1` … `0`, same as @ref dd::sample). Programs
 * without `measure` / `reset` are simulated once and sampled without
 * collapsing (including deterministic control-flow). Programs with mid-circuit
 * `measure` / `reset` are re-simulated per shot with @p rng. Histograms are
 * final computational-basis bitstrings, not classical mid-circuit records.
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

/**
 * @brief Sample measurement outcomes from a QCO `func.func` on a given input.
 *
 * @details Same as the zero-state overload, but starts from @p in. Consumes one
 * reference to @p in (the static path keeps that state for all shots; the
 * dynamic path clones per shot).
 *
 * @param func The QCO function to sample
 * @param in Input state; one reference is consumed
 * @param dd The DD package to use
 * @param shots Number of shots
 * @param rng RNG for collapsing measurements and non-collapsing sampling
 * @return Histogram of outcome strings on success, or failure for unsupported
 *         programs
 */
FailureOr<std::map<std::string, size_t>> sample(func::FuncOp func,
                                                const dd::VectorDD& in,
                                                dd::Package& dd, size_t shots,
                                                std::mt19937_64& rng);

/// Histograms produced by @ref sampleWithClassics.
struct SampleResult {
  /// Final computational-basis outcome histogram.
  std::map<std::string, size_t> shots;
  /// Mid-circuit measurement-bit histogram (encounter order).
  std::map<std::string, size_t> classical;
};

/**
 * @brief Sample final and mid-circuit classical outcomes from a QCO
 * `func.func`.
 *
 * @details Like @ref sample, but also histograms collapsing mid-circuit
 * measurement bits in encounter order into @c SampleResult::classical.
 * Programs without mid-circuit measures leave @c classical empty.
 */
FailureOr<SampleResult> sampleWithClassics(func::FuncOp func, dd::Package& dd,
                                           size_t shots, std::mt19937_64& rng);

/// @copydoc sampleWithClassics(func::FuncOp, dd::Package&, size_t,
/// std::mt19937_64&)
/// Starts from @p in; one reference is consumed.
FailureOr<SampleResult> sampleWithClassics(func::FuncOp func,
                                           const dd::VectorDD& in,
                                           dd::Package& dd, size_t shots,
                                           std::mt19937_64& rng);

} // namespace mlir::qco
