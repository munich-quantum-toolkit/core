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
 * Integer and floating-point attributes bind scalar SSA values. An integer
 * attribute bound to a dynamic one-dimensional qtensor argument supplies its
 * runtime extent. Bindings must target function arguments; unbound symbolic
 * values still cause functionality construction or simulation to fail.
 */
using DDBindings = DenseMap<Value, Attribute>;

/**
 * @brief Sequentially build a matrix DD for a static unitary QCO `func.func`.
 *
 * @details Walks the entry block of @p func, maps `qco.static` SSA values to
 * wire indices (or, if none are present, qubit-typed function arguments as
 * wires `0..n-1`), and applies unitary operations via decision-diagram
 * multiplication.
 *
 * Supported programs:
 * - Standard single-, two-, and three-qubit gates with constant or bound
 *   parameters (sparse DD path)
 * - `ctrl` with a sole standard-gate body (same sparse path)
 * - Other `UnitaryOpInterface` ops with a compile-time known matrix (`inv`,
 *   compound `ctrl`, ...), including `gphase` and `barrier`
 * - `qtensor.from_elements` / `extract` / `insert` / `dealloc` as linear
 *   bookkeeping over existing input wires
 * - Concrete `qco.if` / `qco.index_switch`, bounded `scf.for` / `scf.while`,
 *   standard `scf.if` / `scf.index_switch` / single-block
 *   `scf.execute_region`, and non-recursive single-block `func.call`
 * - Concrete integer, index, and floating-point `arith` operations and
 *   one-dimensional `memref` storage over those scalar types
 * - `qco.static` establishes the wire map (or qubit-typed `func` args if none);
 *   `sink` is ignored; `arith.constant` is ignored for matrix construction;
 *   `func.return` accepts qubit results only in canonical wire order
 *
 * Known one-, two-, and three-qubit matrices are constructed directly as DD
 * gates. Larger compile-time unitaries (including partial wire subsets) use a
 * dense embed into the full register, rewritten from QCO/MSB to DD/LSB, limited
 * to 12 qubits (`2^n × 2^n` storage). Quantum allocations, measurements,
 * resets, unbound symbolic parameters, and non-concrete control flow are not
 * supported.
 *
 * @param func The QCO function to construct the functionality for
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @param bindings Concrete values for symbolic scalar function arguments
 * @return The matrix DD on success, or failure for unsupported programs
 */
FailureOr<dd::MatrixDD>
buildFunctionality(func::FuncOp func, dd::Package& dd,
                   const DDBindings& bindings = DDBindings());

/**
 * @brief Simulate a QCO `func.func` on a given input state without stochastic
 * collapse.
 *
 * @details Same supported unitary op set as @ref buildFunctionality, plus
 * concrete classical control-flow (`qco.if`, `qco.index_switch`, `scf.if`,
 * `scf.index_switch`, and single-block `scf.execute_region`) and static- or
 * concrete dynamic-shape 1-D `memref` registers of integer, index, or
 * floating-point values (`alloc`/`store`/`load`/`dealloc`).
 * `qco.alloc` and `qtensor.alloc` append zero-state wires, while
 * `qtensor.from_elements` / `extract` / `insert` / `dealloc` track their linear
 * ownership. QTensor sizes and indices must be concrete classical values;
 * dynamic qtensor function arguments require an extent in @p bindings.
 * Mid-circuit `measure` / `reset` require the RNG overload below. Concrete-
 * bound `scf.for` loops, concrete `scf.while` loops, and non-recursive
 * single-block `func.call` are supported independently of RNG. Loops are
 * limited to 10000 trips. Qubits and one-dimensional qtensors can be carried
 * through nested regions; multi-block function bodies remain unsupported.
 * Dynamically allocated wires remain in the returned state after deallocation.
 * Consumes one reference to @p in regardless of whether simulation succeeds
 * or fails.
 *
 * @param func The QCO function to simulate
 * @param in The input state, represented as a vector DD; one reference is
 * consumed
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @param bindings Concrete values for symbolic scalar function arguments
 * @return The output statevector DD on success, or failure for unsupported
 *         programs
 */
FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd,
                                 const DDBindings& bindings = DDBindings());

/**
 * @brief Simulate a QCO `func.func` that may contain measurements, resets, and
 * concrete control-flow.
 *
 * @details Supports the unitary op set of @ref buildFunctionality, plus
 * `qco.measure` / `qco.reset` (collapsing via @p rng) and `qco.if` /
 * `qco.index_switch` when the branch selector is a concrete classical SSA value
 * (`arith.constant`, a prior measurement, integer and floating-point
 * arithmetic/comparisons, casts, shifts, and `arith.select`). Classical
 * registers are supported as static- or concrete dynamic-shape 1-D `memref`s
 * of integer, index, or floating-point values with `memref.alloc` / `store` /
 * `load` / `dealloc`. Dynamic qubit and qtensor allocation and qtensor
 * ownership operations are supported as described by the non-RNG overload.
 * Deterministic control-flow without measure/reset also works on the non-RNG
 * overload. Only one-dimensional qtensors of qubits are supported. Nested
 * regions are walked; `scf.for` with concrete positive step, concrete
 * `scf.while`, and
 * non-recursive single-block `func.call` are supported. Loops are limited to
 * 10000 trips; multi-block function bodies remain unsupported. Consumes one
 * reference to @p in regardless of whether simulation succeeds or fails.
 *
 * @param func The QCO function to simulate
 * @param in The input state; one reference is consumed
 * @param dd The DD package to use
 * @param rng RNG used for collapsing measurements and resets
 * @param bindings Concrete values for symbolic scalar function arguments
 * @return The output statevector DD on success, or failure for unsupported
 *         programs
 */
FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng,
                                 const DDBindings& bindings = DDBindings());

/**
 * @brief Sample measurement outcomes from a QCO `func.func`.
 *
 * @details Starts from the all-zero state and draws @p shots bitstrings via
 * `Package::measureAll` (qubit `n-1` … `0`, same as @ref dd::sample). Programs
 * without `measure` / `reset` are simulated once and sampled without
 * collapsing (including deterministic control-flow). Programs with mid-circuit
 * `measure` / `reset` are re-simulated per shot with @p rng. Histograms are
 * final computational-basis bitstrings, not classical mid-circuit records;
 * they include dynamically allocated wires even after those wires are
 * deallocated.
 *
 * @param func The QCO function to sample
 * @param dd The DD package to use
 * @param shots Number of shots
 * @param rng RNG for collapsing measurements and non-collapsing sampling
 * @param bindings Concrete values for symbolic scalar function arguments
 * @return Histogram of outcome strings on success, or failure for unsupported
 *         programs
 */
FailureOr<std::map<std::string, std::size_t>>
sample(func::FuncOp func, dd::Package& dd, std::size_t shots,
       std::mt19937_64& rng, const DDBindings& bindings = DDBindings());

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
 * @param bindings Concrete values for symbolic scalar function arguments
 * @return Histogram of outcome strings on success, or failure for unsupported
 *         programs
 */
FailureOr<std::map<std::string, std::size_t>>
sample(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
       std::size_t shots, std::mt19937_64& rng,
       const DDBindings& bindings = DDBindings());

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
FailureOr<SampleResult>
sampleWithClassics(func::FuncOp func, dd::Package& dd, size_t shots,
                   std::mt19937_64& rng,
                   const DDBindings& bindings = DDBindings());

/// @copydoc sampleWithClassics(func::FuncOp, dd::Package&, size_t,
/// std::mt19937_64&)
/// Starts from @p in; one reference is consumed.
FailureOr<SampleResult>
sampleWithClassics(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
                   size_t shots, std::mt19937_64& rng,
                   const DDBindings& bindings = DDBindings());

} // namespace mlir::qco
