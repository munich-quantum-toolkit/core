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
 * Integer and floating-point attributes bind scalar function arguments. An
 * integer attribute bound to a dynamic one-dimensional qtensor argument gives
 * its runtime extent. Bindings for other values are rejected.
 */
using DDBindings = DenseMap<Value, Attribute>;

/**
 * @brief Sequentially build a matrix DD for a static unitary QCO `func.func`.
 *
 * @details Walks the concrete control-flow path through @p func, maps
 * `qco.static` SSA values to wire indices (or, if none are present,
 * qubit-typed function arguments as wires `0..n-1`), and applies unitary
 * operations via decision-diagram multiplication.
 *
 * Supported programs:
 * - Standard single-, two-, and three-qubit gates with constant or bound
 *   parameters (sparse DD path)
 * - `ctrl` with a sole standard-gate body (same sparse path)
 * - Other `UnitaryOpInterface` ops with a compile-time known matrix (`inv`,
 *   compound `ctrl`, ...), including `gphase` and `barrier`
 * - QTensor bookkeeping over existing input wires
 * - Concrete QCO and SCF control flow, multi-block ControlFlow CFGs, and
 *   non-recursive calls
 * - Concrete integer, index, floating-point, and common Math operations and
 *   one-dimensional memrefs of those scalar types
 * - `qco.static` establishes the wire map (or qubit-typed `func` args if none);
 *   `sink` is ignored; returned qubits and qtensors must preserve canonical
 *   wire order
 *
 * Known one-, two-, and three-qubit matrices are constructed directly as DD
 * gates. Larger compile-time unitaries are embedded directly into a DD over
 * their target wires, so idle register qubits do not enlarge the local matrix.
 * Quantum allocations, measurements, resets, unbound parameters, and
 * non-concrete control flow are not supported.
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
 * @brief Simulate a QCO `func.func` on a given input state without stochastic
 * collapse.
 *
 * @details Same supported unitary op set as @ref buildFunctionality, plus
 * concrete QCO and standard SCF control flow and static- or concrete
 * dynamic-shape one-dimensional memrefs of integer, index, or floating-point
 * values. `qco.alloc` and `qtensor.alloc` append zero-state wires. QTensor
 * extraction, insertion, deallocation, and transport through regions are
 * tracked with linear value semantics. Deallocating a separable QTensor
 * removes its wires from vector DDs; deallocating an entangled wire is
 * rejected. QTensor sizes and indices must be concrete; dynamic qtensor
 * arguments require an extent in @p bindings.
 * Mid-circuit `measure` / `reset` require the RNG overload below. Concrete-
 * bound `scf.for` and `scf.while` loops, multi-block `scf.execute_region`, and
 * non-recursive multi-block `func.call` are supported independently of RNG.
 * Loops and concrete CFG walks are limited to 10000 trips or transitions.
 * Consumes one reference to @p in regardless of success or failure.
 *
 * @param func The QCO function to simulate
 * @param in The input state, represented as a vector DD; one reference is
 * consumed
 * @param dd The DD package to use (must hold at least the function's qubits)
 * @param bindings Concrete values for symbolic function arguments
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
 * arithmetic, comparisons, casts, shifts, and `arith.select`). Dynamic quantum
 * allocation, qtensors, memrefs, loops, regions, and calls are supported as in
 * the non-RNG overload.
 * Consumes one reference to @p in regardless of success or failure.
 *
 * @param func The QCO function to simulate
 * @param in The input state; one reference is consumed
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
 * @brief Construct the density operator @f$|\psi\rangle\langle\psi|@f$.
 *
 * @param state Pure input state; its reference is retained by the caller
 * @param numQubits Number of active qubits represented by @p state
 * @param dd The DD package to use
 * @return A referenced matrix DD representing the pure-state density operator
 * @throws std::invalid_argument If @p numQubits does not cover the highest DD
 *         level in @p state or exceeds the capacity of @p dd
 */
dd::MatrixDD makeDensityMatrix(const dd::VectorDD& state, size_t numQubits,
                               dd::Package& dd);

/**
 * @brief Simulate a QCO function using a density-matrix DD.
 *
 * @details Unitary operations evolve the state as @f$U\rho U^\dagger@f$.
 * Qubit and qtensor deallocation performs a physical partial trace, including
 * for entangled qubits. The RNG overload additionally supports collapsing
 * measurement and reset. Consumes one reference to @p in regardless of
 * success or failure.
 *
 * @param func The QCO function to simulate
 * @param in Input density matrix; one reference is consumed
 * @param dd The DD package to use
 * @param bindings Concrete values for symbolic function arguments
 * @return The output density-matrix DD on success, or failure for unsupported
 *         programs
 */
FailureOr<dd::MatrixDD>
simulateDensity(func::FuncOp func, const dd::MatrixDD& in, dd::Package& dd,
                const DDBindings& bindings = DDBindings());

/// @copydoc simulateDensity(func::FuncOp, const dd::MatrixDD&, dd::Package&,
/// const DDBindings&)
/// Uses @p rng for collapsing measurement and reset.
FailureOr<dd::MatrixDD>
simulateDensity(func::FuncOp func, const dd::MatrixDD& in, dd::Package& dd,
                std::mt19937_64& rng,
                const DDBindings& bindings = DDBindings());

/**
 * @brief Sample measurement outcomes from a QCO `func.func`.
 *
 * @details Starts from the all-zero state and draws @p shots bitstrings via
 * `Package::measureAll` (qubit `n-1` … `0`, same as @ref dd::sample). Programs
 * without `measure` / `reset` are simulated once and sampled without
 * collapsing (including deterministic control-flow). Programs with mid-circuit
 * `measure` / `reset` are re-simulated per shot with @p rng. Histograms are
 * final computational-basis bitstrings, not classical mid-circuit records.
 * Deallocated separable QTensor wires are omitted from the sampled bitstrings.
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
 * @param bindings Concrete values for symbolic function arguments
 * @return Histogram of outcome strings on success, or failure for unsupported
 *         programs
 */
FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd, size_t shots,
       std::mt19937_64& rng, const DDBindings& bindings = DDBindings());

/**
 * @brief Sample a QCO function from an input density-matrix DD.
 *
 * @details Supports mixed states and entangled qubit deallocation. Each final
 * sample collapses a referenced copy of the simulated density state. Programs
 * with mid-circuit measurement or reset are re-simulated per shot. Consumes
 * one reference to @p in regardless of success, failure, or @p shots.
 */
FailureOr<std::map<std::string, size_t>>
sampleDensity(func::FuncOp func, const dd::MatrixDD& in, dd::Package& dd,
              size_t shots, std::mt19937_64& rng,
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
