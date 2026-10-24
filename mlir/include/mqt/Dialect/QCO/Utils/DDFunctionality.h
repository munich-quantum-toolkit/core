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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace mlir::qco {

/// Concrete values for QCO DD entry arguments.
///
/// Values must be integer, index, or `f64` attributes. An index value sets the
/// size of a dynamic one-dimensional QTensor argument.
using DDArgumentBindings = DenseMap<Value, Attribute>;

/// Execution budget shared by nested while loops and calls in one simulation.
///
/// Counted loops and the total number of gates are not bounded by this budget.
struct DDExecutionOptions {
  size_t maxWhileIterations = 1'000'000'000;
};

/// An uncollapsed sampling state and the package that owns its DD nodes.
///
/// A null package means that sampling did not retain a state.
struct DDSamplingState {
  std::unique_ptr<dd::Package> dd;
  dd::VectorDD state{};
};

/// Build a matrix DD for a unitary QCO function.
///
/// The function must have one block. The interpreter supports concrete QCO and
/// SCF structured control, non-recursive calls, common scalar math,
/// identity-layout heap or stack memrefs of integer, index, or f64 values and
/// their strided subviews, dense rank-one integer or f64 tensors and element
/// extraction, and QTensor bookkeeping. Copies between distinct memref views
/// require non-overlapping element locations. `qco.static` values set the wire
/// map; without them, qubit arguments set it. Entry-block `qco.alloc` and
/// statically sized `qtensor.alloc` operations add subsequent wires in
/// instruction order. Measurements, resets, symbolic control, and other runtime
/// allocation are not supported.
///
/// Runtime-bound parameters are supported for standard gates, ordinary and
/// gate-function calls, a sole standard gate inside `qco.ctrl`, and `qco.pow`
/// with a compile-time-known body matrix. Other composite modifiers and custom
/// matrices must have a compile-time-known matrix.
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
    const DDArgumentBindings& argumentBindings = DDArgumentBindings(),
    const DDExecutionOptions& options = {});

/// Simulate a single-block QCO function.
///
/// In addition to the operations supported by `buildFunctionality`, simulation
/// supports measurements, resets, CBit registers, and runtime qubit and QTensor
/// allocation. QCO and SCF structured control requires concrete values. A
/// configurable budget of one billion iterations bounds `scf.while` execution
/// per simulation (per shot for dynamic sampling). Counted
/// loops, branches, and nonrecursive calls have no step limit. `qco.sink` and
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
         const DDArgumentBindings& argumentBindings = DDArgumentBindings(),
         const DDExecutionOptions& options = {});

/// Simulate a zero-state QCO function without collapsing terminal
/// measurements.
///
/// The function must have one block. Measurement results can be unused,
/// returned directly, or stored in returned CBit registers. No later operation
/// can use a measured wire.
/// Called functions cannot measure or reset qubits.
///
/// @param argumentBindings Scalar values and dynamic QTensor argument sizes.
FailureOr<dd::VectorDD> simulateStatevector(
    func::FuncOp func, dd::Package& dd,
    const DDArgumentBindings& argumentBindings = DDArgumentBindings(),
    const DDExecutionOptions& options = {});

/// Sample a single-block QCO function from the zero state.
///
/// Returned CBit registers and Boolean values use conventional count-string
/// order: the last result comes first, and each register is MSB-first. Other
/// classical output types are rejected. Without classical results, the function
/// samples all DD wires. Terminal measurements use
/// one simulation for all shots. Adaptive programs share shot prefixes when
/// their measurements and resets are in the top-level stream or
/// `scf.for`/`scf.while` loops. Other adaptive programs run once per shot
/// across independent workers.
///
/// The containing module must pass MLIR verification and
/// `qco::verifyLinearity`.
///
/// @param func QCO function to sample.
/// @param shots Number of samples.
/// @param seed RNG seed. Zero selects nondeterministic seeding.
/// @param argumentBindings Scalar values and dynamic QTensor argument sizes.
/// @param shotResults Optional individual outcomes in unspecified order,
/// cleared then filled during sampling. On failure, it may contain an
/// incomplete sequence.
/// @param retainedState Optional output, cleared before sampling and populated
/// only after successful terminal sampling. It owns the uncollapsed state.
/// @param workers Maximum workers for adaptive sampling. Zero selects up to
/// eight workers from the shot count and host budget; one uses one worker.
/// Terminal sampling uses one simulation for all shots.
/// @return Outcome counts, or failure for an unsupported program.
FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, size_t shots, uint64_t seed = 0,
       const DDArgumentBindings& argumentBindings = DDArgumentBindings(),
       std::vector<std::string>* shotResults = nullptr,
       DDSamplingState* retainedState = nullptr,
       const DDExecutionOptions& options = {}, size_t workers = 0);

/// Samples outcome counts with an external worker grant.
///
/// The caller holds one external worker slot. The callback runs at most once
/// and returns a total grant from one through the request; the caller releases
/// that grant after sampling.
FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, size_t shots, uint64_t seed,
       const DDArgumentBindings& argumentBindings,
       std::vector<std::string>* shotResults, DDSamplingState* retainedState,
       const DDExecutionOptions& options, size_t workers,
       function_ref<size_t(size_t)> requestWorkers);

/// Sample individual outcomes without constructing a histogram.
///
/// Uses the same program, seed, output, and worker contracts as `sample`.
/// With `requestWorkers`, the caller must already hold one external worker
/// slot. The callback runs at most once, only for parallel adaptive sampling,
/// and returns the total grant (including that slot), between one and the
/// request. The caller releases the external grant after sampling returns.
LogicalResult
sampleShots(func::FuncOp func, size_t shots, uint64_t seed,
            std::vector<std::string>& shotResults,
            const DDArgumentBindings& argumentBindings = DDArgumentBindings(),
            DDSamplingState* retainedState = nullptr,
            const DDExecutionOptions& options = {}, size_t workers = 0,
            function_ref<size_t(size_t)> requestWorkers = nullptr);
} // namespace mlir::qco
