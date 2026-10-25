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

#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

namespace mlir::qco {

/// A completed proof, a counterexample, or an explicitly unfinished check.
enum class VerificationVerdict {
  Equivalent,
  NotEquivalent,
  Inconclusive,
  Unsupported,
};

/// Native DD checking methods. Auto schedules the eligible methods.
enum class VerificationMethod { Auto, Simulation, Construction, Alternating };

/// Whether a coherent comparison preserves its global phase.
enum class VerificationPhase { Preserve, Ignore };

/// Compared mathematical quantity.
///
/// Exact compares the maximum absolute matrix-entry residual. ProcessDistance
/// is sqrt(1 - |Tr(A* B)/d|^2) and ignores phase; AmplitudeDistance is
/// ||A-B||_F/sqrt(d) and preserves it. TerminalProbabilities compares outcome
/// effects for all inputs, not reduced quantum states. Here A* is the adjoint
/// and d is the dimension of the free quantum input space.
enum class VerificationRelation {
  Exact,
  ProcessDistance,
  AmplitudeDistance,
  TerminalProbabilities,
};

/// How entry allocations contribute to the compared quantum input space.
enum class VerificationDomain { CircuitInputs, InitializedProgram };

/// A common interface shared by both programs.
///
/// CircuitInputs leaves all source data wires free except zeroAncillas.
/// InitializedProgram initializes allocations to zero; quantum arguments remain
/// free. numInputs asserts the source wire count, before routing workspaces.
/// zeroAncillas contains source slot indices. observedOutputs selects terminal
/// classical output labels for TerminalProbabilities; empty selects all
/// outputs. Coherent checks preserve every source output and require extra
/// workspaces to return to zero. A sink alone never authorizes discarding
/// quantum information.
struct VerificationContract {
  VerificationDomain domain = VerificationDomain::CircuitInputs;
  std::optional<size_t> numInputs;
  std::vector<size_t> zeroAncillas;
  std::vector<std::string> observedOutputs;
};

/// Cooperative bounds for one verification call, including trace resolution.
///
/// Zero timeout disables the deadline. DD bytes count requested backing
/// storage, not process RSS. Execution bytes bound resolved gate and classical
/// allocation payloads. Interpreter metadata is bounded by maxSteps; DD
/// comparison scratch uses the DD byte budget. Neither budget bounds the MLIR
/// context or process RSS. Workers also share the process budget
/// MQT_CORE_DD_WORKER_BUDGET with sampling.
struct VerificationLimits {
  std::chrono::milliseconds timeout{0};
  size_t maxSteps = 1'000'000;
  size_t maxCallDepth = 128;
  size_t maxGates = 1'000'000;
  size_t maxExecutionBytes = 256ULL * 1024 * 1024;
  size_t maxDDBytes = 512ULL * 1024 * 1024;
  size_t maxQubits = 256;
  size_t maxOutcomes = 4096;
  size_t workers = 2;
};

/// Relation, acceptance threshold, and bounded search settings.
///
/// tolerance is the accepted synthesis distance for approximate relations and
/// the entry residual for Exact and TerminalProbabilities. numericalTolerance
/// marks uncertainty around approximate thresholds; it does not change DD
/// canonicalization. It is an engineering guard, not a certified rounding
/// bound. ProcessDistance requires Ignore; AmplitudeDistance requires Preserve.
/// Passing simulation samples never proves all-input equivalence.
struct VerificationOptions {
  VerificationMethod method = VerificationMethod::Auto;
  VerificationRelation relation = VerificationRelation::Exact;
  VerificationPhase phase = VerificationPhase::Preserve;
  double tolerance = 1e-12;
  double numericalTolerance = 1e-12;
  VerificationLimits limits;
  size_t maxSimulationSamples = 32;
  uint64_t seed = 0;
};

/// One checker's outcome; all state is owned independently of its DD package.
struct VerificationCheckerResult {
  VerificationMethod method = VerificationMethod::Construction;
  VerificationVerdict verdict = VerificationVerdict::Inconclusive;
  std::string reason;
  std::optional<double> distance;
  std::optional<std::complex<double>> phase;
  /// Product-state counterexample in DD BasisStates order, when found.
  std::vector<unsigned> counterexample;
  size_t samples = 0;
  double seconds = 0;
};

/// Evidence for the supplied concrete bindings and shared quantum interface.
///
/// This result has no Boolean conversion. Inconclusive includes cancellation,
/// resource exhaustion, numerical uncertainty, and successful finite sampling.
/// No statement about other classical bindings follows from Equivalent.
struct VerificationResult {
  VerificationVerdict verdict = VerificationVerdict::Inconclusive;
  std::string reason;
  VerificationRelation relation = VerificationRelation::Exact;
  VerificationPhase phase = VerificationPhase::Preserve;
  VerificationContract contract;
  double tolerance = 0;
  double numericalTolerance = 0;
  std::vector<std::string> leftBindings;
  std::vector<std::string> rightBindings;
  size_t numInputs = 0;
  size_t numFreeInputs = 0;
  size_t numQubits = 0;
  size_t resolvedGates = 0;
  size_t peakDDBytes = 0;
  double ddTolerance = 0;
  std::optional<double> distance;
  std::vector<std::string> outputLabels;
  std::vector<VerificationCheckerResult> checkers;
};

/// Verify two explicit QCO entry functions without changing either program.
///
/// Both containing modules must verify, including QCO linearity. Concrete SCF
/// execution is resolved once with the supplied bindings. Measurement-dependent
/// control, resets, unresolved inputs, and general channel equivalence are
/// unsupported. Terminal readout identity is part of every coherent comparison.
/// Concurrent verification calls do not alter DD numerical settings; external
/// changes to the legacy global DD tolerance must not overlap verification.
FailureOr<VerificationResult>
verify(func::FuncOp left, func::FuncOp right,
       const VerificationContract& contract = {},
       const DDArgumentBindings& leftBindings = DDArgumentBindings(),
       const DDArgumentBindings& rightBindings = DDArgumentBindings(),
       const VerificationOptions& options = {}, std::stop_token stop = {});

} // namespace mlir::qco
