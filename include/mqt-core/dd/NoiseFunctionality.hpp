/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dd {

using NrEdges = std::tuple_size<decltype(dNode::e)>;
using ArrayOfEdges = std::array<dCachedEdge, NrEdges::value>;

// noise operations available for deterministic noise aware quantum circuit
// simulation
enum NoiseOperations : std::uint8_t {
  AmplitudeDamping,
  PhaseFlip,
  Depolarization,
  Identity
};

void sanityCheckOfNoiseProbabilities(double noiseProbability,
                                     double amplitudeDampingProb,
                                     double multiQubitGateFactor);

class StochasticNoiseFunctionality {
public:
  StochasticNoiseFunctionality(Package& dd, std::size_t nq,
                               double gateNoiseProbability,
                               double amplitudeDampingProb,
                               double multiQubitGateFactor,
                               const std::string& cNoiseEffects);

  ~StochasticNoiseFunctionality() { package->decRef(identityDD); }

protected:
  Package* package;
  std::size_t nQubits;
  std::uniform_real_distribution<fp> dist;

  double noiseProbability;
  double noiseProbabilityMulti;
  fp sqrtAmplitudeDampingProbability;
  fp oneMinusSqrtAmplitudeDampingProbability;
  fp sqrtAmplitudeDampingProbabilityMulti;
  fp oneMinusSqrtAmplitudeDampingProbabilityMulti;
  GateMatrix ampDampingTrue{};
  GateMatrix ampDampingTrueMulti{};
  GateMatrix ampDampingFalse{};
  GateMatrix ampDampingFalseMulti{};
  std::vector<NoiseOperations> noiseEffects;
  mEdge identityDD;

  [[nodiscard]] std::size_t getNumberOfQubits() const { return nQubits; }
  [[nodiscard]] double getNoiseProbability(bool multiQubitNoiseFlag) const;

  [[nodiscard]] static qc::OpType
  getAmplitudeDampingOperationType(bool multiQubitNoiseFlag,
                                   bool amplitudeDampingFlag);

  [[nodiscard]] GateMatrix
  getAmplitudeDampingOperationMatrix(bool multiQubitNoiseFlag,
                                     bool amplitudeDampingFlag) const;

public:
  void applyNoiseOperation(const std::set<qc::Qubit>& targets, mEdge operation,
                           vEdge& state, std::mt19937_64& generator);

protected:
  [[nodiscard]] mEdge stackOperation(const mEdge& operation, qc::Qubit target,
                                     qc::OpType noiseOperation,
                                     const GateMatrix& matrix) const;

  mEdge generateNoiseOperation(mEdge operation, qc::Qubit target,
                               std::mt19937_64& generator,
                               bool amplitudeDamping, bool multiQubitOperation);

  [[nodiscard]] qc::OpType returnNoiseOperation(NoiseOperations noiseOperation,
                                                double prob,
                                                bool multiQubitNoiseFlag) const;
};

class DeterministicNoiseFunctionality {
public:
  DeterministicNoiseFunctionality(Package& dd, std::size_t nq,
                                  double noiseProbabilitySingleQubit,
                                  double noiseProbabilityMultiQubit,
                                  double ampDampProbSingleQubit,
                                  double ampDampProbMultiQubit,
                                  const std::string& cNoiseEffects);

protected:
  Package* package;
  std::size_t nQubits;

  double noiseProbSingleQubit;
  double noiseProbMultiQubit;
  double ampDampingProbSingleQubit;
  double ampDampingProbMultiQubit;

  std::vector<NoiseOperations> noiseEffects;

  [[nodiscard]] std::size_t getNumberOfQubits() const { return nQubits; }

public:
  void applyNoiseEffects(dEdge& originalEdge,
                         const std::unique_ptr<qc::Operation>& qcOperation);

private:
  dCachedEdge applyNoiseEffects(dEdge& originalEdge,
                                const std::set<qc::Qubit>& usedQubits,
                                bool firstPathEdge, Qubit level);

  static void applyPhaseFlipToEdges(ArrayOfEdges& e, double probability);

  void applyAmplitudeDampingToEdges(ArrayOfEdges& e, double probability) const;

  void applyDepolarisationToEdges(ArrayOfEdges& e, double probability) const;
};

} // namespace dd
