/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QuantumState.hpp"

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::qco {

namespace {
/// Largest number of qubits a group can track (we use uint_64t as datatype).
constexpr unsigned MAX_GROUP_QUBITS = 63;
} // namespace

QuantumState::QuantumState(const ArrayRef<Value> qubits,
                           const size_t maxNonzeroAmplitudes)
    : maxNonzeroAmplitudes(maxNonzeroAmplitudes),
      qubits(qubits.begin(), qubits.end()) {
  if (qubits.size() > MAX_GROUP_QUBITS) {
    markTop();
    return;
  }
  amplitudes[0] = Complex{1.0, 0.0};
}

QuantumState QuantumState::singletonZero(const Value qubit,
                                         const size_t maxNonzeroAmplitudes) {
  return {ArrayRef(qubit), maxNonzeroAmplitudes};
}

std::optional<unsigned> QuantumState::indexOf(const Value q) const {
  for (const auto [idx, qubit] : llvm::enumerate(qubits)) {
    if (qubit == q) {
      return static_cast<unsigned>(idx);
    }
  }
  return std::nullopt;
}

uint64_t QuantumState::maskOf(const ArrayRef<Value> values) const {
  uint64_t mask = 0;
  for (const Value v : values) {
    if (const auto idx = indexOf(v)) {
      mask |= uint64_t{1} << *idx;
    }
  }
  return mask;
}

void QuantumState::markTop() {
  top = true;
  amplitudes.clear();
}

void QuantumState::forwardQubit(const Value from, const Value to) {
  if (const auto idx = indexOf(from)) {
    qubits[*idx] = to;
  }
}

void QuantumState::canonicalize() {
  if (top) {
    return;
  }
  SmallVector<uint64_t> negligible;
  for (const auto& [key, amp] : amplitudes) {
    if (std::abs(amp) <= MATRIX_TOLERANCE) {
      negligible.push_back(key);
    }
  }
  for (const uint64_t key : negligible) {
    amplitudes.erase(key);
  }
  if (amplitudes.size() > maxNonzeroAmplitudes) {
    markTop();
  }
}

LogicalResult QuantumState::applyMatrix1Q(const Value in, const Value out,
                                          const Matrix2x2& matrix,
                                          const ArrayRef<Value> ctrls) {
  const auto idx = indexOf(in);
  if (!idx) {
    return failure();
  }
  if (top) {
    forwardQubit(in, out);
    return success();
  }

  const uint64_t targetBit = uint64_t{1} << *idx;
  const uint64_t ctrlMask = maskOf(ctrls);

  llvm::DenseMap<uint64_t, Complex> result;
  for (const auto& [key, amp] : amplitudes) {
    if ((key & ctrlMask) != ctrlMask) {
      result[key] += amp;
      continue;
    }
    // Scatter this input's matrix column across both output rows.
    const uint64_t base = key & ~targetBit;
    const unsigned col = (key & targetBit) != 0 ? 1U : 0U;
    result[base] += matrix.data[col] * amp;
    result[base | targetBit] += matrix.data[2 + col] * amp;
  }

  amplitudes = std::move(result);
  forwardQubit(in, out);
  canonicalize();

  return success();
}

LogicalResult QuantumState::applyMatrix2Q(const Value in0, const Value in1,
                                          const Value out0, const Value out1,
                                          const Matrix4x4& matrix,
                                          const ArrayRef<Value> ctrls) {
  const auto idx0 = indexOf(in0);
  const auto idx1 = indexOf(in1);
  if (!idx0 || !idx1 || *idx0 == *idx1) {
    return failure();
  }
  if (top) {
    forwardQubit(in0, out0);
    forwardQubit(in1, out1);
    return success();
  }

  // QCO convention: the first target is the high bit of the local 4-index.
  const uint64_t hiBit = uint64_t{1} << *idx0;
  const uint64_t loBit = uint64_t{1} << *idx1;
  const uint64_t bothBits = hiBit | loBit;
  const uint64_t ctrlMask = maskOf(ctrls);

  const auto localKey = [&](const uint64_t base, const unsigned local) {
    return base | ((local & 1U) != 0U ? loBit : 0) |
           ((local & 2U) != 0U ? hiBit : 0);
  };
  const auto localCol = [&](const uint64_t key) {
    return ((key & hiBit) != 0 ? 2U : 0U) | ((key & loBit) != 0 ? 1U : 0U);
  };

  llvm::DenseMap<uint64_t, Complex> result;
  for (const auto& [key, amp] : amplitudes) {
    if ((key & ctrlMask) != ctrlMask) {
      result[key] += amp;
      continue;
    }
    // Scatter this input's matrix column across all four output rows.
    const uint64_t base = key & ~bothBits;
    const unsigned col = localCol(key);
    for (unsigned row = 0; row < 4; ++row) {
      result[localKey(base, row)] += matrix.data[(4 * row) + col] * amp;
    }
  }

  amplitudes = std::move(result);
  forwardQubit(in0, out0);
  forwardQubit(in1, out1);
  canonicalize();

  return success();
}

void QuantumState::applyGlobalPhase(const double phase,
                                    const ArrayRef<Value> ctrls) {
  if (top) {
    return;
  }
  const uint64_t ctrlMask = maskOf(ctrls);
  const Complex factor = std::exp(Complex{0.0, phase});
  for (auto& [key, amp] : amplitudes) {
    if ((key & ctrlMask) == ctrlMask) {
      amp *= factor;
    }
  }
  canonicalize();
}

FailureOr<SmallVector<MeasurementOutcome>>
QuantumState::measure(const Value target) const {
  const auto idx = indexOf(target);
  if (!idx) {
    return failure();
  }
  if (top) {
    return SmallVector<MeasurementOutcome>{};
  }
  const uint64_t targetBit = uint64_t{1} << *idx;

  llvm::DenseMap<uint64_t, Complex> zeroAmps;
  llvm::DenseMap<uint64_t, Complex> oneAmps;
  double probZero = 0.0;
  double probOne = 0.0;
  for (const auto& [key, amp] : amplitudes) {
    if ((key & targetBit) == 0) {
      zeroAmps[key] = amp;
      probZero += std::norm(amp);
    } else {
      oneAmps[key] = amp;
      probOne += std::norm(amp);
    }
  }

  const auto makeBranch = [&](const unsigned bit, const double probability,
                              const llvm::DenseMap<uint64_t, Complex>& amps) {
    auto branch =
        std::unique_ptr<QuantumState>(new QuantumState(maxNonzeroAmplitudes));
    branch->qubits = qubits;
    const double scale = 1.0 / std::sqrt(probability);
    for (const auto& [key, amp] : amps) {
      branch->amplitudes[key] += amp * scale;
    }
    branch->canonicalize();
    return MeasurementOutcome{.bit=bit, .probability=probability, .state=std::move(branch)};
  };

  SmallVector<MeasurementOutcome> outcomes;
  if (!zeroAmps.empty()) {
    outcomes.push_back(makeBranch(0, probZero, zeroAmps));
  }
  if (!oneAmps.empty()) {
    outcomes.push_back(makeBranch(1, probOne, oneAmps));
  }
  return outcomes;
}

FailureOr<SmallVector<MeasurementOutcome>>
QuantumState::reset(const Value target) const {
  auto outcomes = measure(target);
  if (failed(outcomes)) {
    return failure();
  }
  for (auto& outcome : *outcomes) {
    if (outcome.bit == 0 || outcome.state == nullptr) {
      continue;
    }
    const auto idx = outcome.state->indexOf(target);
    if (!idx) {
      return failure();
    }
    const uint64_t targetBit = uint64_t{1} << *idx;
    llvm::DenseMap<uint64_t, Complex> flipped;
    for (const auto& [key, amp] : outcome.state->amplitudes) {
      flipped[key & ~targetBit] += amp;
    }
    outcome.state->amplitudes = std::move(flipped);
    outcome.state->canonicalize();
  }
  return outcomes;
}

QuantumState QuantumState::unify(const QuantumState& that) const {
  QuantumState result(maxNonzeroAmplitudes);
  result.qubits.append(qubits.begin(), qubits.end());
  result.qubits.append(that.qubits.begin(), that.qubits.end());

  if (top || that.top ||
      result.qubits.size() > MAX_GROUP_QUBITS ||
      amplitudes.size() * that.amplitudes.size() > maxNonzeroAmplitudes) {
    result.markTop();
    return result;
  }

  const auto shift = qubits.size();
  for (const auto& [keyA, ampA] : amplitudes) {
    for (const auto& [keyB, ampB] : that.amplitudes) {
      result.amplitudes[keyA | keyB << shift] += ampA * ampB;
    }
  }
  result.canonicalize();
  return result;
}

bool QuantumState::isAlwaysZero(const Value q) const {
  const auto idx = indexOf(q);
  if (top || !idx || amplitudes.empty()) {
    return false;
  }
  return llvm::all_of(amplitudes, [&](const auto& entry) {
    return (entry.first >> *idx & uint64_t{1}) == 0;
  });
}

bool QuantumState::isAlwaysOne(const Value q) const {
  const auto idx = indexOf(q);
  if (top || !idx || amplitudes.empty()) {
    return false;
  }
  return llvm::all_of(amplitudes, [&](const auto& entry) {
    return (entry.first >> *idx & uint64_t{1}) == 1;
  });
}

bool QuantumState::hasAlwaysZeroAmplitude(
    const ArrayRef<std::pair<Value, bool>> basis) const {
  if (top) {
    return false;
  }
  uint64_t mask = 0;
  uint64_t wanted = 0;
  for (const auto& [qubit, one] : basis) {
    const auto idx = indexOf(qubit);
    if (!idx) {
      continue;
    }
    mask |= uint64_t{1} << *idx;
    if (one) {
      wanted |= uint64_t{1} << *idx;
    }
  }
  return llvm::all_of(amplitudes, [&](const auto& entry) {
    return (entry.first & mask) != wanted;
  });
}

bool QuantumState::operator==(const QuantumState& that) const {
  if (top || that.top) {
    return top == that.top;
  }
  if (maxNonzeroAmplitudes != that.maxNonzeroAmplitudes ||
      qubits.size() != that.qubits.size() ||
      amplitudes.size() != that.amplitudes.size()) {
    return false;
  }
  if (!std::equal(qubits.begin(), qubits.end(), that.qubits.begin())) {
    return false;
  }
  return llvm::all_of(amplitudes, [&](const auto& entry) {
    const auto it = that.amplitudes.find(entry.first);
    return it != that.amplitudes.end() &&
           std::abs(entry.second - it->second) <= MATRIX_TOLERANCE;
  });
}

void QuantumState::print(raw_ostream& os) const {
  if (qubits.empty()) {
    return;
  }

  const std::map ordered(amplitudes.begin(), amplitudes.end());
  bool first = true;
  for (const auto& [key, amp] : ordered) {
    if (!first) {
      os << ", ";
    }
    first = false;

    os << '|';
    for (size_t bit = qubits.size(); bit-- > 0;) {
      os << (((key >> bit) & uint64_t{1}) != 0 ? '1' : '0');
    }
    os << "> -> ";

    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%.2f", amp.real());
    const llvm::StringRef real(buf.data());
    os << (real == "-0.00" ? llvm::StringRef("0.00") : real);

    if (std::abs(amp.imag()) > MATRIX_TOLERANCE) {
      os << (amp.imag() > 0 ? " + i" : " - i");
      std::snprintf(buf.data(), buf.size(), "%.2f", std::abs(amp.imag()));
      os << buf.data();
    }
  }
}

} // namespace mlir::qco
