/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/runtime/Runtime.hpp"

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "qir/runtime/QIR.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qir {

auto Runtime::generateRandomSeed() -> uint64_t {
  std::array<std::random_device::result_type, std::mt19937_64::state_size>
      randomData{};
  std::random_device rd;
  std::ranges::generate(randomData, std::ref(rd));
  std::seed_seq seeds(randomData.begin(), randomData.end());
  std::mt19937_64 rng(seeds);
  return rng();
}
Runtime& Runtime::getInstance() {
  static Runtime instance;
  return instance;
}
auto Runtime::reset() -> void {
  addressMode = AddressMode::UNKNOWN;
  qRegister.clear();
  rRegister.clear();
  // NOLINTBEGIN(performance-no-int-to-ptr)
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ZERO_ADDRESS),
                    ResultStruct{.refcount = 0, .r = false});
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ONE_ADDRESS),
                    ResultStruct{.refcount = 0, .r = true});
  // NOLINTEND(performance-no-int-to-ptr)
  measurements.clear();
  currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
  currentMaxQubitId = 0;
  currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
  qState.reset();
  mt.seed(generateRandomSeed());
}

Runtime::Runtime() : Runtime(generateRandomSeed()) {}

Runtime::Runtime(const uint64_t randomSeed)
    : addressMode(AddressMode::UNKNOWN),
      currentMaxQubitAddress(MIN_DYN_QUBIT_ADDRESS), currentMaxQubitId(0),
      currentMaxResultAddress(MIN_DYN_RESULT_ADDRESS), mt(randomSeed) {
  qRegister = std::unordered_map<const Qubit*, qc::Qubit>();
  rRegister = std::unordered_map<Result*, ResultStruct>();
  // NOLINTBEGIN(performance-no-int-to-ptr)
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ZERO_ADDRESS),
                    ResultStruct{.refcount = 0, .r = false});
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ONE_ADDRESS),
                    ResultStruct{.refcount = 0, .r = true});
  // NOLINTEND(performance-no-int-to-ptr)
}

auto Runtime::enlargeState(const std::uint64_t maxQubit) -> void {
  if (maxQubit >= qState.numQubits) {
    const auto d = maxQubit - qState.numQubits + 1;
    qubitPermutation.resize(qState.numQubits + d);
    std::iota(qubitPermutation.begin() + qState.numQubits,
              qubitPermutation.end(), qState.numQubits);
    qState.numQubits += static_cast<dd::Qubit>(d);

    // Resize the DD package only if necessary.
    if (qState.dd->qubits() < qState.numQubits) {
      qState.dd->resize(qState.numQubits);
    }

    // If the state is terminal, we need to create a new node.
    if (qState.edge.isTerminal()) {
      qState.edge = makeZeroState(d, *qState.dd);
      return;
    }

    // Enlarge state.
    // Each iteration adds one level above the current root, raising root.v by
    // one. After the loop, root.v == numQubits - 1.
    for (auto q = qState.edge.p->v; q + 1 < qState.numQubits; ++q) {
      auto old = qState.edge;
      qState.edge = qState.dd->makeDDNode(
          q + 1U, std::array{qState.edge, dd::vEdge::zero()});
      qState.dd->incRef(qState.edge);
      qState.dd->decRef(old);
    }
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
auto Runtime::swap(Qubit* qubit1, Qubit* qubit2) -> void {
  const auto target1 = translateAddresses(std::array{qubit1})[0];
  const auto target2 = translateAddresses(std::array{qubit2})[0];
  std::swap(qubitPermutation[target1], qubitPermutation[target2]);
}

auto Runtime::qAlloc() -> Qubit* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* qubit = reinterpret_cast<Qubit*>(currentMaxQubitAddress++);
  qRegister.emplace(qubit, currentMaxQubitId++);
  return qubit;
}

auto Runtime::qFree(Qubit* qubit) -> void {
  reset<1>({{qubit}});
  qRegister.erase(qubit);
}

auto Runtime::rAlloc() -> Result* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* result = reinterpret_cast<Result*>(currentMaxResultAddress++);
  rRegister.emplace(result, ResultStruct{.refcount = 1, .r = false});
  return result;
}

auto Runtime::rFree(Result* result) -> void { rRegister.erase(result); }

auto Runtime::deref(Result* result) -> ResultStruct& {
  auto it = rRegister.find(result);
  if (it == rRegister.end()) {
    if (addressMode != AddressMode::UNKNOWN) {
      addressMode = AddressMode::STATIC;
    }
    if (addressMode == AddressMode::DYNAMIC) {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__
         << ": Result not allocated (not found): " << result;
      throw std::out_of_range(ss.str());
    }
    it = rRegister.emplace(result, ResultStruct{.refcount = 0, .r = false})
             .first;
  }
  return it->second;
}

auto Runtime::equal(Result* result1, Result* result2) -> bool {
  return deref(result1).r == deref(result2).r;
}

auto Runtime::appendMeasurementBit(bool result) -> void {
  measurements.push_back(result ? '1' : '0');
}

auto Runtime::getMeasurements() const -> const std::string& {
  return measurements;
}

auto Runtime::takeState() -> QState {
  QState ret = std::move(qState);
  reset();
  return ret;
}

auto Runtime::getOstream() const -> std::ostream& { return *os; }

auto Runtime::setOstream(std::ostream& other) -> void { os = &other; }

auto Runtime::resetOstream() -> void { os = &std::cout; }

void Runtime::outputType(const char* type, std::string_view value,
                         const char* label) const {
  *os << "OUTPUT\t" << type << "\t" << value;
  if (label != nullptr && outputSchema == OutputSchema::Labeled) {
    *os << "\t" << label;
  }
  *os << "\n";
}

auto Runtime::outputResult(bool value, const char* label) const -> void {
  outputType("RESULT", value ? "1" : "0", label);
}

auto Runtime::outputBool(bool value, const char* label) const -> void {
  outputType("BOOL", value ? "true" : "false", label);
}

auto Runtime::outputInt(int64_t value, const char* label) const -> void {
  outputType("INT", std::to_string(value), label);
}

auto Runtime::outputFloat(double value, const char* label) const -> void {
  // Use std::ostringstream rather than std::to_string.
  // std::to_string formats with six digits after the decimal point and
  // can print 0.000000 for very small numbers.
  // std::ostringstream uses six significant digits by default and
  // outputs very small numbers with scientific notation.
  std::ostringstream oss;
  oss << value;
  outputType("DOUBLE", oss.str(), label);
}

auto Runtime::outputTuple(int64_t elementCount, const char* label) const
    -> void {
  outputType("TUPLE", std::to_string(elementCount), label);
}

auto Runtime::outputArray(int64_t elementCount, const char* label) const
    -> void {
  outputType("ARRAY", std::to_string(elementCount), label);
}

auto Runtime::outputProgramHeader() const -> void {
  *os << "HEADER\tschema_id\t" << outputSchema << "\n";
  *os << "HEADER\tschema_version\t2.1\n";
}

auto Runtime::outputShotStart() const -> void {
  *os << "START\n";
  *os << "METADATA\toutput_labeling_schema\t" << outputSchema << "\n";
}

auto Runtime::outputShotEnd() const -> void { *os << "END\t0\n"; }

auto Runtime::getOutputSchema() const -> OutputSchema { return outputSchema; }

auto Runtime::setOutputSchema(OutputSchema schema) -> void {
  outputSchema = schema;
}

auto operator<<(std::ostream& os, const Runtime::OutputSchema schema)
    -> std::ostream& {
  return os << (schema == Runtime::OutputSchema::Labeled ? "labeled"
                                                         : "ordered");
}

} // namespace qir
