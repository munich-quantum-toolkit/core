/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Execution/Runtime/Runtime.h"

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "mlir/Dialect/QIR/Execution/Runtime/QIR.h"
#include "mlir/Dialect/QIR/QIRDefinitions.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qir {

namespace {
thread_local Runtime* ActiveRuntime = nullptr;
} // namespace

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
  if (ActiveRuntime != nullptr) {
    return *ActiveRuntime;
  }
  static thread_local Runtime fallback;
  return fallback;
}

auto Runtime::bind(Runtime* runtime) noexcept -> Runtime* {
  return std::exchange(ActiveRuntime, runtime);
}

auto Runtime::reset() -> void {
  qubitMode = ResourceMode::UNKNOWN;
  resultMode = ResourceMode::UNKNOWN;
  qRegister.clear();
  qubitPermutation.clear();
  rRegister.clear();
  measurements.clear();
  currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
  currentMaxQubitId = 0;
  currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
  qState.reset();
}

auto Runtime::seed(const uint64_t randomSeed) -> void { mt.seed(randomSeed); }

Runtime::Runtime() : Runtime(generateRandomSeed()) {}

Runtime::Runtime(const uint64_t randomSeed)
    : qubitMode(ResourceMode::UNKNOWN), resultMode(ResourceMode::UNKNOWN),
      currentMaxQubitAddress(MIN_DYN_QUBIT_ADDRESS), currentMaxQubitId(0),
      currentMaxResultAddress(MIN_DYN_RESULT_ADDRESS), mt(randomSeed) {
  qRegister = std::unordered_map<const Qubit*, qc::Qubit>();
  rRegister = std::unordered_map<Result*, ResultStruct>();
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

auto Runtime::translateAddresses(const std::span<Qubit* const> qubits)
    -> std::vector<qc::Qubit> {
  std::vector<qc::Qubit> qubitIds(qubits.size());
  if (qubitMode != ResourceMode::STATIC) {
    try {
      std::ranges::transform(qubits, qubitIds.begin(), [&](const auto* q) {
        try {
          return qRegister.at(q);
        } catch (const std::out_of_range&) {
          std::ostringstream ss;
          ss << __FILE__ << ":" << __LINE__
             << ": Qubit not allocated (not found): " << q;
          throw std::out_of_range(ss.str());
        }
      });
    } catch (std::out_of_range&) {
      if (qubitMode == ResourceMode::DYNAMIC) {
        throw;
      }
      qubitMode = ResourceMode::STATIC;
    }
  }
  if (qubitMode == ResourceMode::STATIC) {
    std::ranges::transform(qubits, qubitIds.begin(), [](const auto* q) {
      return static_cast<qc::Qubit>(reinterpret_cast<uintptr_t>(q));
    });
  }
  if (!qubitIds.empty()) {
    enlargeState(*std::ranges::max_element(qubitIds));
  }
  return qubitIds;
}

auto Runtime::apply(const qc::OpType op, const std::span<const qc::fp> params,
                    const std::span<Qubit* const> controls,
                    const std::span<Qubit* const> targets) -> void {
  std::vector<Qubit*> qubits;
  qubits.reserve(controls.size() + targets.size());
  qubits.insert(qubits.end(), controls.begin(), controls.end());
  qubits.insert(qubits.end(), targets.begin(), targets.end());
  auto addresses = translateAddresses(qubits);
  std::ranges::transform(addresses, addresses.begin(), [&](const auto address) {
    return qubitPermutation[address];
  });

  if (op == qc::SWAP && controls.empty() && targets.size() == 2) {
    swap(targets[0], targets[1]);
    return;
  }

  const auto controlEnd =
      addresses.cbegin() + static_cast<std::ptrdiff_t>(controls.size());
  const qc::Controls mappedControls(addresses.cbegin(), controlEnd);
  const qc::Targets mappedTargets(controlEnd, addresses.cend());
  const qc::StandardOperation operation(
      mappedControls, mappedTargets, op,
      std::vector<qc::fp>(params.begin(), params.end()));
  qState.edge = applyUnitaryOperation(operation, qState.edge, *qState.dd);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
auto Runtime::swap(Qubit* qubit1, Qubit* qubit2) -> void {
  const auto target1 = translateAddresses(std::array{qubit1})[0];
  const auto target2 = translateAddresses(std::array{qubit2})[0];
  std::swap(qubitPermutation[target1], qubitPermutation[target2]);
}

auto Runtime::qAlloc() -> Qubit* {
  if (qubitMode == ResourceMode::STATIC) {
    throw std::logic_error(
        "Cannot dynamically allocate qubits after using static qubit IDs");
  }
  qubitMode = ResourceMode::DYNAMIC;
  auto* qubit = reinterpret_cast<Qubit*>(currentMaxQubitAddress++);
  qRegister.emplace(qubit, currentMaxQubitId++);
  return qubit;
}

auto Runtime::qFree(Qubit* qubit) -> void {
  if (qubitMode != ResourceMode::DYNAMIC || !qRegister.contains(qubit)) {
    throw std::out_of_range("QIR qubit was not dynamically allocated");
  }
  reset<1>({{qubit}});
  qRegister.erase(qubit);
}

auto Runtime::rAlloc() -> Result* {
  if (resultMode == ResourceMode::STATIC) {
    throw std::logic_error(
        "Cannot dynamically allocate results after using static result IDs");
  }
  resultMode = ResourceMode::DYNAMIC;
  auto* result = reinterpret_cast<Result*>(currentMaxResultAddress++);
  rRegister.emplace(result, ResultStruct{.r = false});
  return result;
}

auto Runtime::rFree(Result* result) -> void {
  if (resultMode != ResourceMode::DYNAMIC || rRegister.erase(result) == 0) {
    throw std::out_of_range("QIR result was not dynamically allocated");
  }
}

auto Runtime::deref(Result* result) -> ResultStruct& {
  auto it = rRegister.find(result);
  if (it == rRegister.end()) {
    if (resultMode == ResourceMode::DYNAMIC) {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__
         << ": Result not allocated (not found): " << result;
      throw std::out_of_range(ss.str());
    }
    resultMode = ResourceMode::STATIC;
    it = rRegister.emplace(result, ResultStruct{.r = false}).first;
  }
  return it->second;
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

auto Runtime::outputResultArray(const std::string_view values,
                                const char* label) const -> void {
  outputType("RESULT_ARRAY", values, label);
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
  if (metadata.empty()) {
    *os << "METADATA\toutput_labeling_schema\t" << outputSchema << "\n";
    return;
  }
  for (const auto& [name, value] : metadata) {
    *os << "METADATA\t" << name;
    if (!value.empty()) {
      *os << "\t" << value;
    }
    *os << "\n";
  }
}

auto Runtime::outputShotEnd(const int64_t exitCode) const -> void {
  *os << "END\t" << exitCode << "\n";
}

auto Runtime::getOutputSchema() const -> OutputSchema { return outputSchema; }

auto Runtime::setOutputSchema(OutputSchema schema) -> void {
  outputSchema = schema;
}

auto Runtime::setMetadata(
    std::vector<std::pair<std::string, std::string>> entryPointMetadata)
    -> void {
  metadata = std::move(entryPointMetadata);
}

auto operator<<(std::ostream& os, const Runtime::OutputSchema schema)
    -> std::ostream& {
  return os << (schema == Runtime::OutputSchema::Labeled ? LABELED_SCHEMA
                                                         : ORDERED_SCHEMA);
}

} // namespace qir
