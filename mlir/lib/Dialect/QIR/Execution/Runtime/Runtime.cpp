/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/DDAdapter.h"
#include "mqt/Dialect/QIR/Execution/Runtime/QIR.h"
#include "mqt/Dialect/QIR/QIRDefinitions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
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
  qubitMode = staticQubits_ ? ResourceMode::STATIC : ResourceMode::UNKNOWN;
  resultMode = staticResults_ ? ResourceMode::STATIC : ResourceMode::UNKNOWN;
  qRegister.clear();
  freeQubits_.clear();
  qubitPermutation.clear();
  rRegister.clear();
  std::ranges::fill(resultValues_, ResultStruct{});
  measurements.clear();
  measuredQubits_.clear();
  invalidStateExtraction_ = false;
  currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
  currentMaxQubitId = 0;
  currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
  qState.reset();
  if (qState.dd && staticQubits_ && *staticQubits_ != 0) {
    enlargeState(*staticQubits_ - 1);
  }
}

void Runtime::configureStaticResources(std::optional<size_t> qubits,
                                       std::optional<size_t> results) {
  if (qubits && *qubits > dd::Package::MAX_POSSIBLE_QUBITS) {
    throw std::out_of_range(
        "Static QIR qubit capacity exceeds the supported range");
  }
  if (results && *results > resultValues_.max_size()) {
    throw std::out_of_range(
        "Static QIR result capacity exceeds the supported range");
  }
  staticQubits_ = qubits;
  staticResults_ = results;
  resultValues_.resize(results.value_or(0));
  reset();
}

auto Runtime::seed(const uint64_t randomSeed) -> void { mt.seed(randomSeed); }

Runtime::Runtime() : Runtime(generateRandomSeed()) {}

Runtime::Runtime(const uint64_t randomSeed)
    : qubitMode(ResourceMode::UNKNOWN), resultMode(ResourceMode::UNKNOWN),
      currentMaxQubitAddress(MIN_DYN_QUBIT_ADDRESS), currentMaxQubitId(0),
      currentMaxResultAddress(MIN_DYN_RESULT_ADDRESS), mt(randomSeed) {}

auto Runtime::enlargeState(size_t maxQubit) -> void {
  if (maxQubit >= dd::Package::MAX_POSSIBLE_QUBITS ||
      (staticQubits_ && maxQubit >= *staticQubits_)) {
    throw std::out_of_range("QIR qubit ID exceeds the supported qubit range");
  }
  if (staticQubits_) {
    maxQubit = std::max(maxQubit, *staticQubits_ - 1);
  }
  if (maxQubit < qState.numQubits) {
    return;
  }
  const auto numQubits = maxQubit + 1;
  const auto capacity = qState.dd ? qState.dd->qubits() : 0;
  if (capacity < numQubits) {
    /// Unknown resources grow geometrically; declared resources fit exactly.
    const auto newCapacity = staticQubits_.value_or(std::min(
        dd::Package::MAX_POSSIBLE_QUBITS,
        std::max({numQubits, dd::Package::DEFAULT_QUBITS, 2 * capacity})));
    if (!qState.dd) {
      qState.dd = std::make_unique<dd::Package>(newCapacity);
    } else {
      qState.dd->resize(newCapacity);
    }
  }
  qubitPermutation.resize(numQubits);
  std::iota(qubitPermutation.begin() + static_cast<ptrdiff_t>(qState.numQubits),
            qubitPermutation.end(), static_cast<dd::Qubit>(qState.numQubits));

  /// Extending the root preserves its weight, including a scalar global phase.
  auto edge = qState.edge;
  for (auto q = qState.numQubits; q < numQubits; ++q) {
    edge = qState.dd->makeDDNode(static_cast<dd::Qubit>(q),
                                 std::array{edge, dd::vEdge::zero()});
  }
  qState.dd->incRef(edge);
  qState.dd->decRef(qState.edge);
  qState.edge = edge;
  qState.numQubits = numQubits;
}

auto Runtime::resolveAddress(const Qubit* qubit) -> dd::Qubit {
  if (qubitMode == ResourceMode::UNKNOWN) {
    qubitMode = ResourceMode::STATIC;
  }
  if (qubitMode == ResourceMode::STATIC) {
    const auto id = staticQubitId(qubit);
    if (staticQubits_ && id >= *staticQubits_) {
      throw std::out_of_range(
          "Static QIR qubit ID exceeds its declared capacity");
    }
    return id;
  }

  const auto it = qRegister.find(qubit);
  if (it == qRegister.end()) {
    std::ostringstream ss;
    ss << __FILE__ << ":" << __LINE__
       << ": Qubit not allocated (not found): " << qubit;
    throw std::out_of_range(ss.str());
  }
  return it->second;
}

auto Runtime::translateAddresses(const std::span<Qubit* const> qubits,
                                 const std::span<Qubit* const> additionalQubits)
    -> llvm::SmallVector<dd::Qubit, 5> {
  llvm::SmallVector<dd::Qubit, 5> qubitIds;
  qubitIds.reserve(qubits.size() + additionalQubits.size());
  for (const auto* qubit : qubits) {
    qubitIds.push_back(resolveAddress(qubit));
  }
  for (const auto* qubit : additionalQubits) {
    qubitIds.push_back(resolveAddress(qubit));
  }
  if (extractState_ && std::ranges::any_of(qubitIds, [&](const auto id) {
        return measuredQubits_.contains(id);
      })) {
    /// Report after JIT execution returns without unwinding generated frames.
    invalidStateExtraction_ = true;
  }
  if (!qubitIds.empty()) {
    enlargeState(*std::ranges::max_element(qubitIds));
  }
  return qubitIds;
}

auto Runtime::apply(const std::span<const std::complex<dd::fp>> matrix,
                    std::span<Qubit* const> controls,
                    std::span<Qubit* const> targets) -> void {
  auto addresses = translateAddresses(controls, targets);
  if (invalidStateExtraction_) {
    return;
  }
  if (!qState.dd) {
    qState.dd = std::make_unique<dd::Package>(0);
  }
  std::ranges::transform(addresses, addresses.begin(), [&](const auto address) {
    return qubitPermutation[address];
  });

  const llvm::ArrayRef mappedAddresses(addresses);
  const auto mappedTargets = mappedAddresses.drop_front(controls.size());
  const dd::Controls mappedControls(mappedAddresses.begin(),
                                    mappedTargets.begin());
  qState.edge = qState.dd->applyOperation(
      mlir::qco::makeGateDD(*qState.dd, matrix, qState.numQubits, mappedTargets,
                            mappedControls),
      qState.edge);
}

auto Runtime::applyGlobalPhase(dd::fp phase) -> void {
  if (!qState.dd) {
    qState.dd = std::make_unique<dd::Package>(0);
  }
  qState.edge = dd::applyGlobalPhase(qState.edge, phase, *qState.dd);
}

auto Runtime::measure(Qubit* qubit, Result* result) -> void {
  const auto target = resolveAddress(qubit);
  enlargeState(target);
  auto& value = deref(result);
  if (extractState_) {
    measuredQubits_.insert(target);
  } else if (!deferMeasurements_) {
    value.r = qState.dd->measureOneCollapsing(
                  qState.edge, qubitPermutation[target], mt) == '1';
  }
}

auto Runtime::sampleMeasurements(std::span<const uintptr_t> qubits,
                                 size_t shots,
                                 std::vector<std::string>& results) -> void {
  measurements.clear();
  if (qubits.empty()) {
    results.resize(shots);
    return;
  }
  bool ascending = qubits.size() == qState.numQubits;
  for (size_t i = 0; ascending && i < qubits.size(); ++i) {
    ascending = qubitPermutation[qubits[i]] == i;
  }
  if (ascending) {
    for (size_t i = 0; i < shots; ++i) {
      auto basis = qState.dd->measureAll(qState.edge, false, mt);
      std::ranges::reverse(basis);
      results.push_back(std::move(basis));
    }
    if (shots != 0) {
      measurements = results.back();
    }
    return;
  }
  measurements.reserve(qubits.size());
  for (size_t i = 0; i < shots; ++i) {
    const auto basis = qState.dd->measureAll(qState.edge, false, mt);
    measurements.clear();
    for (const auto qubit : qubits) {
      measurements.push_back(basis[basis.size() - 1 - qubitPermutation[qubit]]);
    }
    results.push_back(measurements);
  }
}

auto Runtime::reset(std::span<Qubit* const> qubits) -> void {
  if (extractState_) {
    invalidStateExtraction_ = true;
    return;
  }
  auto targets = translateAddresses(qubits);
  std::ranges::transform(targets, targets.begin(), [&](const auto target) {
    return qubitPermutation[target];
  });
  const auto matrix = mlir::qco::getStandardGateMatrix<mlir::qco::XOp>({});
  for (const auto target : targets) {
    if (qState.dd->measureOneCollapsing(qState.edge, target, mt) == '1') {
      const std::array targetArray{target};
      qState.edge = qState.dd->applyOperation(
          mlir::qco::makeGateDD(*qState.dd, matrix, qState.numQubits,
                                targetArray),
          qState.edge);
    }
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
auto Runtime::swap(Qubit* qubit1, Qubit* qubit2) -> void {
  const auto targets = translateAddresses(std::array{qubit1, qubit2});
  if (invalidStateExtraction_) {
    return;
  }
  std::swap(qubitPermutation[targets[0]], qubitPermutation[targets[1]]);
}

auto Runtime::qAlloc() -> Qubit* {
  if (qubitMode == ResourceMode::STATIC) {
    throw std::logic_error(
        "Cannot dynamically allocate qubits after using static qubit IDs");
  }
  qubitMode = ResourceMode::DYNAMIC;
  if ((freeQubits_.empty() &&
       currentMaxQubitId >= dd::Package::MAX_POSSIBLE_QUBITS) ||
      currentMaxQubitAddress == std::numeric_limits<uintptr_t>::max()) {
    throw std::out_of_range("QIR runtime exceeds the supported qubit range");
  }
  auto* qubit = reinterpret_cast<Qubit*>(currentMaxQubitAddress++);
  const auto id = freeQubits_.empty()
                      ? static_cast<dd::Qubit>(currentMaxQubitId++)
                      : freeQubits_.back();
  if (!freeQubits_.empty()) {
    freeQubits_.pop_back();
  }
  qRegister.emplace(qubit, id);
  if (extractState_) {
    enlargeState(id);
  }
  return qubit;
}

auto Runtime::qFree(Qubit* qubit) -> void {
  const auto it = qRegister.find(qubit);
  if (qubitMode != ResourceMode::DYNAMIC || it == qRegister.end()) {
    throw std::out_of_range("QIR qubit was not dynamically allocated");
  }
  const auto id = it->second;
  /// Extraction retains released wires as part of the exported state.
  if (!extractState_) {
    if (id < qState.numQubits) {
      reset(std::array{qubit});
    }
    freeQubits_.push_back(id);
  }
  qRegister.erase(it);
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
  if (staticResults_) {
    const auto id = reinterpret_cast<uintptr_t>(result);
    if (id >= *staticResults_) {
      throw std::out_of_range(
          "Static QIR result ID exceeds its declared capacity");
    }
    return resultValues_[id];
  }
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
  if (!extractState_) {
    measurements.push_back(result ? '1' : '0');
  }
}

auto Runtime::getMeasurements() const -> const std::string& {
  return measurements;
}

auto Runtime::takeState() -> QState {
  if (invalidStateExtraction_) {
    throw std::invalid_argument(
        "QIR state extraction cannot reset or operate on a measured qubit");
  }
  if (staticQubits_ && *staticQubits_ != 0) {
    enlargeState(*staticQubits_ - 1);
  }
  const auto matrix = mlir::qco::getStandardGateMatrix<mlir::qco::SWAPOp>({});
  for (size_t q = 0; q < qubitPermutation.size(); ++q) {
    /// Each transposition places at least one logical wire at its own index.
    while (qubitPermutation[q] != q) {
      const auto other = qubitPermutation[q];
      const std::array targets{other, qubitPermutation[other]};
      qState.edge = qState.dd->applyOperation(
          mlir::qco::makeGateDD(*qState.dd, matrix, qState.numQubits, targets),
          qState.edge);
      std::swap(qubitPermutation[q], qubitPermutation[other]);
    }
  }
  if (!qState.dd) {
    qState.dd = std::make_unique<dd::Package>(0);
  }
  QState ret = std::move(qState);
  reset();
  return ret;
}

auto Runtime::setOstream(std::ostream& other) -> void { os = &other; }

auto Runtime::resetOstream() -> void { os = &std::cout; }

auto Runtime::disableOutput() -> void { os = nullptr; }

void Runtime::outputType(const char* type, std::string_view value,
                         const char* label) const {
  if (!hasOutput()) {
    return;
  }
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
  if (!hasOutput()) {
    return;
  }
  outputType("INT", std::to_string(value), label);
}

auto Runtime::outputFloat(double value, const char* label) const -> void {
  if (!hasOutput()) {
    return;
  }
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
  if (!hasOutput()) {
    return;
  }
  outputType("TUPLE", std::to_string(elementCount), label);
}

auto Runtime::outputArray(int64_t elementCount, const char* label) const
    -> void {
  if (!hasOutput()) {
    return;
  }
  outputType("ARRAY", std::to_string(elementCount), label);
}

auto Runtime::outputProgramHeader() const -> void {
  if (!hasOutput()) {
    return;
  }
  *os << "HEADER\tschema_id\t" << outputSchema << "\n";
  *os << "HEADER\tschema_version\t2.1\n";
}

auto Runtime::outputShotStart() const -> void {
  if (!hasOutput()) {
    return;
  }
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
  if (!hasOutput()) {
    return;
  }
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
