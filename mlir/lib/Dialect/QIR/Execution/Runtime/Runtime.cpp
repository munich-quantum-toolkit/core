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
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qir {

namespace {
thread_local Runtime* ActiveRuntime = nullptr;
} // namespace

Runtime::QState::QState()
    : dd(llvm::cantFail(mlir::qco::ddResult(dd::Package::create(0)))),
      edge(dd::vEdge::one()) {}

void Runtime::QState::reset() {
  if (dd) {
    llvm::cantFail(mlir::qco::ddResult(dd->decRef(edge)));
    dd->garbageCollect();
  }
  edge = dd::vEdge::one();
  numQubits = 0;
}

Runtime::~Runtime() { llvm::consumeError(std::move(error_)); }

void Runtime::recordError(llvm::Error error) {
  if (error_) {
    llvm::consumeError(std::move(error));
  } else {
    error_ = std::move(error);
  }
}

bool Runtime::hasError() { return static_cast<bool>(error_); }
llvm::Error Runtime::takeError() { return std::move(error_); }

void Runtime::ownAllocation(void* pointer, void (*destroy)(void*)) {
  allocations_.emplace(
      pointer, std::unique_ptr<void, void (*)(void*)>(pointer, destroy));
}

void Runtime::releaseAllocation(void* pointer) { allocations_.erase(pointer); }

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
  llvm::consumeError(std::move(error_));
  allocations_.clear();
  currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
  currentMaxQubitId = 0;
  currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
  qState.reset();
  if (qState.dd && staticQubits_ && *staticQubits_ != 0) {
    llvm::cantFail(enlargeState(*staticQubits_ - 1));
  }
}

llvm::Error Runtime::configureStaticResources(std::optional<size_t> qubits,
                                              std::optional<size_t> results) {
  if (qubits && *qubits > dd::Package::MAX_POSSIBLE_QUBITS) {
    return llvm::createStringError(
        "Static QIR qubit capacity exceeds the supported range");
  }
  if (results && *results > resultValues_.max_size()) {
    return llvm::createStringError(
        "Static QIR result capacity exceeds the supported range");
  }
  staticQubits_ = qubits;
  staticResults_ = results;
  resultValues_.resize(results.value_or(0));
  reset();
  return llvm::Error::success();
}

auto Runtime::seed(const uint64_t randomSeed) -> void { mt.seed(randomSeed); }

Runtime::Runtime() : Runtime(generateRandomSeed()) {}

Runtime::Runtime(const uint64_t randomSeed)
    : qubitMode(ResourceMode::UNKNOWN), resultMode(ResourceMode::UNKNOWN),
      currentMaxQubitAddress(MIN_DYN_QUBIT_ADDRESS), currentMaxQubitId(0),
      currentMaxResultAddress(MIN_DYN_RESULT_ADDRESS), mt(randomSeed) {}

auto Runtime::enlargeState(size_t maxQubit) -> llvm::Error {
  if (maxQubit >= dd::Package::MAX_POSSIBLE_QUBITS ||
      (staticQubits_ && maxQubit >= *staticQubits_)) {
    return llvm::createStringError(
        std::make_error_code(std::errc::result_out_of_range),
        "QIR qubit ID exceeds the supported qubit range");
  }
  if (staticQubits_) {
    maxQubit = std::max(maxQubit, *staticQubits_ - 1);
  }
  if (maxQubit < qState.numQubits) {
    return llvm::Error::success();
  }
  const auto numQubits = maxQubit + 1;
  const auto capacity = qState.dd ? qState.dd->qubits() : 0;
  if (capacity < numQubits) {
    /// Unknown resources grow geometrically; declared resources fit exactly.
    const auto newCapacity = staticQubits_.value_or(std::min(
        dd::Package::MAX_POSSIBLE_QUBITS,
        std::max({numQubits, dd::Package::DEFAULT_QUBITS, 2 * capacity})));
    if (!qState.dd) {
      auto package = mlir::qco::ddResult(dd::Package::create(newCapacity));
      if (!package) {
        return package.takeError();
      }
      qState.dd = std::move(*package);
    } else {
      if (auto error = mlir::qco::ddResult(qState.dd->resize(newCapacity))) {
        return error;
      }
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
  llvm::cantFail(mlir::qco::ddResult(qState.dd->decRef(qState.edge)));
  qState.dd->incRef(edge);
  qState.edge = edge;
  qState.numQubits = numQubits;
  return llvm::Error::success();
}

auto Runtime::resolveAddress(const Qubit* qubit) -> llvm::Expected<dd::Qubit> {
  if (qubitMode == ResourceMode::UNKNOWN) {
    qubitMode = ResourceMode::STATIC;
  }
  if (qubitMode == ResourceMode::STATIC) {
    const auto id = reinterpret_cast<uintptr_t>(qubit);
    if (id >= dd::Package::MAX_POSSIBLE_QUBITS) {
      return llvm::createStringError(
          std::make_error_code(std::errc::result_out_of_range),
          "Static QIR qubit ID exceeds the supported qubit range");
    }
    if (staticQubits_ && id >= *staticQubits_) {
      return llvm::createStringError(
          std::make_error_code(std::errc::result_out_of_range),
          "Static QIR qubit ID exceeds its declared capacity");
    }
    return static_cast<dd::Qubit>(id);
  }

  const auto it = qRegister.find(qubit);
  if (it == qRegister.end()) {
    std::ostringstream ss;
    ss << __FILE__ << ":" << __LINE__
       << ": Qubit not allocated (not found): " << qubit;
    return llvm::createStringError(
        std::make_error_code(std::errc::result_out_of_range), ss.str());
  }
  return it->second;
}

auto Runtime::translateAddresses(const std::span<Qubit* const> qubits,
                                 const std::span<Qubit* const> additionalQubits)
    -> llvm::Expected<llvm::SmallVector<dd::Qubit, 5>> {
  llvm::SmallVector<dd::Qubit, 5> qubitIds;
  qubitIds.reserve(qubits.size() + additionalQubits.size());
  for (const auto* qubit : qubits) {
    auto id = resolveAddress(qubit);
    if (!id) {
      return id.takeError();
    }
    qubitIds.push_back(*id);
  }
  for (const auto* qubit : additionalQubits) {
    auto id = resolveAddress(qubit);
    if (!id) {
      return id.takeError();
    }
    qubitIds.push_back(*id);
  }
  if (extractState_ && std::ranges::any_of(qubitIds, [&](const auto id) {
        return measuredQubits_.contains(id);
      })) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "QIR state extraction cannot reset or operate on a measured qubit");
  }
  if (!qubitIds.empty()) {
    if (auto error = enlargeState(*std::ranges::max_element(qubitIds))) {
      return std::move(error);
    }
  }
  return qubitIds;
}

auto Runtime::apply(const std::span<const std::complex<dd::fp>> matrix,
                    std::span<Qubit* const> controls,
                    std::span<Qubit* const> targets) -> llvm::Error {
  auto addresses = translateAddresses(controls, targets);
  if (!addresses) {
    return addresses.takeError();
  }
  if (!qState.dd) {
    qState.dd = llvm::cantFail(mlir::qco::ddResult(dd::Package::create(0)));
  }
  std::ranges::transform(
      *addresses, addresses->begin(),
      [&](const auto address) { return qubitPermutation[address]; });
  const llvm::ArrayRef mappedAddresses(*addresses);
  const auto mappedTargets = mappedAddresses.drop_front(controls.size());
  const dd::Controls mappedControls(mappedAddresses.begin(),
                                    mappedTargets.begin());
  auto gate = mlir::qco::makeGateDD(*qState.dd, matrix, qState.numQubits,
                                    mappedTargets, mappedControls);
  if (!gate) {
    return gate.takeError();
  }
  auto edge =
      mlir::qco::ddResult(qState.dd->applyOperation(*gate, qState.edge));
  if (!edge) {
    return edge.takeError();
  }
  qState.edge = *edge;
  return llvm::Error::success();
}

auto Runtime::applyGlobalPhase(dd::fp phase) -> llvm::Error {
  if (!qState.dd) {
    qState.dd = llvm::cantFail(mlir::qco::ddResult(dd::Package::create(0)));
  }
  auto edge =
      mlir::qco::ddResult(dd::applyGlobalPhase(qState.edge, phase, *qState.dd));
  if (!edge) {
    return edge.takeError();
  }
  qState.edge = *edge;
  return llvm::Error::success();
}

auto Runtime::measure(Qubit* qubit, Result* result) -> llvm::Error {
  auto target = resolveAddress(qubit);
  if (!target) {
    return target.takeError();
  }
  if (auto error = enlargeState(*target)) {
    return error;
  }
  auto value = deref(result);
  if (!value) {
    return value.takeError();
  }
  if (extractState_) {
    measuredQubits_.insert(*target);
  } else if (!deferMeasurements_) {
    auto bit = mlir::qco::ddResult(qState.dd->measureOneCollapsing(
        qState.edge, qubitPermutation[*target], mt));
    if (!bit) {
      return bit.takeError();
    }
    value->r = *bit == '1';
  }
  return llvm::Error::success();
}

auto Runtime::sampleMeasurements(std::span<const uintptr_t> qubits,
                                 size_t shots,
                                 std::vector<std::string>& results)
    -> llvm::Error {
  measurements.clear();
  if (qubits.empty()) {
    results.resize(shots);
    return llvm::Error::success();
  }
  bool ascending = qubits.size() == qState.numQubits;
  for (size_t i = 0; ascending && i < qubits.size(); ++i) {
    ascending = qubitPermutation[qubits[i]] == i;
  }
  measurements.reserve(qubits.size());
  for (size_t i = 0; i < shots; ++i) {
    auto basis =
        mlir::qco::ddResult(qState.dd->measureAll(qState.edge, false, mt));
    if (!basis) {
      return basis.takeError();
    }
    if (ascending) {
      std::ranges::reverse(*basis);
      results.push_back(std::move(*basis));
    } else {
      measurements.clear();
      for (const auto qubit : qubits) {
        measurements.push_back(
            (*basis)[basis->size() - 1 - qubitPermutation[qubit]]);
      }
      results.push_back(measurements);
    }
  }
  if (shots != 0) {
    measurements = results.back();
  }
  return llvm::Error::success();
}

auto Runtime::reset(std::span<Qubit* const> qubits) -> llvm::Error {
  if (extractState_) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "QIR state extraction cannot reset or operate on a measured qubit");
  }
  auto targets = translateAddresses(qubits);
  if (!targets) {
    return targets.takeError();
  }
  const auto matrix =
      llvm::cantFail(mlir::qco::getStandardGateMatrix<mlir::qco::XOp>({}));
  for (const auto target : *targets) {
    const auto mapped = qubitPermutation[target];
    auto bit = mlir::qco::ddResult(
        qState.dd->measureOneCollapsing(qState.edge, mapped, mt));
    if (!bit) {
      return bit.takeError();
    }
    if (*bit == '1') {
      const std::array targetArray{mapped};
      auto gate = mlir::qco::makeGateDD(*qState.dd, matrix, qState.numQubits,
                                        targetArray);
      if (!gate) {
        return gate.takeError();
      }
      auto edge =
          mlir::qco::ddResult(qState.dd->applyOperation(*gate, qState.edge));
      if (!edge) {
        return edge.takeError();
      }
      qState.edge = *edge;
    }
  }
  return llvm::Error::success();
}

auto Runtime::swap(Qubit* qubit1, Qubit* qubit2) -> llvm::Error {
  auto targets = translateAddresses(std::array{qubit1, qubit2});
  if (!targets) {
    return targets.takeError();
  }
  std::swap(qubitPermutation[(*targets)[0]], qubitPermutation[(*targets)[1]]);
  return llvm::Error::success();
}

auto Runtime::qAlloc() -> llvm::Expected<Qubit*> {
  if (qubitMode == ResourceMode::STATIC) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Cannot dynamically allocate qubits after using static qubit IDs");
  }
  qubitMode = ResourceMode::DYNAMIC;
  if ((freeQubits_.empty() &&
       currentMaxQubitId >= dd::Package::MAX_POSSIBLE_QUBITS) ||
      currentMaxQubitAddress == std::numeric_limits<uintptr_t>::max()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::result_out_of_range),
        "QIR runtime exceeds the supported qubit range");
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
    if (auto error = enlargeState(id)) {
      return std::move(error);
    }
  }
  return qubit;
}

auto Runtime::qFree(Qubit* qubit) -> llvm::Error {
  const auto it = qRegister.find(qubit);
  if (qubitMode != ResourceMode::DYNAMIC || it == qRegister.end()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::result_out_of_range),
        "QIR qubit was not dynamically allocated");
  }
  const auto id = it->second;
  /// Extraction retains released wires as part of the exported state.
  if (!extractState_) {
    if (id < qState.numQubits) {
      if (auto error = reset(std::array{qubit})) {
        return error;
      }
    }
    freeQubits_.push_back(id);
  }
  qRegister.erase(it);
  return llvm::Error::success();
}

auto Runtime::rAlloc() -> llvm::Expected<Result*> {
  if (resultMode == ResourceMode::STATIC) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Cannot dynamically allocate results after using static result IDs");
  }
  if (currentMaxResultAddress == std::numeric_limits<uintptr_t>::max()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::result_out_of_range),
        "QIR runtime exceeds the supported result range");
  }
  resultMode = ResourceMode::DYNAMIC;
  auto* result = reinterpret_cast<Result*>(currentMaxResultAddress++);
  rRegister.emplace(result, ResultStruct{.r = false});
  return result;
}

auto Runtime::rFree(Result* result) -> llvm::Error {
  if (resultMode != ResourceMode::DYNAMIC || rRegister.erase(result) == 0) {
    return llvm::createStringError(
        std::make_error_code(std::errc::result_out_of_range),
        "QIR result was not dynamically allocated");
  }
  return llvm::Error::success();
}

auto Runtime::deref(Result* result) -> llvm::Expected<ResultStruct&> {
  if (staticResults_) {
    const auto id = reinterpret_cast<uintptr_t>(result);
    if (id >= *staticResults_) {
      return llvm::createStringError(
          std::make_error_code(std::errc::result_out_of_range),
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
      return llvm::createStringError(
          std::make_error_code(std::errc::result_out_of_range), ss.str());
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

auto Runtime::takeState() -> llvm::Expected<QState> {
  if (hasError()) {
    return takeError();
  }
  if (staticQubits_ && *staticQubits_ != 0) {
    if (auto error = enlargeState(*staticQubits_ - 1)) {
      return std::move(error);
    }
  }
  const auto matrix =
      llvm::cantFail(mlir::qco::getStandardGateMatrix<mlir::qco::SWAPOp>({}));
  for (size_t q = 0; q < qubitPermutation.size(); ++q) {
    /// Each transposition places at least one logical wire at its own index.
    while (qubitPermutation[q] != q) {
      const auto other = qubitPermutation[q];
      const std::array targets{other, qubitPermutation[other]};
      auto gate =
          mlir::qco::makeGateDD(*qState.dd, matrix, qState.numQubits, targets);
      if (!gate) {
        return gate.takeError();
      }
      auto edge =
          mlir::qco::ddResult(qState.dd->applyOperation(*gate, qState.edge));
      if (!edge) {
        return edge.takeError();
      }
      qState.edge = *edge;
      std::swap(qubitPermutation[q], qubitPermutation[other]);
    }
  }
  if (!qState.dd) {
    qState.dd = llvm::cantFail(mlir::qco::ddResult(dd::Package::create(0)));
  }
  QState ret = std::move(qState);
  reset();
  return ret;
}

auto Runtime::setOstream(std::ostream& other) -> void { os = &other; }

auto Runtime::resetOstream() -> void { os = &std::cout; }

auto Runtime::disableOutput() -> void { os = nullptr; }

llvm::Error Runtime::checkOutput() const {
  if (os->exceptions() != std::ios::goodbit) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "QIR output streams must have exceptions disabled");
  }
  if (!*os) {
    return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                   "Failed to write QIR output");
  }
  return llvm::Error::success();
}

llvm::Error Runtime::outputType(const char* type, std::string_view value,
                                const char* label) const {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  if (auto error = checkOutput()) {
    return error;
  }
  *os << "OUTPUT\t" << type << "\t" << value;
  if (label != nullptr && outputSchema == OutputSchema::Labeled) {
    *os << "\t" << label;
  }
  *os << "\n";
  return checkOutput();
}

auto Runtime::outputResult(bool value, const char* label) const -> llvm::Error {
  return outputType("RESULT", value ? "1" : "0", label);
}

auto Runtime::outputResultArray(const std::string_view values,
                                const char* label) const -> llvm::Error {
  return outputType("RESULT_ARRAY", values, label);
}

auto Runtime::outputBool(bool value, const char* label) const -> llvm::Error {
  return outputType("BOOL", value ? "true" : "false", label);
}

auto Runtime::outputInt(int64_t value, const char* label) const -> llvm::Error {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  return outputType("INT", std::to_string(value), label);
}

auto Runtime::outputFloat(double value, const char* label) const
    -> llvm::Error {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  // Use std::ostringstream rather than std::to_string.
  // std::to_string formats with six digits after the decimal point and
  // can print 0.000000 for very small numbers.
  // std::ostringstream uses six significant digits by default and
  // outputs very small numbers with scientific notation.
  std::ostringstream oss;
  oss << value;
  return outputType("DOUBLE", oss.str(), label);
}

auto Runtime::outputTuple(int64_t elementCount, const char* label) const
    -> llvm::Error {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  return outputType("TUPLE", std::to_string(elementCount), label);
}

auto Runtime::outputArray(int64_t elementCount, const char* label) const
    -> llvm::Error {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  return outputType("ARRAY", std::to_string(elementCount), label);
}

auto Runtime::outputProgramHeader() const -> llvm::Error {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  if (auto error = checkOutput()) {
    return error;
  }
  *os << "HEADER\tschema_id\t" << outputSchema << "\n";
  *os << "HEADER\tschema_version\t2.1\n";
  return checkOutput();
}

auto Runtime::outputShotStart() const -> llvm::Error {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  if (auto error = checkOutput()) {
    return error;
  }
  *os << "START\n";
  if (metadata.empty()) {
    *os << "METADATA\toutput_labeling_schema\t" << outputSchema << "\n";
    return checkOutput();
  }
  for (const auto& [name, value] : metadata) {
    *os << "METADATA\t" << name;
    if (!value.empty()) {
      *os << "\t" << value;
    }
    *os << "\n";
  }
  return checkOutput();
}

auto Runtime::outputShotEnd(const int64_t exitCode) const -> llvm::Error {
  if (!hasOutput()) {
    return llvm::Error::success();
  }
  if (auto error = checkOutput()) {
    return error;
  }
  *os << "END\t" << exitCode << "\n";
  return checkOutput();
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
