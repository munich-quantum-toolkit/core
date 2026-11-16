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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

using llvm::createStringError;
using llvm::Error;
using llvm::Expected;

namespace qir {

namespace {
thread_local Runtime* ActiveRuntime = nullptr;
} // namespace

auto Runtime::generateRandomSeed() -> uint64_t {
  std::random_device rd;
  return std::uniform_int_distribution<uint64_t>{}(rd);
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
  if (deferredOutputs_ != nullptr) {
    deferredOutputs_->clear();
  }
  if (samplingBranches_ != nullptr) {
    samplingBranches_->position = 0;
  }
  measuredQubits_.clear();
  invalidStateExtraction_ = false;
  currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
  currentMaxQubitId = 0;
  currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
  qState.reset();
  if (qState.dd && staticQubits_ && *staticQubits_ != 0) {
    llvm::cantFail(enlargeState(*staticQubits_ - 1));
  }
}

Error Runtime::configureStaticResources(std::optional<size_t> qubits,
                                        std::optional<size_t> results) {
  if (qubits && *qubits > maxQubits_) {
    return createStringError(
        std::errc::result_out_of_range,
        "Static QIR qubit capacity exceeds the supported range");
  }
  if (results && *results > resultValues_.max_size()) {
    return createStringError(
        std::errc::result_out_of_range,
        "Static QIR result capacity exceeds the supported range");
  }
  staticQubits_ = qubits;
  staticResults_ = results;
  resultValues_.resize(results.value_or(0));
  reset();
  return Error::success();
}

auto Runtime::seed(const uint64_t randomSeed) -> void { mt.seed(randomSeed); }

Runtime::Runtime() : Runtime(generateRandomSeed()) {}

Runtime::Runtime(const uint64_t randomSeed)
    : qubitMode(ResourceMode::UNKNOWN), resultMode(ResourceMode::UNKNOWN),
      currentMaxQubitAddress(MIN_DYN_QUBIT_ADDRESS), currentMaxQubitId(0),
      currentMaxResultAddress(MIN_DYN_RESULT_ADDRESS), mt(randomSeed) {}

auto Runtime::enlargeState(size_t maxQubit) -> Error {
  if (maxQubit >= maxQubits_ || (staticQubits_ && maxQubit >= *staticQubits_)) {
    return createStringError(std::errc::result_out_of_range,
                             "QIR qubit ID exceeds the supported qubit range");
  }
  if (staticQubits_) {
    maxQubit = std::max(maxQubit, *staticQubits_ - 1);
  }
  if (maxQubit < qState.numQubits) {
    return Error::success();
  }
  const auto numQubits = maxQubit + 1;
  const auto capacity = qState.dd ? qState.dd->qubits() : 0;
  if (capacity < numQubits) {
    // Unknown resources grow geometrically; declared resources fit exactly.
    const auto newCapacity = staticQubits_.value_or(std::min(
        maxQubits_,
        std::max({numQubits, dd::Package::DEFAULT_QUBITS, 2 * capacity})));
    if (!qState.dd) {
      auto package = dd::Package::create(newCapacity);
      if (!package) {
        return package.takeError();
      }
      qState.dd = std::move(*package);
    } else {
      if (auto error = qState.dd->resize(newCapacity)) {
        return error;
      }
    }
  }
  qubitPermutation.resize(numQubits);
  std::iota(qubitPermutation.begin() + static_cast<ptrdiff_t>(qState.numQubits),
            qubitPermutation.end(), static_cast<dd::Qubit>(qState.numQubits));

  // Extending the root preserves its weight, including a scalar global phase.
  auto edge = qState.edge;
  for (auto q = qState.numQubits; q < numQubits; ++q) {
    edge = qState.dd->makeDDNode(static_cast<dd::Qubit>(q),
                                 std::array{edge, dd::vEdge::zero()});
  }
  qState.dd->incRef(edge);
  qState.dd->decRef(qState.edge);
  qState.edge = edge;
  qState.numQubits = numQubits;
  return Error::success();
}

auto Runtime::resolveAddress(const Qubit* qubit) -> Expected<dd::Qubit> {
  if (qubitMode == ResourceMode::UNKNOWN) {
    qubitMode = ResourceMode::STATIC;
  }
  if (qubitMode == ResourceMode::STATIC) {
    auto id = staticQubitId(qubit);
    if (!id) {
      return id.takeError();
    }
    if (staticQubits_ && *id >= *staticQubits_) {
      return createStringError(
          std::errc::result_out_of_range,
          "Static QIR qubit ID exceeds its declared capacity");
    }
    return *id;
  }

  const auto it = qRegister.find(qubit);
  if (it == qRegister.end()) {
    return createStringError(std::errc::invalid_argument,
                             "Qubit not allocated (not found)");
  }
  return it->second;
}

auto Runtime::translateAddresses(const std::span<Qubit* const> qubits,
                                 const std::span<Qubit* const> additionalQubits)
    -> Expected<llvm::SmallVector<dd::Qubit>> {
  llvm::SmallVector<dd::Qubit> qubitIds;
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
    // Report after JIT execution returns without unwinding generated frames.
    invalidStateExtraction_ = true;
  }
  if (!qubitIds.empty()) {
    if (auto error = enlargeState(*std::ranges::max_element(qubitIds))) {
      return error;
    }
  }
  return qubitIds;
}

auto Runtime::apply(const std::span<const std::complex<dd::fp>> matrix,
                    std::span<Qubit* const> controls,
                    std::span<Qubit* const> targets) -> Error {
  auto addresses = translateAddresses(controls, targets);
  if (!addresses) {
    return addresses.takeError();
  }
  if (invalidStateExtraction_) {
    return Error::success();
  }
  if (!qState.dd) {
    qState.dd = llvm::cantFail(dd::Package::create(0));
  }
  std::ranges::transform(
      *addresses, addresses->begin(),
      [&](const auto address) { return qubitPermutation[address]; });

  const llvm::ArrayRef mappedAddresses(*addresses);
  const auto mappedTargets = mappedAddresses.drop_front(controls.size());
  const dd::Controls mappedControls(mappedAddresses.begin(),
                                    mappedTargets.begin());
  auto gate =
      mlir::qco::makeGateDD(*qState.dd, matrix, mappedTargets, mappedControls);
  if (!gate) {
    return gate.takeError();
  }
  qState.edge = qState.dd->applyOperation(*gate, qState.edge);
  return Error::success();
}

auto Runtime::applyGlobalPhase(dd::fp phase) -> void {
  if (!qState.dd) {
    qState.dd = llvm::cantFail(dd::Package::create(0));
  }
  qState.edge = dd::applyGlobalPhase(qState.edge, phase, *qState.dd);
}

auto Runtime::measure(Qubit* qubit, Result* result) -> Error {
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
    (*value)->deferredWire = qubitPermutation[*target];
  } else if (!deferMeasurements_) {
    auto measured = measureQubit(qubitPermutation[*target]);
    if (!measured) {
      return measured.takeError();
    }
    (*value)->r = *measured;
  }
  return Error::success();
}

Expected<bool> Runtime::measureQubit(dd::Qubit target) {
  if (samplingBranches_ == nullptr ||
      (samplingBranches_->count == 1 &&
       samplingBranches_->position == samplingBranches_->path.size())) {
    auto measured = qState.dd->measureOneCollapsing(qState.edge, target, mt);
    if (!measured) {
      return measured.takeError();
    }
    return *measured == '1';
  }
  auto& branches = *samplingBranches_;
  const auto [pzero, pone] =
      dd::Package::determineMeasurementProbabilities(qState.edge, target);
  const auto sum = pzero + pone;
  if (std::abs(sum - 1) > 0.001) {
    return createStringError(std::errc::invalid_argument,
                             "Numerical instability during measurement");
  }
  bool one = false;
  if (branches.position < branches.path.size()) {
    one = branches.path[branches.position];
  } else {
    const auto zeros =
        std::binomial_distribution<size_t>(branches.count, pzero / sum)(mt);
    const auto ones = branches.count - zeros;
    one = zeros == 0;
    if (zeros != 0 && ones != 0) {
      auto sibling = branches.path;
      sibling.push_back(true);
      branches.pending.emplace_back(std::move(sibling), ones);
      branches.count = zeros;
    }
    branches.path.push_back(one);
  }
  ++branches.position;
  qState.dd->performCollapsingMeasurement(qState.edge, target,
                                          one ? pone : pzero, !one);
  return one;
}

auto Runtime::sampleMeasurements(
    std::span<const uintptr_t> qubits, size_t shots,
    llvm::function_ref<void(std::string&&)> emitOwned,
    llvm::function_ref<void(std::string_view)> emitBorrowed) -> Error {
  measurements.clear();
  if (qubits.empty()) {
    for (size_t i = 0; i < shots; ++i) {
      emitBorrowed({});
    }
    return Error::success();
  }
  bool ascending = qubits.size() == qState.numQubits;
  for (size_t i = 0; ascending && i < qubits.size(); ++i) {
    ascending = qubitPermutation[qubits[i]] == i;
  }
  if (ascending) {
    for (size_t i = 0; i < shots; ++i) {
      auto basis = qState.dd->measureAll(qState.edge, false, mt);
      if (!basis) {
        return basis.takeError();
      }
      std::ranges::reverse(*basis);
      if (i + 1 == shots) {
        measurements = *basis;
      }
      emitOwned(std::move(*basis));
    }
    return Error::success();
  }
  measurements.reserve(qubits.size());
  for (size_t i = 0; i < shots; ++i) {
    auto basis = qState.dd->measureAll(qState.edge, false, mt);
    if (!basis) {
      return basis.takeError();
    }
    measurements.clear();
    for (const auto qubit : qubits) {
      measurements.push_back(
          (*basis)[basis->size() - 1 - qubitPermutation[qubit]]);
    }
    emitBorrowed(measurements);
  }
  return Error::success();
}

auto Runtime::reset(std::span<Qubit* const> qubits) -> Error {
  if (extractState_) {
    invalidStateExtraction_ = true;
    return Error::success();
  }
  auto targets = translateAddresses(qubits);
  if (!targets) {
    return targets.takeError();
  }
  std::ranges::transform(*targets, targets->begin(), [&](const auto target) {
    return qubitPermutation[target];
  });
  const auto matrix = mlir::qco::XOp::getUnitaryMatrix();
  for (const auto target : *targets) {
    if (samplingBranches_ != nullptr && samplingBranches_->count > 1 &&
        samplingBranches_->position == samplingBranches_->path.size()) {
      const auto [pzero, pone] =
          dd::Package::determineMeasurementProbabilities(qState.edge, target);
      if (pzero > 0 && pone > 0) {
        auto zero = qState.edge;
        auto one = qState.edge;
        qState.dd->incRef(zero);
        qState.dd->incRef(one);
        qState.dd->performCollapsingMeasurement(zero, target, pzero, true);
        qState.dd->performCollapsingMeasurement(one, target, pone, false);
        one = qState.dd->applyOperation(
            llvm::cantFail(
                mlir::qco::makeGateDD(*qState.dd, matrix, std::array{target})),
            one);
        // Reset outcomes with equal normalized states can keep one population.
        const bool equal = zero.p == one.p;
        qState.dd->decRef(one);
        if (equal) {
          qState.dd->decRef(qState.edge);
          qState.edge = zero;
          samplingBranches_->path.push_back(false);
          ++samplingBranches_->position;
          continue;
        }
        qState.dd->decRef(zero);
      }
    }
    auto measured = measureQubit(target);
    if (!measured) {
      return measured.takeError();
    }
    if (*measured) {
      const std::array targetArray{target};
      qState.edge =
          qState.dd->applyOperation(llvm::cantFail(mlir::qco::makeGateDD(
                                        *qState.dd, matrix, targetArray)),
                                    qState.edge);
    }
  }
  return Error::success();
}

auto Runtime::swap(Qubit* qubit1, Qubit* qubit2) -> Error {
  auto targets = translateAddresses(std::array{qubit1, qubit2});
  if (!targets) {
    return targets.takeError();
  }
  if (invalidStateExtraction_) {
    return Error::success();
  }
  std::swap(qubitPermutation[(*targets)[0]], qubitPermutation[(*targets)[1]]);
  return Error::success();
}

auto Runtime::qAlloc() -> Expected<Qubit*> {
  if (qubitMode == ResourceMode::STATIC) {
    return createStringError(
        std::errc::invalid_argument,
        "Cannot dynamically allocate qubits after using static qubit IDs");
  }
  qubitMode = ResourceMode::DYNAMIC;
  if ((freeQubits_.empty() && currentMaxQubitId >= maxQubits_) ||
      currentMaxQubitAddress == std::numeric_limits<uintptr_t>::max()) {
    return createStringError(std::errc::result_out_of_range,
                             "QIR runtime exceeds the supported qubit range");
  }
  auto* qubit = reinterpret_cast<Qubit*>(currentMaxQubitAddress++);
  const auto id = freeQubits_.empty()
                      ? static_cast<dd::Qubit>(currentMaxQubitId++)
                      : freeQubits_.back();
  if (!freeQubits_.empty()) {
    if (deferredOutputs_ != nullptr) {
      invalidStateExtraction_ = true;
    }
    freeQubits_.pop_back();
  }
  qRegister.emplace(qubit, id);
  if (extractState_) {
    if (auto error = enlargeState(id)) {
      return error;
    }
  }
  return qubit;
}

auto Runtime::qFree(Qubit* qubit) -> Error {
  const auto it = qRegister.find(qubit);
  if (qubitMode != ResourceMode::DYNAMIC || it == qRegister.end()) {
    return createStringError(std::errc::result_out_of_range,
                             "QIR qubit was not dynamically allocated");
  }
  const auto id = it->second;
  // Extraction retains released wires as part of the exported state.
  if (!extractState_) {
    if (id < qState.numQubits) {
      if (auto error = reset(std::array{qubit})) {
        return error;
      }
    }
    freeQubits_.push_back(id);
  } else if (deferredOutputs_ != nullptr) {
    // A sampling trial must stay within the original allocation bound.
    // Reusing a released wire invalidates deferral and triggers ordinary
    // sampling.
    freeQubits_.push_back(id);
  }
  qRegister.erase(it);
  return Error::success();
}

auto Runtime::rAlloc() -> Expected<Result*> {
  if (resultMode == ResourceMode::STATIC) {
    return createStringError(
        std::errc::invalid_argument,
        "Cannot dynamically allocate results after using static result IDs");
  }
  if (currentMaxResultAddress == std::numeric_limits<uintptr_t>::max()) {
    return createStringError(std::errc::result_out_of_range,
                             "QIR runtime exceeds the supported result range");
  }
  resultMode = ResourceMode::DYNAMIC;
  auto* result = reinterpret_cast<Result*>(currentMaxResultAddress++);
  rRegister.emplace(result, ResultStruct{.r = false});
  return result;
}

auto Runtime::rFree(Result* result) -> Error {
  if (resultMode != ResourceMode::DYNAMIC || rRegister.erase(result) == 0) {
    return createStringError(std::errc::result_out_of_range,
                             "QIR result was not dynamically allocated");
  }
  return Error::success();
}

auto Runtime::deref(Result* result) -> Expected<ResultStruct*> {
  if (staticResults_) {
    const auto id = reinterpret_cast<uintptr_t>(result);
    if (id >= *staticResults_) {
      return createStringError(
          std::errc::result_out_of_range,
          "Static QIR result ID exceeds its declared capacity");
    }
    return &resultValues_[id];
  }
  auto it = rRegister.find(result);
  if (it == rRegister.end()) {
    if (resultMode == ResourceMode::DYNAMIC) {
      return createStringError(std::errc::invalid_argument,
                               "Result not allocated (not found)");
    }
    resultMode = ResourceMode::STATIC;
    it = rRegister.emplace(result, ResultStruct{.r = false}).first;
  }
  return &it->second;
}

auto Runtime::appendMeasurementBit(const bool result,
                                   const std::optional<dd::Qubit> deferredWire)
    -> void {
  if (deferredOutputs_ != nullptr) {
    if (deferredWire) {
      deferredOutputs_->emplace_back(*deferredWire);
    } else {
      deferredOutputs_->emplace_back(result);
    }
  }
  if (!extractState_) {
    measurements.push_back(result ? '1' : '0');
  }
}

auto Runtime::getMeasurements() const -> const std::string& {
  return measurements;
}

auto Runtime::takeState() -> Expected<QState> {
  if (invalidStateExtraction_) {
    return createStringError(
        std::errc::invalid_argument,
        "QIR state extraction cannot reset or operate on a measured qubit");
  }
  if (staticQubits_ && *staticQubits_ != 0) {
    llvm::cantFail(enlargeState(*staticQubits_ - 1));
  }
  const auto matrix = mlir::qco::SWAPOp::getUnitaryMatrix();
  for (size_t q = 0; q < qubitPermutation.size(); ++q) {
    // Each transposition places at least one logical wire at its own index.
    while (qubitPermutation[q] != q) {
      const auto other = qubitPermutation[q];
      const std::array targets{other, qubitPermutation[other]};
      qState.edge = qState.dd->applyOperation(
          llvm::cantFail(mlir::qco::makeGateDD(*qState.dd, matrix, targets)),
          qState.edge);
      std::swap(qubitPermutation[q], qubitPermutation[other]);
    }
  }
  if (!qState.dd) {
    qState.dd = llvm::cantFail(dd::Package::create(0));
  }
  QState ret = std::move(qState);
  reset();
  return ret;
}

auto Runtime::setOstream(llvm::raw_ostream& other) -> void { os = &other; }

auto Runtime::resetOstream() -> void { os = &llvm::outs(); }

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
  if (outputResultPositions_ != nullptr) {
    outputResultPositions_->push_back(
        os->tell() + std::string_view("OUTPUT\tRESULT\t").size());
  }
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
  // A shortest double fits in 24 characters, including sign and exponent.
  std::array<char, 32> text{};
  const auto result =
      std::to_chars(text.data(), text.data() + text.size(), value);
  assert(result.ec == std::errc{});
  outputType("DOUBLE", std::string_view(text.data(), result.ptr), label);
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

auto operator<<(llvm::raw_ostream& os, const Runtime::OutputSchema schema)
    -> llvm::raw_ostream& {
  return os << (schema == Runtime::OutputSchema::Labeled ? LABELED_SCHEMA
                                                         : ORDERED_SCHEMA);
}

} // namespace qir
