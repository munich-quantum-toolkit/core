/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QDMIDevice.h"

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/ResultLayout.h"
#include "qdmi/common/Common.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mqt::catalyst {
namespace {

struct Registry {
  std::mutex mutex;
  uint64_t next = 1;
  llvm::DenseMap<uint64_t, std::weak_ptr<Context>> contexts;
};

} // namespace

static void require(bool valid, const llvm::Twine& message) {
  if (!valid) {
    throw std::runtime_error(message.str());
  }
}

template <class T> static T unwrap(llvm::Expected<T> value) {
  if (!value) {
    throw std::runtime_error(llvm::toString(value.takeError()));
  }
  return std::move(*value);
}

static void validateShots(size_t shots) {
  require(shots > 0 && shots <= static_cast<size_t>(INT64_MAX),
          "QDMI requires a positive shot count that fits int64_t.");
}

static size_t product(size_t lhs, size_t rhs) {
  require(rhs == 0 || lhs <= SIZE_MAX / rhs,
          "Result dimensions overflow size_t.");
  return lhs * rhs;
}

static size_t dimension(size_t width) {
  require(width < std::numeric_limits<size_t>::digits,
          "Dense result dimension does not fit size_t.");
  return size_t{1} << width;
}

static Registry& registry() {
  // Catalyst may destroy pooled devices after other static objects.
  static auto* instance = new Registry();
  return *instance;
}

using Parameters = std::array<std::optional<qdmi::CustomJobParameter>, 5>;

static Parameters borrowParameters(const JobParameters& parameters) {
  Parameters result;
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i]) {
      result[i] = std::visit(
          [](const auto& value) -> qdmi::CustomJobParameter {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                         std::vector<std::byte>>) {
              return std::span<const std::byte>(value);
            } else {
              return value;
            }
          },
          *parameters[i]);
    }
  }
  return result;
}

static uint64_t parseContext(const char* kwargs) {
  require(kwargs != nullptr, "Missing QDMI context arguments.");
  auto text = llvm::StringRef(kwargs).ltrim();
  require(text.consume_front("{"), "Invalid QDMI context arguments.");
  text = text.ltrim();
  require(text.consume_front("'context'") || text.consume_front("\"context\""),
          "Expected the QDMI context key.");
  text = text.ltrim();
  require(text.consume_front(":"), "Invalid QDMI context arguments.");
  text = text.ltrim();
  uint64_t id = 0;
  require(!text.consumeInteger(10, id) && id != 0, "Invalid QDMI context ID.");
  text = text.ltrim();
  require(text.consume_front("}") && text.trim().empty(),
          "Unexpected QDMI context arguments.");
  return id;
}

Context::Context(qdmi::Device device, JobParameters parameters,
                 std::optional<QDMI_Program_Format> programFormat)
    : device_(std::move(device)), parameters_(std::move(parameters)),
      environment_(
          unwrap(mlir::targetEnvironmentFromDevice(device_, programFormat))),
      qubits_(unwrap(device_.getQubitsNum())) {}

std::shared_ptr<Context>
Context::create(qdmi::Device device, JobParameters parameters,
                std::optional<QDMI_Program_Format> programFormat) {
  auto context = std::shared_ptr<Context>(
      new Context(std::move(device), std::move(parameters), programFormat));
  auto& entries = registry();
  const std::scoped_lock lock(entries.mutex);
  require(entries.next < UINT64_MAX - 1, "QDMI context ID space exhausted.");
  context->id_ = entries.next++;
  entries.contexts.try_emplace(context->id(), context);
  return context;
}

Context::~Context() {
  auto& entries = registry();
  const std::scoped_lock lock(entries.mutex);
  entries.contexts.erase(id());
}

uint64_t Context::id() const noexcept { return id_; }
size_t Context::qubits() const noexcept { return qubits_; }

std::optional<qdmi::Job>
Context::trySubmit(llvm::ArrayRef<const mlir::CompiledProgram*> programs,
                   int64_t shots) const {
  const auto parameters = borrowParameters(parameters_);
  return unwrap(mlir::trySubmitPrograms(
      device_, programs, shots, parameters[0], parameters[1], parameters[2],
      parameters[3], parameters[4], &environment_));
}

qdmi::Job Context::submit(const mlir::CompiledProgram& program,
                          int64_t shots) const {
  auto job = trySubmit({&program}, shots);
  require(job.has_value(),
          "Device declined the compiled program before submission.");
  return std::move(*job);
}

llvm::SmallVector<size_t>
QDMIDevice::wireIndices(llvm::ArrayRef<QubitIdType> wires) const {
  return recorder_.wireIndices(wires);
}

const QDMIDevice::ObservableWires&
QDMIDevice::getObservable(ObsIdType id) const {
  require(id >= firstObservable_ &&
              static_cast<uint64_t>(id - firstObservable_) <
                  observables_.size(),
          "Invalid or released QDMI observable ID.");
  return observables_[static_cast<size_t>(id - firstObservable_)];
}

ObsIdType QDMIDevice::addObservable(ObservableWires value) {
  require(nextObservable_ != INTPTR_MAX, "QDMI observable ID space exhausted.");
  observables_.push_back(std::move(value));
  return nextObservable_++;
}

const QDMIDevice::Readout& QDMIDevice::read(bool samples) {
  validateShots(shots_);
  require(context_ != nullptr, "No QDMI execution context is configured.");
  if (!result_ && !recorder_.hasTouchedWires()) {
    result_ = Readout{{std::string(recorder_.wires().size(), '0'), shots_}};
    sampleReadout_ = true;
  }
  if (!result_ || (samples && !sampleReadout_)) {
    if (!compiled_) {
      compiled_ = unwrap(mlir::CompiledProgram::compile(
          recorder_.takeProgram(), context_->environment_));
    }
    if (!job_) {
      auto job = context_->submit(*compiled_, static_cast<int64_t>(shots_));
      require(unwrap(job.wait()) && unwrap(job.check()) == QDMI_JOB_STATUS_DONE,
              "QDMI job did not complete successfully.");
      job_ = std::move(job);
    }
    result_ = runtime::readout(*job_, recorder_.wires().size(), shots_, 0,
                               compiled_->resultLayout(), samples);
    sampleReadout_ = samples;
  }
  return *result_;
}

QDMIDevice::QDMIDevice(std::shared_ptr<Context> context)
    : context_(std::move(context)) {
  require(context_ != nullptr, "Missing QDMI context.");
}
QDMIDevice::~QDMIDevice() = default;

std::vector<QubitIdType> QDMIDevice::AllocateQubits(size_t count) {
  require(count <= context_->qubits(),
          "Qubit allocation exceeds the device capacity.");
  return recorder_.allocate(count);
}

void QDMIDevice::ReleaseQubits(const std::vector<QubitIdType>& wires) {
  recorder_.release(wires);
  compiled_.reset();
  result_.reset();
  job_.reset();
  sampleReadout_ = false;
  observables_.clear();
  firstObservable_ = nextObservable_;
}

size_t QDMIDevice::GetNumQubits() const { return recorder_.wires().size(); }
void QDMIDevice::SetDeviceShots(size_t shots) {
  validateShots(shots);
  if (shots_ != shots) {
    result_.reset();
    job_.reset();
    sampleReadout_ = false;
  }
  shots_ = shots;
}
size_t QDMIDevice::GetDeviceShots() const { return shots_; }
void QDMIDevice::SetDevicePRNG(std::mt19937* generator) {
  require(generator == nullptr, "Catalyst PRNG seeding is unsupported; use the "
                                "QDMI provider's job parameters.");
}

void QDMIDevice::NamedOperation(
    const std::string& name, const std::vector<double>& parameters,
    const std::vector<QubitIdType>& wires, bool inverse,
    const std::vector<QubitIdType>& controls,
    const std::vector<bool>& controlValues,
    const std::vector<std::string>& optionalParameters) {
  require(optionalParameters.empty(),
          "Optional Catalyst gate parameters are unsupported.");
  require(!result_, "Operations after terminal readout are unsupported.");
  recorder_.namedOperation(name, parameters, wires, inverse, controls,
                           controlValues);
}

void QDMIDevice::MatrixOperation(
    const std::vector<std::complex<double>>& matrix,
    const std::vector<QubitIdType>& wires, bool inverse,
    const std::vector<QubitIdType>& controls,
    const std::vector<bool>& controlValues) {
  require(!result_, "Operations after terminal readout are unsupported.");
  recorder_.matrixOperation(matrix, wires, inverse, controls, controlValues);
}

void QDMIDevice::SetBasisState(DataView<int8_t, 1>& state,
                               std::vector<QubitIdType>& wires) {
  require(!result_, "Operations after terminal readout are unsupported.");
  std::vector<int8_t> values;
  values.reserve(state.size());
  for (const auto bit : state) {
    values.push_back(bit);
  }
  recorder_.setBasisState(values, wires);
}

Result QDMIDevice::Measure(QubitIdType, std::optional<int32_t>) {
  throw std::runtime_error(
      "Interactive mid-circuit QDMI measurement is unsupported.");
}

ObsIdType
QDMIDevice::Observable(ObsId id,
                       const std::vector<std::complex<double>>& matrix,
                       const std::vector<QubitIdType>& wires) {
  auto indices = wireIndices(wires);
  require(matrix.empty() && (id == ObsId::Identity || id == ObsId::PauliZ),
          "QDMI supports only Identity and PauliZ native observables.");
  require(id == ObsId::Identity || wires.size() == 1,
          "PauliZ requires exactly one wire.");
  if (id == ObsId::Identity) {
    indices.clear();
  }
  return addObservable(std::move(indices));
}

ObsIdType
QDMIDevice::TensorObservable(const std::vector<ObsIdType>& observables) {
  require(!observables.empty(), "An observable product cannot be empty.");
  ObservableWires product;
  for (const auto id : observables) {
    const auto& value = getObservable(id);
    product.insert(product.end(), value.begin(), value.end());
  }
  return addObservable(std::move(product));
}

void QDMIDevice::Sample(DataView<double, 2>& samples) {
  PartialSample(samples, recorder_.wires());
}

void QDMIDevice::PartialSample(DataView<double, 2>& samples,
                               const std::vector<QubitIdType>& wires) {
  const auto indices = wireIndices(wires);
  require(samples.size() == product(shots_, indices.size()),
          "Unexpected Catalyst sample buffer size.");
  const auto& result = read(true);
  auto output = samples.begin();
  for (const auto& [bits, count] : result) {
    for (size_t shot = 0; shot < count; ++shot) {
      for (const auto wire : indices) {
        *output++ = static_cast<double>(bits[bits.size() - wire - 1] - '0');
      }
    }
  }
}

void QDMIDevice::Counts(DataView<double, 1>& eigenvalues,
                        DataView<int64_t, 1>& counts) {
  PartialCounts(eigenvalues, counts, recorder_.wires());
}

void QDMIDevice::PartialCounts(DataView<double, 1>& eigenvalues,
                               DataView<int64_t, 1>& counts,
                               const std::vector<QubitIdType>& wires) {
  const auto indices = wireIndices(wires);
  require(indices.size() <= std::numeric_limits<double>::digits,
          "Catalyst count keys cannot exactly represent more than 53 qubits; "
          "use samples.");
  const auto size = dimension(indices.size());
  require(eigenvalues.size() == size && counts.size() == size,
          "Unexpected Catalyst counts buffer size.");
  const auto& result = read();
  for (size_t i = 0; i < size; ++i) {
    eigenvalues(i) = static_cast<double>(i);
    counts(i) = 0;
  }
  for (const auto& [bits, count] : result) {
    counts(runtime::basisIndex(bits, indices)) += static_cast<int64_t>(count);
  }
}

void QDMIDevice::Probs(DataView<double, 1>& probabilities) {
  PartialProbs(probabilities, recorder_.wires());
}

void QDMIDevice::PartialProbs(DataView<double, 1>& probabilities,
                              const std::vector<QubitIdType>& wires) {
  const auto indices = wireIndices(wires);
  require(probabilities.size() == dimension(indices.size()),
          "Unexpected Catalyst probability buffer size.");
  const auto& result = read();
  for (auto& probability : probabilities) {
    probability = 0.0;
  }
  for (const auto& [bits, count] : result) {
    probabilities(runtime::basisIndex(bits, indices)) +=
        static_cast<double>(count) / static_cast<double>(shots_);
  }
}

double QDMIDevice::Expval(ObsIdType observable) {
  const auto& parity = getObservable(observable);
  const auto& result = read();
  size_t negative = 0;
  for (const auto& [bits, count] : result) {
    bool odd = false;
    for (const auto wire : parity) {
      odd = odd != (bits[bits.size() - wire - 1] == '1');
    }
    if (odd) {
      negative += count;
    }
  }
  return 1.0 -
         2.0 * static_cast<double>(negative) / static_cast<double>(shots_);
}

double QDMIDevice::Var(ObsIdType observable) {
  const auto mean = Expval(observable);
  return std::max(0.0, 1.0 - mean * mean);
}

Catalyst::Runtime::QuantumDevice* createDevice(const char* kwargs) {
  const auto id = parseContext(kwargs);
  auto& entries = registry();
  std::shared_ptr<Context> context;
  {
    const std::scoped_lock lock(entries.mutex);
    if (const auto entry = entries.contexts.find(id);
        entry != entries.contexts.end()) {
      context = entry->second.lock();
    }
  }
  require(context != nullptr, "Unknown or expired QDMI context ID.");
  return new QDMIDevice(std::move(context));
}

} // namespace mqt::catalyst
