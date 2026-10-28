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
#include "mqt/Dialect/MQT/Utils/DenseUnitary.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/Translation/StandardGate.h"
#include "qdmi/common/Common.hpp"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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

using Readout = llvm::SmallVector<std::pair<std::string, size_t>>;

using Gate = mlir::qc::StandardGate;
struct NamedGate {
  llvm::StringLiteral name;
  Gate gate;
  size_t controls = 0;
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

static Readout readout(const qdmi::Job& job, size_t width, size_t shots,
                       size_t programIndex,
                       const std::optional<mlir::ResultLayout>& layout) {
  validateShots(shots);
  Readout result;
  std::string shotsError;
  try {
    auto samples = unwrap(job.getShots(programIndex));
    result.reserve(samples.size());
    for (auto& bits : samples) {
      result.emplace_back(std::move(bits), 1);
    }
  } catch (const std::exception& error) {
    shotsError = error.what();
  }
  if (result.empty()) {
    try {
      for (const auto& [bits, count] : unwrap(job.getCounts(programIndex))) {
        if (count != 0) {
          result.emplace_back(bits, count);
        }
      }
    } catch (const std::exception& error) {
      throw std::runtime_error("Cannot read QDMI samples; shots: " +
                               shotsError + "; counts: " + error.what());
    }
  }
  size_t total = 0;
  for (auto& [bits, count] : result) {
    std::erase(bits, ' ');
    if (layout) {
      bits = unwrap(layout->decodeBits(bits));
    }
    require(bits.size() == width &&
                bits.find_first_not_of("01") == std::string::npos,
            "QDMI returned a malformed " + llvm::Twine(width) + "-bit sample.");
    require(count <= shots - total,
            "QDMI returned more samples than requested.");
    total += count;
    std::ranges::reverse(bits);
  }
  require(total == shots, "QDMI returned " + std::to_string(total) +
                              " samples for a " + std::to_string(shots) +
                              "-shot job.");
  return result;
}

static size_t basisIndex(const std::string& bits,
                         llvm::ArrayRef<size_t> wires) {
  size_t index = 0;
  for (const auto wire : wires) {
    index = (index << 1U) | static_cast<size_t>(bits[wire] - '0');
  }
  return index;
}

constexpr std::array GATES{
    NamedGate{.name = "Identity", .gate = Gate::Id},
    NamedGate{.name = "PauliX", .gate = Gate::X},
    NamedGate{.name = "PauliY", .gate = Gate::Y},
    NamedGate{.name = "PauliZ", .gate = Gate::Z},
    NamedGate{.name = "Hadamard", .gate = Gate::H},
    NamedGate{.name = "S", .gate = Gate::S},
    NamedGate{.name = "T", .gate = Gate::T},
    NamedGate{.name = "SX", .gate = Gate::SX},
    NamedGate{.name = "RX", .gate = Gate::RX},
    NamedGate{.name = "RY", .gate = Gate::RY},
    NamedGate{.name = "RZ", .gate = Gate::RZ},
    NamedGate{.name = "PhaseShift", .gate = Gate::P},
    NamedGate{.name = "GlobalPhase", .gate = Gate::GPhase},
    NamedGate{.name = "U1", .gate = Gate::P},
    NamedGate{.name = "U2", .gate = Gate::U2},
    NamedGate{.name = "U3", .gate = Gate::U3},
    NamedGate{.name = "SWAP", .gate = Gate::SWAP},
    NamedGate{.name = "ISWAP", .gate = Gate::ISWAP},
    NamedGate{.name = "ECR", .gate = Gate::ECR},
    NamedGate{.name = "IsingXX", .gate = Gate::RXX},
    NamedGate{.name = "IsingYY", .gate = Gate::RYY},
    NamedGate{.name = "IsingZZ", .gate = Gate::RZZ},
    NamedGate{.name = "CNOT", .gate = Gate::X, .controls = 1},
    NamedGate{.name = "CY", .gate = Gate::Y, .controls = 1},
    NamedGate{.name = "CZ", .gate = Gate::Z, .controls = 1},
    NamedGate{.name = "Toffoli", .gate = Gate::X, .controls = 2},
    NamedGate{.name = "CSWAP", .gate = Gate::SWAP, .controls = 1},
    NamedGate{.name = "ControlledPhaseShift", .gate = Gate::P, .controls = 1},
    NamedGate{.name = "CRX", .gate = Gate::RX, .controls = 1},
    NamedGate{.name = "CRY", .gate = Gate::RY, .controls = 1},
    NamedGate{.name = "CRZ", .gate = Gate::RZ, .controls = 1},
};

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
                 mlir::SerializationProfile profile)
    : device_(std::move(device)), parameters_(std::move(parameters)),
      environment_(unwrap(mlir::targetEnvironmentFromDevice(
          device_, profile == mlir::SerializationProfile::Default
                       ? std::nullopt
                       : std::make_optional(QDMI_PROGRAM_FORMAT_QASM3)))),
      profile_(profile), qubits_(unwrap(device_.getQubitsNum())) {}

std::shared_ptr<Context> Context::create(qdmi::Device device,
                                         JobParameters parameters,
                                         mlir::SerializationProfile profile) {
  auto context = std::shared_ptr<Context>(
      new Context(std::move(device), std::move(parameters), profile));
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

std::vector<int8_t>
decodeSamples(const qdmi::Job& job, size_t width, size_t shots,
              size_t programIndex,
              const std::optional<mlir::ResultLayout>& layout) {
  const auto result = readout(job, width, shots, programIndex, layout);
  std::vector<int8_t> values;
  values.reserve(product(shots, width));
  for (const auto& [bits, count] : result) {
    for (size_t i = 0; i < count; ++i) {
      for (const char bit : bits) {
        values.push_back(static_cast<int8_t>(bit - '0'));
      }
    }
  }
  return values;
}

llvm::SmallVector<size_t>
QDMIDevice::wireIndices(llvm::ArrayRef<QubitIdType> wires) const {
  require(!ids_.empty(), "No active QDMI qubit register.");
  llvm::SmallVector<size_t> result;
  result.reserve(wires.size());
  for (const auto wire : wires) {
    require(wire >= ids_.front() &&
                static_cast<uint64_t>(wire - ids_.front()) < ids_.size(),
            "Invalid or released QDMI qubit ID.");
    const auto index = static_cast<size_t>(wire - ids_.front());
    require(std::ranges::find(result, index) == result.end(),
            "Duplicate QDMI qubit operand.");
    result.push_back(index);
  }
  return result;
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

const QDMIDevice::Readout& QDMIDevice::read() {
  validateShots(shots_);
  require(context_ != nullptr, "No QDMI execution context is configured.");
  if (!result_) {
    if (!compiled_) {
      compiled_ = unwrap(mlir::CompiledProgram::compile(
          takeProgram(), context_->environment_, {}, context_->profile_));
    }
    const auto parameters = borrowParameters(context_->parameters_);
    auto job = unwrap(mlir::submitProgram(
        context_->device_, *compiled_, static_cast<int64_t>(shots_),
        parameters[0], parameters[1], parameters[2], parameters[3],
        parameters[4]));
    require(unwrap(job.wait()) && unwrap(job.check()) == QDMI_JOB_STATUS_DONE,
            "QDMI job did not complete successfully.");
    result_ = readout(job, ids_.size(), shots_, 0, compiled_->resultLayout());
  }
  return *result_;
}

QDMIDevice::QDMIDevice(size_t numWires) { AllocateQubits(numWires); }

QDMIDevice::QDMIDevice(std::shared_ptr<Context> context)
    : context_(std::move(context)) {
  require(context_ != nullptr, "Missing QDMI context.");
}
QDMIDevice::~QDMIDevice() = default;

std::vector<QubitIdType> QDMIDevice::AllocateQubits(size_t count) {
  require(ids_.empty(), "Dynamic QDMI qubit allocation is unsupported.");
  require(count > 0 && (!context_ || count <= context_->qubits()) &&
              count <= static_cast<size_t>(INTPTR_MAX) -
                           static_cast<size_t>(nextQubit_),
          "QDMI qubit allocation exceeds the device capacity.");
  compiler_ = mlir::createCompilerContext();
  builder_ = std::make_unique<mlir::qc::QCProgramBuilder>(compiler_.get());
  builder_->initialize();
  auto reg = builder_->allocQubitRegister(static_cast<int64_t>(count));
  qubits_ = std::move(reg.qubits);
  ids_.resize(count);
  std::iota(ids_.begin(), ids_.end(), nextQubit_);
  nextQubit_ += static_cast<QubitIdType>(count);
  initialState_ = true;
  return ids_;
}

void QDMIDevice::ReleaseQubits(const std::vector<QubitIdType>& wires) {
  const auto indices = wireIndices(wires);
  require(indices.size() == ids_.size(),
          "Partial QDMI qubit release is unsupported.");
  builder_.reset();
  qubits_.clear();
  ids_.clear();
  compiled_.reset();
  result_.reset();
  observables_.clear();
  firstObservable_ = nextObservable_;
  compiler_.reset();
}

size_t QDMIDevice::GetNumQubits() const { return ids_.size(); }
void QDMIDevice::SetDeviceShots(size_t shots) {
  validateShots(shots);
  if (shots_ != shots) {
    result_.reset();
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
  require(builder_ != nullptr,
          "QDMI gates require an active register before terminal measurement.");
  require(optionalParameters.empty(),
          "Optional Catalyst gate parameters are unsupported.");
  require(controls.size() == controlValues.size(),
          "Control wires and values have different lengths.");
  const auto gates = llvm::ArrayRef(GATES);
  const auto* const gate =
      llvm::find_if(gates, [&](const NamedGate& candidate) {
        return candidate.name == name;
      });
  require(gate != gates.end(), "Unsupported QDMI gate: " + name);
  const auto& descriptor = mlir::qc::getStandardGateDescriptor(gate->gate);
  require(parameters.size() == descriptor.parameterCount &&
              (gate->gate == Gate::GPhase || gate->gate == Gate::Id ||
               wires.size() == descriptor.targetCount + gate->controls),
          "Invalid parameter or wire count for " + name + '.');
  require(std::ranges::all_of(
              parameters, [](double value) { return std::isfinite(value); }),
          "Non-finite parameter for " + name + '.');
  auto operands = controls;
  operands.insert(operands.end(), wires.begin(), wires.end());
  const auto indices = wireIndices(operands);
  if (gate->gate == Gate::Id) {
    initialState_ = false;
    return;
  }
  auto& builder = *builder_;
  llvm::SmallVector<mlir::Value> angles;
  for (const double angle : parameters) {
    angles.push_back(
        builder.floatConstant(gate->gate == Gate::GPhase ? -angle : angle));
  }
  auto operationIndices = llvm::ArrayRef(indices);
  if (gate->gate == Gate::GPhase) {
    operationIndices = operationIndices.take_front(controls.size());
  }
  emitOperation(
      operationIndices, controls.size() + gate->controls, controlValues,
      inverse, [&](mlir::ValueRange values) {
        require(mlir::succeeded(mlir::qc::emitStandardGate(
                    builder, builder.getLoc(), gate->gate, angles, values)),
                "Cannot emit QDMI gate " + name + '.');
      });
}

void QDMIDevice::emitOperation(
    llvm::ArrayRef<size_t> indices, size_t numControls,
    const std::vector<bool>& controlValues, bool inverse,
    llvm::function_ref<void(mlir::ValueRange)> primitive) {
  auto& builder = *builder_;
  llvm::SmallVector<mlir::Value> controlQubits;
  llvm::SmallVector<mlir::Value> targets;
  for (size_t i = 0; i < indices.size(); ++i) {
    (i < numControls ? controlQubits : targets).push_back(qubits_[indices[i]]);
  }
  const auto emit = [&](mlir::ValueRange values) {
    if (inverse) {
      builder.inv(values, primitive);
    } else {
      primitive(values);
    }
  };
  for (size_t i = 0; i < controlValues.size(); ++i) {
    if (!controlValues[i]) {
      builder.x(controlQubits[i]);
    }
  }
  if (controlQubits.empty()) {
    emit(targets);
  } else {
    builder.ctrl(controlQubits, targets, emit);
  }
  for (size_t i = 0; i < controlValues.size(); ++i) {
    if (!controlValues[i]) {
      builder.x(controlQubits[i]);
    }
  }
  initialState_ = false;
}

void QDMIDevice::MatrixOperation(
    const std::vector<std::complex<double>>& matrix,
    const std::vector<QubitIdType>& wires, bool inverse,
    const std::vector<QubitIdType>& controls,
    const std::vector<bool>& controlValues) {
  require(builder_ != nullptr,
          "QDMI gates require an active register before terminal measurement.");
  require(
      !wires.empty() && wires.size() <= 2 &&
          controls.size() <= 2 - wires.size(),
      "QDMI dense unitaries support at most two wires, including controls.");
  require(controls.size() == controlValues.size(),
          "Control wires and values have different lengths.");
  const auto size = dimension(wires.size());
  require(matrix.size() == size * size,
          "QDMI unitary matrix dimensions do not match its wires.");
  auto operands = controls;
  operands.insert(operands.end(), wires.begin(), wires.end());
  const auto indices = wireIndices(operands);
  llvm::SmallVector<mlir::Value> targets;
  for (const auto index : llvm::ArrayRef(indices).drop_front(controls.size())) {
    targets.push_back(qubits_[index]);
  }
  auto type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(size), static_cast<int64_t>(size)},
      mlir::ComplexType::get(builder_->getF64Type()));
  auto values = mlir::DenseElementsAttr::get(
      type, llvm::ArrayRef<std::complex<double>>(matrix));
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      compiler_.get(), [&](mlir::Diagnostic& diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  const auto valid = mlir::succeeded(mlir::mqt::verifyDenseUnitaryMatrix(
      builder_->getInsertionBlock()->getParentOp(), values, targets));
  require(valid, "Invalid QDMI unitary matrix: " + diagnostics);
  emitOperation(
      indices, controls.size(), controlValues, inverse,
      [&](mlir::ValueRange qubits) { builder_->unitary(qubits, values); });
}

void QDMIDevice::SetBasisState(DataView<int8_t, 1>& state,
                               std::vector<QubitIdType>& wires) {
  require(builder_ != nullptr && initialState_,
          "QDMI basis state preparation must precede all other operations.");
  require(state.size() == wires.size() && !wires.empty(),
          "QDMI basis state length must match its wires.");
  const auto indices = wireIndices(wires);
  for (const auto bit : state) {
    require(bit == 0 || bit == 1,
            "QDMI basis state entries must be zero or one.");
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (state(i) == 1) {
      builder_->x(qubits_[indices[i]]);
    }
  }
  initialState_ = false;
}

Result QDMIDevice::Measure(QubitIdType, std::optional<int32_t>) {
  throw std::runtime_error(
      "Interactive mid-circuit QDMI measurement is unsupported.");
}

mlir::QCProgram QDMIDevice::takeProgram() {
  require(builder_ != nullptr, "No QDMI circuit is available for preparation.");
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      compiler_.get(), [&](mlir::Diagnostic& diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        stream << '\n';
        return mlir::success();
      });
  auto& builder = *builder_;
  auto bits = builder.allocClassicalBitRegister(
      static_cast<int64_t>(qubits_.size()), "c");
  for (size_t i = 0; i < qubits_.size(); ++i) {
    builder.measure(qubits_[i], bits, static_cast<int64_t>(i));
  }
  auto moduleOp = builder.finalize({bits});
  builder_.reset();
  auto program = mlir::QCProgram::fromModule(compiler_, std::move(moduleOp));
  require(mlir::succeeded(program),
          "Invalid recorded QDMI circuit: " + diagnostics);
  return std::move(*program);
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
  PartialSample(samples, ids_);
}

void QDMIDevice::PartialSample(DataView<double, 2>& samples,
                               const std::vector<QubitIdType>& wires) {
  const auto indices = wireIndices(wires);
  require(samples.size() == product(shots_, indices.size()),
          "Unexpected Catalyst sample buffer size.");
  const auto& result = read();
  auto output = samples.begin();
  for (const auto& [bits, count] : result) {
    for (size_t shot = 0; shot < count; ++shot) {
      for (const auto wire : indices) {
        *output++ = static_cast<double>(bits[wire] - '0');
      }
    }
  }
}

void QDMIDevice::Counts(DataView<double, 1>& eigenvalues,
                        DataView<int64_t, 1>& counts) {
  PartialCounts(eigenvalues, counts, ids_);
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
    counts(basisIndex(bits, indices)) += static_cast<int64_t>(count);
  }
}

void QDMIDevice::Probs(DataView<double, 1>& probabilities) {
  PartialProbs(probabilities, ids_);
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
    probabilities(basisIndex(bits, indices)) +=
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
      odd = odd != (bits[wire] == '1');
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
