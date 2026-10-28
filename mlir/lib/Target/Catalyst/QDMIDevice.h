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

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/ResultLayout.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "qdmi/QDMI.hpp"

#include "QuantumDevice.hpp"

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::catalyst {

using JobParameter =
    std::variant<std::string, bool, int, double, std::vector<std::byte>>;
using JobParameters = std::array<std::optional<JobParameter>, 5>;

/// Own one provider session and its retained submission configuration.
class Context {
public:
  static std::shared_ptr<Context> create(
      qdmi::Device device, JobParameters parameters = {},
      mlir::SerializationProfile profile = mlir::SerializationProfile::Default);
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  ~Context();
  [[nodiscard]] uint64_t id() const noexcept;
  [[nodiscard]] size_t qubits() const noexcept;

private:
  explicit Context(qdmi::Device device, JobParameters parameters,
                   mlir::SerializationProfile profile);
  qdmi::Device device_;
  JobParameters parameters_;
  mlir::TargetEnvironment environment_;
  mlir::SerializationProfile profile_;
  size_t qubits_;
  uint64_t id_ = 0;
  friend class QDMIDevice;
};

/// Record one terminal-measurement circuit through Catalyst's device API.
class QDMIDevice final : public Catalyst::Runtime::QuantumDevice {
public:
  /// Construct a provider-independent recorder for a PennyLane tape.
  explicit QDMIDevice(size_t numWires);
  explicit QDMIDevice(std::shared_ptr<Context> context);
  ~QDMIDevice() override;
  /// Consume the recording and expose every wire as a classical output.
  [[nodiscard]] mlir::QCProgram takeProgram();

  std::vector<QubitIdType> AllocateQubits(size_t count) override;
  void ReleaseQubits(const std::vector<QubitIdType>& wires) override;
  [[nodiscard]] size_t GetNumQubits() const override;
  void SetDeviceShots(size_t shots) override;
  [[nodiscard]] size_t GetDeviceShots() const override;
  void SetDevicePRNG(std::mt19937* generator) override;
  void NamedOperation(
      const std::string& name, const std::vector<double>& parameters,
      const std::vector<QubitIdType>& wires, bool inverse = false,
      const std::vector<QubitIdType>& controls = {},
      const std::vector<bool>& controlValues = {},
      const std::vector<std::string>& optionalParameters = {}) override;
  void MatrixOperation(const std::vector<std::complex<double>>& matrix,
                       const std::vector<QubitIdType>& wires,
                       bool inverse = false,
                       const std::vector<QubitIdType>& controls = {},
                       const std::vector<bool>& controlValues = {}) override;
  void SetBasisState(DataView<int8_t, 1>& state,
                     std::vector<QubitIdType>& wires) override;
  Result Measure(QubitIdType wire, std::optional<int32_t> postselect) override;
  ObsIdType Observable(ObsId id,
                       const std::vector<std::complex<double>>& matrix,
                       const std::vector<QubitIdType>& wires) override;
  ObsIdType
  TensorObservable(const std::vector<ObsIdType>& observables) override;
  void Sample(DataView<double, 2>& samples) override;
  void PartialSample(DataView<double, 2>& samples,
                     const std::vector<QubitIdType>& wires) override;
  void Counts(DataView<double, 1>& eigenvalues,
              DataView<int64_t, 1>& counts) override;
  void PartialCounts(DataView<double, 1>& eigenvalues,
                     DataView<int64_t, 1>& counts,
                     const std::vector<QubitIdType>& wires) override;
  void Probs(DataView<double, 1>& probabilities) override;
  void PartialProbs(DataView<double, 1>& probabilities,
                    const std::vector<QubitIdType>& wires) override;
  double Expval(ObsIdType observable) override;
  double Var(ObsIdType observable) override;

private:
  using Readout = llvm::SmallVector<std::pair<std::string, size_t>>;
  using ObservableWires = llvm::SmallVector<size_t>;
  [[nodiscard]] llvm::SmallVector<size_t>
  wireIndices(llvm::ArrayRef<QubitIdType> wires) const;
  [[nodiscard]] const ObservableWires& getObservable(ObsIdType id) const;
  ObsIdType addObservable(ObservableWires value);
  void emitOperation(llvm::ArrayRef<size_t> indices, size_t numControls,
                     const std::vector<bool>& controlValues, bool inverse,
                     llvm::function_ref<void(mlir::ValueRange)> primitive);
  const Readout& read();

  std::shared_ptr<Context> context_;
  std::shared_ptr<mlir::MLIRContext> compiler_;
  std::unique_ptr<mlir::qc::QCProgramBuilder> builder_;
  llvm::SmallVector<mlir::Value> qubits_;
  std::vector<QubitIdType> ids_;
  QubitIdType nextQubit_ = 0;
  bool initialState_ = true;
  llvm::SmallVector<ObservableWires> observables_;
  ObsIdType firstObservable_ = 0;
  ObsIdType nextObservable_ = 0;
  size_t shots_ = 0;
  std::optional<mlir::CompiledProgram> compiled_;
  std::optional<Readout> result_;
};

/// Decode all logical wires into a row-major shot-by-wire array.
[[nodiscard]] std::vector<int8_t>
decodeSamples(const qdmi::Job& job, size_t width, size_t shots,
              size_t programIndex = 0,
              const std::optional<mlir::ResultLayout>& layout = std::nullopt);

/// Resolve a retained configuration for Catalyst's device factory.
[[nodiscard]] Catalyst::Runtime::QuantumDevice*
createDevice(const char* kwargs);

} // namespace mqt::catalyst
