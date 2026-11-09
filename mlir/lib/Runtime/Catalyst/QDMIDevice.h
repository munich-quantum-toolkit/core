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

#include "CircuitRecorder.h"
#include "QDMIReadout.h"
#include "QuantumDevice.hpp"

#include "llvm/ADT/ArrayRef.h"
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

namespace mqt::catalyst {

using JobParameter =
    std::variant<std::string, bool, int, double, std::vector<std::byte>>;
using JobParameters = std::array<std::optional<JobParameter>, 5>;

/// Own one provider session and its retained submission configuration.
class Context {
public:
  static std::shared_ptr<Context>
  create(qdmi::Device device, JobParameters parameters = {},
         std::optional<QDMI_Program_Format> programFormat = std::nullopt);
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  ~Context();
  [[nodiscard]] uint64_t id() const noexcept;
  [[nodiscard]] size_t qubits() const noexcept;
  [[nodiscard]] std::optional<qdmi::Job>
  trySubmit(llvm::ArrayRef<const mlir::CompiledProgram*> programs,
            int64_t shots) const;
  [[nodiscard]] qdmi::Job submit(const mlir::CompiledProgram& program,
                                 int64_t shots) const;
  [[nodiscard]] const mlir::TargetEnvironment& environment() const noexcept {
    return environment_;
  }

private:
  explicit Context(qdmi::Device device, JobParameters parameters,
                   std::optional<QDMI_Program_Format> programFormat);
  qdmi::Device device_;
  JobParameters parameters_;
  mlir::TargetEnvironment environment_;
  size_t qubits_;
  uint64_t id_ = 0;
  friend class QDMIDevice;
};

/// Record one terminal-measurement circuit through Catalyst's device API.
class QDMIDevice final : public Catalyst::Runtime::QuantumDevice {
public:
  explicit QDMIDevice(std::shared_ptr<Context> context);
  ~QDMIDevice() override;
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
  using ObservableWires = llvm::SmallVector<size_t>;
  [[nodiscard]] llvm::SmallVector<size_t>
  wireIndices(llvm::ArrayRef<QubitIdType> wires) const;
  [[nodiscard]] const ObservableWires& getObservable(ObsIdType id) const;
  ObsIdType addObservable(ObservableWires value);
  const runtime::Readout& read(bool samples = false);

  std::shared_ptr<Context> context_;
  pennylane::CircuitRecorder recorder_;
  llvm::SmallVector<ObservableWires> observables_;
  ObsIdType firstObservable_ = 0;
  ObsIdType nextObservable_ = 0;
  size_t shots_ = 0;
  std::optional<mlir::CompiledProgram> compiled_;
  std::optional<runtime::Readout> result_;
  std::optional<qdmi::Job> job_;
  bool sampleReadout_ = false;
};

/// Resolve a retained configuration for Catalyst's device factory.
[[nodiscard]] Catalyst::Runtime::QuantumDevice*
createDevice(const char* kwargs);

} // namespace mqt::catalyst
