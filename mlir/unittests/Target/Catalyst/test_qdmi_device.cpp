/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/ResultLayout.h"
#include "qdmi/QDMI.hpp"

#include "Driver.hpp"
#include "QDMIDevice.h"

#include "gtest/gtest.h"
#include "qdmi/device.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mqt::catalyst::Context;
using mqt::catalyst::createDevice;
using mqt::catalyst::decodeSamples;
using mqt::catalyst::JobParameters;
using mqt::catalyst::QDMIDevice;

thread_local std::optional<std::string> returnedShots;
thread_local std::string returnedKeys;
thread_local std::vector<size_t> returnedCounts;
thread_local std::vector<std::byte> receivedParameter;
thread_local decltype(QDMI_device_job_set_parameter)* setParameter;

} // namespace

static int copyResult(std::span<const std::byte> bytes, size_t size,
                      void* value, size_t* sizeRet) {
  if (sizeRet != nullptr) {
    *sizeRet = bytes.size();
  }
  if (value != nullptr) {
    if (size < bytes.size()) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::memcpy(value, bytes.data(), bytes.size());
  }
  return QDMI_SUCCESS;
}

static int getResults(QDMI_Device_Job, size_t, QDMI_Job_Result result,
                      size_t size, void* value, size_t* sizeRet) {
  if (result == QDMI_JOB_RESULT_SHOTS && returnedShots) {
    return copyResult(std::as_bytes(std::span(returnedShots->c_str(),
                                              returnedShots->size() + 1)),
                      size, value, sizeRet);
  }
  if (result == QDMI_JOB_RESULT_HIST_KEYS && !returnedKeys.empty()) {
    return copyResult(
        std::as_bytes(std::span(returnedKeys.c_str(), returnedKeys.size() + 1)),
        size, value, sizeRet);
  }
  if (result == QDMI_JOB_RESULT_HIST_VALUES && !returnedKeys.empty()) {
    return copyResult(std::as_bytes(std::span(returnedCounts)), size, value,
                      sizeRet);
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

TEST(CatalystQDMI, ValidatesProviderResultsAndRetainsBinaryParameters) {
  auto device =
      llvm::cantFail(qdmi::builtin_driver::openDevice("mqt.ddsim.default"));
  auto& api =
      const_cast<qdmi::DeviceAPI&>(static_cast<QDMI_Device>(device)->api());
  const auto originalGetResults = api.device_job_get_results;
  setParameter = api.device_job_set_parameter;
  const auto restore = llvm::make_scope_exit([&] {
    api.device_job_get_results = originalGetResults;
    api.device_job_set_parameter = setParameter;
  });
  api.device_job_set_parameter = [](QDMI_Device_Job job,
                                    QDMI_Device_Job_Parameter parameter,
                                    size_t size, const void* value) -> int {
    if (parameter == QDMI_DEVICE_JOB_PARAMETER_CUSTOM5) {
      const auto* bytes = static_cast<const std::byte*>(value);
      receivedParameter.assign(bytes, bytes + size);
      return QDMI_SUCCESS;
    }
    return setParameter(job, parameter, size, value);
  };
  const std::vector<std::byte> expected{
      std::byte{0},
      std::byte{42},
      std::byte{255},
  };
  auto context = [&] {
    JobParameters parameters;
    parameters[0] = 42;
    parameters[4] = expected;
    return Context::create(device, std::move(parameters));
  }();
  QDMIDevice recorder(context);
  const auto wires = recorder.AllocateQubits(3);
  recorder.NamedOperation("PauliX", {}, {wires[0]});
  recorder.SetDeviceShots(2);
  std::vector<double> probabilities(8);
  DataView<double, 1> view(probabilities);
  recorder.Probs(view);
  EXPECT_EQ(receivedParameter, expected);
  EXPECT_EQ(probabilities, (std::vector<double>{0, 0, 0, 0, 1, 0, 0, 0}));
  QDMIDevice importer(3);
  importer.NamedOperation("PauliX", {}, {0});
  const auto program =
      llvm::cantFail(mlir::compileProgram(importer.takeProgram(), device));
  auto job = llvm::cantFail(mlir::submitProgram(device, program, 2));
  ASSERT_TRUE(llvm::cantFail(job.wait()));
  EXPECT_EQ(decodeSamples(job, 3, 2), (std::vector<int8_t>{1, 0, 0, 1, 0, 0}));
  api.device_job_get_results = getResults;
  returnedKeys.clear();
  returnedShots = "10 0,001";
  EXPECT_EQ(decodeSamples(job, 3, 2), (std::vector<int8_t>{0, 0, 1, 1, 0, 0}));
  const mlir::ResultLayout layout{
      .rawWidth = 3,
      .bits = {size_t{2}, false, size_t{0}, size_t{2}},
  };
  EXPECT_EQ(decodeSamples(job, 4, 2, 0, layout),
            (std::vector<int8_t>{1, 0, 0, 1, 0, 0, 1, 0}));
  EXPECT_THROW(decodeSamples(job, 3, 2, 0, layout), std::runtime_error);
  for (const auto* malformed :
       {"001", "0000,001", "0x1,001", "00\xFF,001", "000,001,001"}) {
    returnedShots = malformed;
    EXPECT_THROW(decodeSamples(job, 3, 2), std::runtime_error);
  }
  returnedShots.reset();
  returnedKeys = "100,001";
  returnedCounts = {2, 3};
  const auto samples = decodeSamples(job, 3, 5);
  EXPECT_EQ(samples,
            (std::vector<int8_t>{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1}));
  EXPECT_EQ(decodeSamples(job, 4, 5, 0, layout),
            (std::vector<int8_t>{0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                 1, 0, 1, 0, 0, 1, 1, 0, 0, 1}));
  EXPECT_THROW(decodeSamples(job, 3, 4), std::runtime_error);
  EXPECT_THROW(decodeSamples(job, 3, 6), std::runtime_error);
  returnedKeys = "10x,001";
  EXPECT_THROW(decodeSamples(job, 3, 5), std::runtime_error);
}

TEST(CatalystQDMI, MatrixGatesShareCorrelatedTerminalReadouts) {
  constexpr size_t shots = 512;
  JobParameters parameters;
  parameters[0] = 42;
  QDMIDevice runtime(Context::create(
      llvm::cantFail(qdmi::builtin_driver::openDevice("mqt.ddsim.default")),
      parameters));
  const auto wires = runtime.AllocateQubits(3);
  runtime.SetDeviceShots(shots);
  // H, S-dagger, and RX(pi/6) give P(q0 = 1) = 3/4.
  runtime.NamedOperation("Hadamard", {}, {wires[0]});
  runtime.MatrixOperation({1, 0, 0, {0, 1}}, {wires[0]}, true);
  runtime.NamedOperation("RX", {std::numbers::pi / 6}, {wires[0]});
  runtime.MatrixOperation({0, {0, 1}, {0, 1}, 0}, {wires[2]}, false, {wires[0]},
                          {false});
  runtime.NamedOperation("PauliX", {}, {wires[1]});
  const auto z0 = runtime.Observable(ObsId::PauliZ, {}, {wires[0]});
  const auto z2 = runtime.Observable(ObsId::PauliZ, {}, {wires[2]});
  const auto parity = runtime.TensorObservable({z0, z2});
  const auto identity = runtime.Observable(ObsId::Identity, {}, {});

  const std::array<size_t, 2> sizes{shots, 3};
  const std::array<size_t, 2> strides{3, 1};
  std::vector<double> samples(shots * 3);
  DataView<double, 2> sampleView(samples.data(), 0, sizes.data(),
                                 strides.data());
  runtime.Sample(sampleView);
  std::vector<int64_t> expectedCounts(8);
  for (size_t shot = 0; shot < shots; ++shot) {
    const auto q0 = sampleView(shot, 0);
    const auto q2 = sampleView(shot, 2);
    EXPECT_EQ(sampleView(shot, 1), 1);
    EXPECT_EQ(q0 + q2, 1);
    ++expectedCounts[static_cast<size_t>(4 * q0 + 2 + q2)];
  }
  std::vector<double> eigenvalues(8);
  std::vector<int64_t> counts(8);
  DataView<double, 1> eigenvalueView(eigenvalues);
  DataView<int64_t, 1> countView(counts);
  runtime.Counts(eigenvalueView, countView);
  EXPECT_EQ(eigenvalues, (std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(counts, expectedCounts);
  const auto p = static_cast<double>(expectedCounts[6]) / shots;
  EXPECT_NEAR(p, 0.75, 0.125);
  std::vector<double> probabilities(4);
  DataView<double, 1> probabilityView(probabilities);
  runtime.PartialProbs(probabilityView, {wires[2], wires[0]});
  EXPECT_EQ(probabilities, (std::vector<double>{0, p, 1 - p, 0}));
  EXPECT_EQ(runtime.Expval(z0), 1 - 2 * p);
  EXPECT_EQ(runtime.Var(z0), 1 - (1 - 2 * p) * (1 - 2 * p));
  EXPECT_EQ(runtime.Expval(parity), -1);
  EXPECT_EQ(runtime.Expval(identity), 1);
}

TEST(CatalystQDMI, ReceiverProfilesPreserveWireOrder) {
  const auto device =
      llvm::cantFail(qdmi::builtin_driver::openDevice("mqt.ddsim.default"));
  auto& api =
      const_cast<qdmi::DeviceAPI&>(static_cast<QDMI_Device>(device)->api());
  static thread_local decltype(QDMI_device_session_query_device_property)*
      query;
  static thread_local std::vector<QDMI_Site> sites;
  query = api.device_session_query_device_property;
  const auto allSites = llvm::cantFail(device.getSites());
  sites.clear();
  for (size_t i = 0; i < 3; ++i) {
    sites.push_back(static_cast<QDMI_Site>(allSites[i]));
  }
  const auto restore = llvm::make_scope_exit(
      [&] { api.device_session_query_device_property = query; });
  // IBM declares the target's full register; bound the simulator's 65,535-site
  // capacity.
  api.device_session_query_device_property =
      [](QDMI_Device_Session session, QDMI_Device_Property property,
         size_t size, void* value, size_t* sizeRet) -> int {
    if (property == QDMI_DEVICE_PROPERTY_QUBITSNUM) {
      const size_t count = sites.size();
      return copyResult(std::as_bytes(std::span(&count, 1)), size, value,
                        sizeRet);
    }
    if (property == QDMI_DEVICE_PROPERTY_SITES) {
      return copyResult(std::as_bytes(std::span(sites)), size, value, sizeRet);
    }
    return query(session, property, size, value, sizeRet);
  };
  for (const auto profile : {
           mlir::SerializationProfile::BraketSimulator,
           mlir::SerializationProfile::IBM,
       }) {
    QDMIDevice runtime(Context::create(device, {}, profile));
    const auto wires = runtime.AllocateQubits(3);
    runtime.NamedOperation("PauliX", {}, {wires[1]});
    runtime.SetDeviceShots(2);
    std::vector<double> samples(6);
    const std::array<size_t, 2> sizes{2, 3};
    const std::array<size_t, 2> strides{3, 1};
    DataView<double, 2> view(samples.data(), 0, sizes.data(), strides.data());
    runtime.PartialSample(view, {wires[2], wires[0], wires[1]});
    EXPECT_EQ(samples, (std::vector<double>{0, 0, 1, 0, 0, 1}));
  }
}

TEST(CatalystQDMI, InvalidBasisStateLeavesRecordingUsable) {
  QDMIDevice runtime(Context::create(
      llvm::cantFail(qdmi::builtin_driver::openDevice("mqt.ddsim.default"))));
  const auto wires = runtime.AllocateQubits(3);
  runtime.SetDeviceShots(4);
  std::vector<int8_t> bits{1, 2};
  DataView<int8_t, 1> state(bits);
  std::vector<QubitIdType> prepared{wires[2], wires[0]};
  EXPECT_THROW(runtime.SetBasisState(state, prepared), std::runtime_error);
  bits[1] = 0;
  runtime.SetBasisState(state, prepared);
  std::vector<double> probabilities(8);
  DataView<double, 1> probabilityView(probabilities);
  runtime.Probs(probabilityView);
  EXPECT_EQ(probabilities, (std::vector<double>{0, 1, 0, 0, 0, 0, 0, 0}));
}

TEST(CatalystQDMI, FactoryRetainsContextAndRejectsInvalidLifetimes) {
  auto context = Context::create(
      llvm::cantFail(qdmi::builtin_driver::openDevice("mqt.ddsim.default")));
  const auto kwargs = "{'context': " + std::to_string(context->id()) + '}';
  auto runtime = std::unique_ptr<Catalyst::Runtime::QuantumDevice>(
      createDevice(kwargs.c_str()));
  context.reset();
  auto second = std::unique_ptr<Catalyst::Runtime::QuantumDevice>(
      createDevice(kwargs.c_str()));
  const auto wires = runtime->AllocateQubits(1);
  EXPECT_THROW(runtime->NamedOperation(
                   "RY", {std::numeric_limits<double>::quiet_NaN()}, wires),
               std::runtime_error);
  runtime->ReleaseQubits(wires);
  const auto replacement = runtime->AllocateQubits(1);
  EXPECT_THROW(runtime->NamedOperation("PauliX", {}, wires),
               std::runtime_error);
  runtime->ReleaseQubits(replacement);
  runtime.reset();
  second.reset();
  EXPECT_THROW(createDevice(kwargs.c_str()), std::runtime_error);
}
