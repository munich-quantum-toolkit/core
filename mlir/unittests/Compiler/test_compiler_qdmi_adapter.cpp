/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/Target.h"
#include "qdmi/Client.hpp"
#include "qdmi/driver/Driver.hpp"

#include "gtest/gtest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using mlir::CompilerTarget;

[[nodiscard]] static const CompilerTarget::OperationCapability&
findOperation(const CompilerTarget& target, const llvm::StringRef name) {
  const auto* const found =
      llvm::find_if(target.operations(),
                    [&](const auto& op) { return op.canonicalName() == name; });
  assert(found != target.operations().end() && "Target operation not found");
  return *found;
}

TEST(CompilerQDMIAdapterTest, SnapshotsIQMCalibrationAndLifetime) {
  const auto target = llvm::cantFail([] {
    const auto device = qdmi::Session::openDevice("mqt.sc.iqm.garnet");
    return mlir::compilerTargetFromDevice(device);
  }());

  ASSERT_TRUE(target.name());
  EXPECT_EQ(*target.name(), "IQM Garnet");
  EXPECT_EQ(target.numSites(), 20);
  EXPECT_EQ(target.connectivityKind(),
            CompilerTarget::Connectivity::Kind::Explicit);
  EXPECT_EQ(target.couplings().size(), 30);

  ASSERT_TRUE(target.durationUnit());
  EXPECT_EQ(target.durationUnit()->unit(), "us");
  EXPECT_DOUBLE_EQ(target.durationUnit()->scaleFactor(), 0.001);

  ASSERT_EQ(target.sites().size(), 20);
  ASSERT_TRUE(target.sites().front().name());
  EXPECT_EQ(*target.sites().front().name(), "QB1");
  EXPECT_EQ(target.sites().front().t1(), 26626);
  EXPECT_EQ(target.sites().front().t2(), 8376);

  ASSERT_EQ(target.operations().size(), 3);
  const auto& r = findOperation(target, "r");
  const auto& cz = findOperation(target, "cz");
  const auto& measure = findOperation(target, "measure");
  EXPECT_EQ(r.siteTuples().size(), 20);
  EXPECT_EQ(cz.siteTuples().size(), 30);
  EXPECT_EQ(measure.siteTuples().size(), 20);
  for (const auto& operation : target.operations()) {
    EXPECT_FALSE(operation.duration());
    for (const auto& tuple : operation.siteTuples()) {
      EXPECT_FALSE(tuple.duration());
      EXPECT_TRUE(tuple.fidelity());
    }
  }

  EXPECT_EQ(target.supportsOperation("r", 1, 2), true);
  EXPECT_EQ(target.supportsOperation("cz", 2, 0), true);
  EXPECT_EQ(target.supportsOperation("measure", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("rx", 1, 1), false);
  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->singleQubit,
            CompilerTarget::SingleQubitBasis::R);
  EXPECT_EQ(target.synthesisBasis()->entangler, CompilerTarget::GateKind::CZ);
}

TEST(CompilerQDMIAdapterTest, QueriesNamesAndSiteIndicesOncePerSnapshot) {
  auto library = std::make_shared<qdmi::DynamicDeviceLibrary>(
      MQT_CORE_MLIR_SC_DEVICE_LIBRARY, "MQT_SC");
  static thread_local decltype(QDMI_device_session_query_site_property)*
      querySite = nullptr;
  static thread_local decltype(QDMI_device_session_query_operation_property)*
      queryOperation = nullptr;
  static thread_local size_t indexQueries = 0;
  static thread_local size_t nameQueries = 0;
  querySite = library->device_session_query_site_property;
  queryOperation = library->device_session_query_operation_property;
  library->device_session_query_site_property =
      [](QDMI_Device_Session session, QDMI_Site site,
         QDMI_Site_Property property, size_t size, void* value,
         size_t* sizeRet) {
        indexQueries += property == QDMI_SITE_PROPERTY_INDEX;
        return querySite(session, site, property, size, value, sizeRet);
      };
  library->device_session_query_operation_property =
      [](QDMI_Device_Session session, QDMI_Operation operation, size_t numSites,
         const QDMI_Site* sites, size_t numParams, const double* params,
         QDMI_Operation_Property property, size_t size, void* value,
         size_t* sizeRet) {
        nameQueries += property == QDMI_OPERATION_PROPERTY_NAME;
        return queryOperation(session, operation, numSites, sites, numParams,
                              params, property, size, value, sizeRet);
      };
  QDMI_Device_impl_d rawDevice(library);
  const auto device = qdmi::Session::createSessionlessDevice(&rawDevice);
  for (int snapshot = 0; snapshot < 2; ++snapshot) {
    indexQueries = 0;
    nameQueries = 0;
    const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));
    /// Bound provider calls independently of coupling and operation counts.
    EXPECT_EQ(indexQueries, target.numSites());
    EXPECT_EQ(nameQueries, 2 * target.operations().size());
    EXPECT_EQ(target.numSites(), 100);
    EXPECT_EQ(target.operations().size(), 3);
  }
}

TEST(CompilerQDMIAdapterTest, InfersDDSIMTargetFacts) {
  const auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  EXPECT_EQ(target.numSites(), 65535);
  EXPECT_EQ(target.connectivityKind(),
            CompilerTarget::Connectivity::Kind::AllToAll);
  EXPECT_EQ(target.nativeOperationsKind(),
            CompilerTarget::NativeOperations::Kind::Explicit);
  const auto& gphase = findOperation(target, "gphase");
  EXPECT_EQ(gphase.arity().kind(),
            CompilerTarget::OperationCapability::Arity::Kind::Fixed);
  EXPECT_EQ(gphase.arity().value(), 0);
  for (const auto [name, minimum] :
       std::initializer_list<std::pair<llvm::StringRef, size_t>>{
           {"id", 1},
           {"h", 1},
           {"rx", 1},
           {"swap", 2},
           {"rxx", 2},
           {"rccx", 3},
       }) {
    const auto& operation = findOperation(target, name);
    EXPECT_EQ(operation.arity().kind(),
              CompilerTarget::OperationCapability::Arity::Kind::Variadic)
        << name.str();
    EXPECT_EQ(operation.arity().value(), minimum) << name.str();
    EXPECT_TRUE(operation.siteTuples().empty()) << name.str();
    EXPECT_TRUE(
        target.supportsOperation(name, minimum, operation.numParameters()))
        << name.str();
    EXPECT_TRUE(
        target.supportsOperation(name, minimum + 4, operation.numParameters()))
        << name.str();
    std::vector<CompilerTarget::SiteId> sites(minimum + 4);
    std::iota(sites.begin(), sites.end(), 0);
    EXPECT_TRUE(target.supportsOperation(name, minimum + 4,
                                         operation.numParameters(), sites))
        << name.str();
  }
  EXPECT_TRUE(target.supportsOperation("gphase", 0, 1));
  EXPECT_EQ(target.supportsOperation("h", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("cx", 2, 0), true);
  EXPECT_EQ(target.supportsOperation("cswap", 3, 0), true);
  EXPECT_EQ(target.supportsOperation("measure", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("reset", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("barrier", 0, 0), false);
}

TEST(CompilerQDMIAdapterTest, ListsRegisteredDeviceIds) {
  const auto deviceIds = llvm::cantFail(mlir::registeredQDMIDeviceIds());
  EXPECT_TRUE(llvm::is_contained(deviceIds, "mqt.ddsim.default"));
}

TEST(CompilerQDMIAdapterTest, ConvertsUnknownDeviceFailureToError) {
  auto target = mlir::compilerTargetFromDeviceId("mqt.unknown.device");
  ASSERT_FALSE(target);
  const auto message = llvm::toString(target.takeError());
  EXPECT_NE(message.find("mqt.unknown.device"), std::string::npos);
  EXPECT_NE(message.find("Unknown QDMI device ID"), std::string::npos);
}

TEST(CompilerQDMIAdapterTest, RejectsNonhomogeneousOperationSupport) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration =
      qdmi::FileDeviceConfiguration{MQT_CORE_MLIR_HETEROGENEOUS_SC_CONFIG};
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  auto target = mlir::compilerTargetFromDevice(device);
  ASSERT_FALSE(target);
  const auto message = llvm::toString(target.takeError());
  EXPECT_NE(message.find("homogeneous"), std::string::npos);
  EXPECT_NE(message.find("all topology edges"), std::string::npos);
}

TEST(CompilerQDMIAdapterTest, SnapshotsHomogeneousHigherArityOperation) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration =
      qdmi::FileDeviceConfiguration{MQT_CORE_MLIR_HIGHER_ARITY_SC_CONFIG};
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0));
  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0, {0, 1, 2}));
  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0, {2, 1, 0}));
  EXPECT_FALSE(target.supportsOperation("ccnot", 3, 0, {0, 1, 3}));
}

TEST(CompilerQDMIAdapterTest, PreservesOneWayDirectionalOperationSupport) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::FileDeviceConfiguration{
      MQT_CORE_MLIR_DIRECTIONAL_ONE_WAY_SC_CONFIG,
  };
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  ASSERT_EQ(target.couplings().size(), 1U);
  const auto& cx = findOperation(target, "cx");
  ASSERT_EQ(cx.siteTuples().size(), 1U);
  EXPECT_EQ(cx.siteTuples()[0].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{0, 1}));
  EXPECT_FALSE(cx.siteTuples()[0].duration());
  EXPECT_FALSE(cx.siteTuples()[0].fidelity());
  EXPECT_TRUE(target.supportsOperation("cx", 2, 0, {0, 1}));
  EXPECT_FALSE(target.supportsOperation("cx", 2, 0, {1, 0}));
  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->entangler, CompilerTarget::GateKind::CX);
}

TEST(CompilerQDMIAdapterTest, OmitsOperationsWithNoSupportedPlacements) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::InlineDeviceConfiguration{
      .json = R"({
    "schema-version": 1,
    "name": "Unavailable operation",
    "numQubits": 1,
    "durationUnit": {"unit": "ns", "scaleFactor": 1},
    "qubitProperties": {"defaults": {}, "overrides": []},
    "couplings": [],
    "operations": [
      {"name": "x", "numQubits": 1, "numParameters": 0, "sites": []}
    ]
  })",
  };
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));
  EXPECT_EQ(target.nativeOperationsKind(),
            CompilerTarget::NativeOperations::Kind::Explicit);
  EXPECT_TRUE(target.operations().empty());
  EXPECT_FALSE(target.supportsOperation("x", 1, 0, {0}));
}

TEST(CompilerQDMIAdapterTest,
     PreservesDirectionalCalibrationWhenBothOrientationsExist) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::FileDeviceConfiguration{
      MQT_CORE_MLIR_DIRECTIONAL_TWO_WAY_SC_CONFIG,
  };
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  ASSERT_EQ(target.couplings().size(), 1);
  const auto& cx = findOperation(target, "cx");
  EXPECT_TRUE(target.supportsOperation("cx", 2, 0, {0, 1}));
  EXPECT_TRUE(target.supportsOperation("cx", 2, 0, {1, 0}));
  ASSERT_EQ(cx.siteTuples().size(), 2);
  EXPECT_EQ(cx.siteTuples()[0].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{0, 1}));
  EXPECT_DOUBLE_EQ(*cx.siteTuples()[0].fidelity(), 0.91);
  EXPECT_EQ(cx.siteTuples()[1].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{1, 0}));
  EXPECT_DOUBLE_EQ(*cx.siteTuples()[1].fidelity(), 0.92);
}

TEST(CompilerQDMIAdapterTest,
     SelectsPayloadByPreferenceAndIncludesMaximalCapabilities) {
  auto library = std::make_shared<qdmi::DynamicDeviceLibrary>(
      MQT_CORE_MLIR_DDSIM_DEVICE_LIBRARY, "MQT_DDSIM");
  static thread_local decltype(QDMI_device_session_query_device_property)*
      query = nullptr;
  static thread_local std::vector<QDMI_Program_Format> formats;
  query = library->device_session_query_device_property;
  library->device_session_query_device_property =
      [](QDMI_Device_Session session, QDMI_Device_Property property,
         size_t size, void* value, size_t* sizeRet) -> int {
    if (property != QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS) {
      return query(session, property, size, value, sizeRet);
    }
    const size_t required = formats.size() * sizeof(QDMI_Program_Format);
    if (sizeRet != nullptr) {
      *sizeRet = required;
    }
    if (value != nullptr) {
      if (size < required) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      std::memcpy(value, formats.data(), required);
    }
    return QDMI_SUCCESS;
  };
  QDMI_Device_impl_d rawDevice(library);
  const auto device = qdmi::Session::createSessionlessDevice(&rawDevice);
  formats = {
      QDMI_PROGRAM_FORMAT_QIRBASESTRING,
      QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
      QDMI_PROGRAM_FORMAT_QASM3,
      QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
      QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
  };
  while (!formats.empty()) {
    const auto environment =
        llvm::cantFail(mlir::targetEnvironmentFromDevice(device));
    const auto baseline = llvm::cantFail(
        mlir::payloadSpecificationForProgramFormat(formats.back()));
    EXPECT_EQ(environment.payloadSpecification().format(), baseline.format());
    EXPECT_EQ(environment.payloadSpecification().capabilities(),
              baseline.capabilities());
    formats.pop_back();
  }
  EXPECT_TRUE(
      llvm::errorToBool(mlir::targetEnvironmentFromDevice(device).takeError()));
  formats = {QDMI_PROGRAM_FORMAT_QASM3};
  EXPECT_TRUE(
      llvm::errorToBool(mlir::targetEnvironmentFromDevice(
                            device, QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE)
                            .takeError()));
  EXPECT_TRUE(llvm::errorToBool(
      mlir::payloadSpecificationForProgramFormat(QDMI_PROGRAM_FORMAT_QASM2)
          .takeError()));

  const auto adaptive =
      llvm::cantFail(mlir::payloadSpecificationForProgramFormat(
          QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE));
  ASSERT_EQ(adaptive.capabilities().size(), 11);
  EXPECT_EQ(adaptive.capabilities().front().id, "forward-branching");
  const auto qasm = llvm::cantFail(
      mlir::payloadSpecificationForProgramFormat(QDMI_PROGRAM_FORMAT_QASM3));
  EXPECT_EQ(qasm.format().version, "3.1.0");
  ASSERT_EQ(qasm.capabilities().size(), 4);
  EXPECT_EQ(qasm.capabilities()[1].id, "counted-iteration");
  EXPECT_EQ(qasm.capabilities()[2].id, "conditional-loop");
  EXPECT_EQ(qasm.capabilities()[3].id, "multiway-branching");
}

TEST(CompilerQDMIAdapterTest,
     CompatibilityIgnoresCalibrationButPreservesLegality) {
  const auto makeTarget = [](bool calibrated,
                             const std::vector<CompilerTarget::SiteId>& ids,
                             std::vector<CompilerTarget::SiteId> operands,
                             double timeScale = 1.) {
    std::vector<CompilerTarget::Site> sites;
    sites.reserve(ids.size());
    for (const auto id : ids) {
      sites.push_back(llvm::cantFail(CompilerTarget::Site::create(
          id, calibrated ? "new name" : "old name", calibrated ? 200 : 100)));
    }
    const auto tuple = llvm::cantFail(CompilerTarget::SiteTuple::create(
        std::move(operands), calibrated ? 50 : 10, calibrated ? 0.99 : 0.9));
    const auto operation =
        llvm::cantFail(CompilerTarget::OperationCapability::create(
            "cx", 2, 0, {tuple}, calibrated ? 50 : 10,
            calibrated ? 0.99 : 0.9));
    return llvm::cantFail(CompilerTarget::create(
        calibrated ? "updated device" : "original device", std::move(sites),
        CompilerTarget::Connectivity::allToAll(),
        CompilerTarget::NativeOperations::fromOperations({operation}),
        llvm::cantFail(CompilerTarget::DurationUnit::create("ns", timeScale))));
  };
  const auto payload = llvm::cantFail(
      mlir::payloadSpecificationForProgramFormat(QDMI_PROGRAM_FORMAT_QASM3));
  const mlir::TargetEnvironment original(makeTarget(false, {2, 5}, {2, 5}),
                                         payload);
  const mlir::TargetEnvironment calibrated(makeTarget(true, {2, 5}, {2, 5}),
                                           payload);
  EXPECT_FALSE(llvm::errorToBool(
      mlir::validateTargetCompatibility(original, calibrated)));
  for (const auto& target : {
           makeTarget(false, {5, 2}, {2, 5}),
           makeTarget(false, {2, 5}, {5, 2}),
           makeTarget(false, {2, 5}, {2, 5}, 2.),
       }) {
    const mlir::TargetEnvironment changed(target, payload);
    const auto error =
        llvm::toString(mlir::validateTargetCompatibility(original, changed));
    EXPECT_NE(error.find("recompile for this device"), std::string::npos);
  }
  const auto constrained = llvm::cantFail(mlir::PayloadSpecification::create(
      payload.format(),
      {
          {
              .id = "forward-branching",
              .value = 0,
              .constraints =
                  {
                      {.id = "max-control-flow-nesting-depth", .value = 1},
                  },
          },
      }));
  EXPECT_TRUE(llvm::errorToBool(mlir::validateTargetCompatibility(
      original, mlir::TargetEnvironment(original.target(), constrained))));
}

TEST(CompilerQDMIAdapterTest,
     CompilationCreatesNoJobAndSubmissionChecksContract) {
  auto library = std::make_shared<qdmi::DynamicDeviceLibrary>(
      MQT_CORE_MLIR_DDSIM_DEVICE_LIBRARY, "MQT_DDSIM");
  static thread_local decltype(QDMI_device_session_create_device_job)*
      createJob = nullptr;
  static thread_local size_t creations = 0;
  createJob = library->device_session_create_device_job;
  creations = 0;
  library->device_session_create_device_job = [](QDMI_Device_Session session,
                                                 QDMI_Device_Job* job) {
    ++creations;
    return createJob(session, job);
  };
  QDMI_Device_impl_d rawDevice(library);
  const auto device = qdmi::Session::createSessionlessDevice(&rawDevice);
  constexpr auto source =
      "OPENQASM 3.0; include \"stdgates.inc\"; "
      "qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c = measure q;";
  const auto compiled = llvm::cantFail(
      mlir::compileProgram(mlir::OpenQASMProgram(source), device));
  EXPECT_EQ(creations, 0);
  EXPECT_TRUE(
      llvm::errorToBool(mlir::submitProgram(device, compiled, -1).takeError()));
  EXPECT_EQ(creations, 0);
  EXPECT_EQ(compiled.programFormat(), QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE);
  ASSERT_GE(compiled.payload().size(), 4);
  EXPECT_EQ(compiled.payload().substr(0, 2), "BC");
  auto job = llvm::cantFail(mlir::submitProgram(device, compiled, 32));
  EXPECT_EQ(creations, 1);
  EXPECT_TRUE(job.wait());
  EXPECT_EQ(job.getShots().size(), 32);
  const auto otherSession = qdmi::Session::openDevice("mqt.ddsim.default");
  auto otherJob =
      llvm::cantFail(mlir::submitProgram(otherSession, compiled, 16));
  EXPECT_TRUE(otherJob.wait());
  EXPECT_EQ(otherJob.getShots().size(), 16);

  const auto explicitTarget = llvm::cantFail(
      CompilerTarget::create(2, CompilerTarget::Connectivity::allToAll(),
                             CompilerTarget::NativeOperations::unrestricted()));
  const auto payload =
      llvm::cantFail(mlir::payloadSpecificationForProgramFormat(
          QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE));
  const auto otherArtifact = llvm::cantFail(mlir::CompiledProgram::compile(
      mlir::OpenQASMProgram(source),
      mlir::TargetEnvironment(explicitTarget, payload)));
  EXPECT_TRUE(llvm::errorToBool(
      mlir::submitProgram(device, otherArtifact).takeError()));
  EXPECT_EQ(creations, 1);

  const auto restricted = llvm::cantFail(mlir::PayloadSpecification::create(
      payload.format(), {{.id = "forward-branching"}}));
  const auto unsupported = llvm::toString(
      mlir::CompiledProgram::compile(
          mlir::OpenQASMProgram(source),
          mlir::TargetEnvironment(compiled.environment().target(), restricted))
          .takeError());
  EXPECT_NE(unsupported.find("qir.dynamic-result-management"),
            std::string::npos);
}

TEST(CompilerQDMIAdapterTest, CompilesAdaptiveMeasurementControlledLoop) {
  const auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  constexpr auto source =
      "OPENQASM 3.0; include \"stdgates.inc\"; qubit q; bit c; x q; "
      "c = measure q; while (c) { x q; c = measure q; }";
  auto compiled = llvm::cantFail(
      mlir::compileProgram(mlir::OpenQASMProgram(source), device));
  auto job = llvm::cantFail(mlir::submitProgram(device, compiled, 8));
  ASSERT_TRUE(job.wait());
  EXPECT_EQ(job.getCounts().at("0"), 8);
}

TEST(CompilerQDMIAdapterTest, ReleasesOnlyOwnedTensorSlots) {
  const auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  for (const bool dynamicShape : {false, true}) {
    for (const bool releaseTensorFirst : {false, true}) {
      SCOPED_TRACE(dynamicShape);
      SCOPED_TRACE(releaseTensorFirst);
      const std::string release = "qtensor.dealloc %rest : !register\n";
      const auto source = std::string("!register = tensor<") +
                          (dynamicShape ? "?" : "2") + R"mlir(x!qco.qubit>
        module {
          func.func @main() -> i1 attributes {mqt.entry_point} {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %tensor = qtensor.alloc(%c2) : !register
            %index = scf.for %iv = %c0 to %c3 step %c1
                iter_args(%i = %c0) -> index {
              %next = arith.subi %c1, %i : index
              scf.yield %next : index
            }
            %rest, %q = qtensor.extract %tensor[%index] : !register
        )mlir" + (releaseTensorFirst ? release : "") +
                          R"mlir(
            %x = qco.x %q : !qco.qubit -> !qco.qubit
            %out, %bit = qco.measure %x : !qco.qubit
            qco.sink %out : !qco.qubit
        )mlir" + (releaseTensorFirst ? "" : release) +
                          R"mlir(
            return %bit : i1
          }
        })mlir";
      auto program = mlir::QCOProgram::fromMLIRString(source);
      ASSERT_TRUE(program);
      /// Exercise the conversion's ownership contract without QCO cleanup.
      auto qc = std::move(*program).intoQC();
      ASSERT_TRUE(qc);
      auto qir = std::move(*qc).intoQIR(mlir::QIRProfile::Adaptive);
      ASSERT_TRUE(qir);
      auto llvmIR = qir->llvmIR();
      ASSERT_TRUE(llvmIR);
      auto job =
          device.submitJob(*llvmIR, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING, 8);
      ASSERT_TRUE(job.wait());
      EXPECT_EQ(job.getCounts().at("1"), 8);
    }
  }
}

TEST(CompilerQDMIAdapterTest, TransfersQubitOwnershipToAnotherTensor) {
  auto program = mlir::QCOProgram::fromMLIRString(R"mlir(module {
    func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %bits = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
      %tensor = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
      %rest, %q = qtensor.extract %tensor[%c1] : tensor<2x!qco.qubit>
      %adopted = qtensor.from_elements %q : tensor<1x!qco.qubit>
      qtensor.dealloc %rest : tensor<2x!qco.qubit>
      %empty, %taken = qtensor.extract %adopted[%c0] : tensor<1x!qco.qubit>
      %x = qco.x %taken : !qco.qubit -> !qco.qubit
      %out, %bit = qco.measure %x : !qco.qubit
      %complete = qtensor.insert %out into %empty[%c0] : tensor<1x!qco.qubit>
      qtensor.dealloc %complete : tensor<1x!qco.qubit>
      cbit.store %bit, %bits[%c0] : !cbit.reg<1>
      return %bits : !cbit.reg<1>
    }
  })mlir");
  ASSERT_TRUE(program);
  auto qc = std::move(*program).intoQC();
  ASSERT_TRUE(qc);
  auto qir = std::move(*qc).intoQIR(mlir::QIRProfile::Adaptive);
  ASSERT_TRUE(qir);
  auto llvmIR = qir->llvmIR();
  ASSERT_TRUE(llvmIR);
  const auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  auto job =
      device.submitJob(*llvmIR, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING, 8);
  ASSERT_TRUE(job.wait());
  EXPECT_EQ(job.getCounts().at("1"), 8);
}

TEST(CompilerQDMIAdapterTest,
     CompatibilityPreservesOperandsWithinReorderedTuples) {
  const auto makeEnvironment =
      [](std::vector<std::vector<CompilerTarget::SiteId>> tuples) {
        std::vector<CompilerTarget::SiteTuple> siteTuples;
        siteTuples.reserve(tuples.size());
        for (auto& tuple : tuples) {
          siteTuples.push_back(llvm::cantFail(
              CompilerTarget::SiteTuple::create(std::move(tuple))));
        }
        const auto operation =
            llvm::cantFail(CompilerTarget::OperationCapability::create(
                "cx", 2, 0, std::move(siteTuples)));
        const auto target = llvm::cantFail(CompilerTarget::create(
            {
                llvm::cantFail(CompilerTarget::Site::create(0)),
                llvm::cantFail(CompilerTarget::Site::create(1)),
                llvm::cantFail(CompilerTarget::Site::create(2)),
            },
            CompilerTarget::Connectivity::allToAll(),
            CompilerTarget::NativeOperations::fromOperations({operation})));
        return mlir::TargetEnvironment(
            target, llvm::cantFail(mlir::payloadSpecificationForProgramFormat(
                        QDMI_PROGRAM_FORMAT_QASM3)));
      };
  const auto original = makeEnvironment({{0, 1}, {1, 2}, {0, 2}});
  EXPECT_FALSE(llvm::errorToBool(mlir::validateTargetCompatibility(
      original, makeEnvironment({{0, 2}, {1, 2}, {0, 1}}))));
  EXPECT_TRUE(llvm::errorToBool(mlir::validateTargetCompatibility(
      original, makeEnvironment({{0, 2}, {2, 1}, {0, 1}}))));
}

TEST(CompilerQDMIAdapterTest, SubmissionQueriesOnlyTheRequiredMetadata) {
  auto library = std::make_shared<qdmi::DynamicDeviceLibrary>(
      MQT_CORE_MLIR_DDSIM_DEVICE_LIBRARY, "MQT_DDSIM");
  static thread_local decltype(QDMI_device_session_query_device_property)*
      queryDevice = nullptr;
  static thread_local decltype(QDMI_device_session_query_site_property)*
      querySite = nullptr;
  static thread_local size_t siteLists = 0;
  static thread_local size_t calibrationQueries = 0;
  queryDevice = library->device_session_query_device_property;
  querySite = library->device_session_query_site_property;
  library->device_session_query_device_property =
      [](QDMI_Device_Session session, QDMI_Device_Property property,
         size_t size, void* value, size_t* sizeRet) {
        siteLists += property == QDMI_DEVICE_PROPERTY_SITES && value != nullptr;
        return queryDevice(session, property, size, value, sizeRet);
      };
  library->device_session_query_site_property =
      [](QDMI_Device_Session session, QDMI_Site site,
         QDMI_Site_Property property, size_t size, void* value,
         size_t* sizeRet) {
        calibrationQueries += property == QDMI_SITE_PROPERTY_NAME ||
                              property == QDMI_SITE_PROPERTY_T1 ||
                              property == QDMI_SITE_PROPERTY_T2;
        return querySite(session, site, property, size, value, sizeRet);
      };
  QDMI_Device_impl_d rawDevice(library);
  const auto device = qdmi::Session::createSessionlessDevice(&rawDevice);
  constexpr auto source = "OPENQASM 3.1; qubit q; bit c = measure q;";
  const auto compiled = llvm::cantFail(
      mlir::compileProgram(mlir::OpenQASMProgram(source), device));
  siteLists = calibrationQueries = 0;
  auto job = llvm::cantFail(mlir::submitProgram(device, compiled, 4));
  EXPECT_EQ(siteLists, 1);
  EXPECT_EQ(calibrationQueries, 0);
  ASSERT_TRUE(job.wait());
  EXPECT_EQ(job.getCounts().at("0"), 4);

  siteLists = calibrationQueries = 0;
  auto sourceJob = llvm::cantFail(
      mlir::submitProgram(device, mlir::OpenQASMProgram(source), 4));
  EXPECT_EQ(siteLists, 1);
  EXPECT_GT(calibrationQueries, 0);
  ASSERT_TRUE(sourceJob.wait());
  EXPECT_EQ(sourceJob.getCounts().at("0"), 4);
}
