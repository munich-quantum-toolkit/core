/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/QDMIAdapter.h"
#include "mlir/Compiler/Target.h"
#include "qdmi/Client.hpp"
#include "qdmi/driver/Driver.hpp"

#include <gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <cassert>
#include <initializer_list>
#include <string>
#include <utility>

using mlir::CompilerTarget;

[[nodiscard]] static const CompilerTarget::Operation&
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
            CompilerTarget::Operation::Arity::Kind::Fixed);
  EXPECT_EQ(gphase.arity().value(), 0);
  for (const auto [name, minimum] :
       std::initializer_list<std::pair<llvm::StringRef, size_t>>{{"id", 1},
                                                                 {"h", 1},
                                                                 {"rx", 1},
                                                                 {"swap", 2},
                                                                 {"rxx", 2},
                                                                 {"rccx", 3}}) {
    const auto& operation = findOperation(target, name);
    EXPECT_EQ(operation.arity().kind(),
              CompilerTarget::Operation::Arity::Kind::Variadic)
        << name.str();
    EXPECT_EQ(operation.arity().value(), minimum) << name.str();
    EXPECT_TRUE(
        target.supportsOperation(name, minimum, operation.numParameters()))
        << name.str();
    EXPECT_TRUE(
        target.supportsOperation(name, minimum + 4, operation.numParameters()))
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
}

TEST(CompilerQDMIAdapterTest, PreservesOneWayDirectionalOperationSupport) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::FileDeviceConfiguration{
      MQT_CORE_MLIR_DIRECTIONAL_ONE_WAY_SC_CONFIG};
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  ASSERT_EQ(target.couplings().size(), 1);
  const auto& cx = findOperation(target, "cx");
  EXPECT_TRUE(cx.hasExplicitSiteTuples());
  ASSERT_EQ(cx.siteTuples().size(), 1);
  EXPECT_EQ(cx.siteTuples()[0].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{0, 1}));
  EXPECT_FALSE(cx.siteTuples()[0].duration());
  EXPECT_FALSE(cx.siteTuples()[0].fidelity());
  EXPECT_TRUE(target.supports(CompilerTarget::GateKind::CX, {0, 1}));
  EXPECT_FALSE(target.supports(CompilerTarget::GateKind::CX, {1, 0}));
}

TEST(CompilerQDMIAdapterTest,
     PreservesDirectionalCalibrationWhenBothOrientationsExist) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::FileDeviceConfiguration{
      MQT_CORE_MLIR_DIRECTIONAL_TWO_WAY_SC_CONFIG};
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  ASSERT_EQ(target.couplings().size(), 1);
  const auto& cx = findOperation(target, "cx");
  EXPECT_TRUE(cx.hasExplicitSiteTuples());
  ASSERT_EQ(cx.siteTuples().size(), 2);
  EXPECT_EQ(cx.siteTuples()[0].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{0, 1}));
  EXPECT_DOUBLE_EQ(*cx.siteTuples()[0].fidelity(), 0.91);
  EXPECT_EQ(cx.siteTuples()[1].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{1, 0}));
  EXPECT_DOUBLE_EQ(*cx.siteTuples()[1].fidelity(), 0.92);
}
