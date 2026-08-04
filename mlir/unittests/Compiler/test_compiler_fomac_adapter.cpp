/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "fomac/FoMaC.hpp"
#include "mlir/Compiler/FoMaCAdapter.h"
#include "mlir/Compiler/Target.h"
#include "qdmi/driver/Driver.hpp"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>

#include <stdexcept>
#include <string>

using mlir::CompilerTarget;

[[nodiscard]] static const CompilerTarget::Operation&
findOperation(const CompilerTarget& target, const llvm::StringRef name) {
  const auto* const found =
      llvm::find_if(target.operations(),
                    [&](const auto& op) { return op.canonicalName() == name; });
  if (found == target.operations().end()) {
    throw std::out_of_range("Target operation not found");
  }
  return *found;
}

TEST(CompilerFoMaCAdapterTest, SnapshotsIQMCalibrationAndLifetime) {
  const auto target = [] {
    const auto device = fomac::Session::openDevice("mqt.sc.iqm.garnet");
    return mlir::compilerTargetFromDevice(device);
  }();

  ASSERT_TRUE(target.name());
  EXPECT_EQ(*target.name(), "IQM Garnet");
  EXPECT_EQ(target.numQubits(), 20);
  EXPECT_TRUE(target.hasExplicitTopology());
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

  EXPECT_TRUE(target.supportsOperation("r", 1, 2));
  EXPECT_TRUE(target.supportsOperation("cz", 2, 0));
  EXPECT_TRUE(target.supportsOperation("measure", 1, 0));
  EXPECT_FALSE(target.supportsOperation("rx", 1, 1));
  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->singleQubit,
            CompilerTarget::SingleQubitBasis::R);
  EXPECT_EQ(target.synthesisBasis()->entangler, CompilerTarget::GateKind::CZ);
}

TEST(CompilerFoMaCAdapterTest, PreservesMissingTopologyAsAllToAll) {
  const auto device = fomac::Session::openDevice("mqt.ddsim.default");
  const auto target = mlir::compilerTargetFromDevice(device);

  EXPECT_EQ(target.numQubits(), 65535);
  EXPECT_FALSE(target.hasExplicitTopology());
  EXPECT_TRUE(target.areAdjacent(0, target.numQubits() - 1));
  EXPECT_TRUE(target.supportsOperation("h", 1, 0));
  EXPECT_TRUE(target.supportsOperation("cx", 2, 0));
  EXPECT_TRUE(target.supportsOperation("measure", 1, 0));
}

TEST(CompilerFoMaCAdapterTest, RejectsNonhomogeneousOperationSupport) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration =
      qdmi::FileDeviceConfiguration{MQT_CORE_MLIR_HETEROGENEOUS_SC_CONFIG};
  const auto device = fomac::Session::openDevice("mqt.sc.default", overrides);
  try {
    const auto target = mlir::compilerTargetFromDevice(device);
    FAIL() << "Expected a homogeneous-operation diagnostic, got "
           << target.operations().size() << " target operations";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("homogeneous"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("every topology edge"),
              std::string::npos);
  }
}

TEST(CompilerFoMaCAdapterTest, RejectsNeutralAtomZoneModels) {
  const auto device = fomac::Session::openDevice("mqt.na.default");
  try {
    const auto target = mlir::compilerTargetFromDevice(device);
    FAIL() << "Expected a neutral-atom diagnostic, got " << target.numQubits()
           << " target sites";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("only circuit-model devices"),
              std::string::npos);
    EXPECT_NE(std::string(error.what()).find("zone"), std::string::npos);
  }
}
