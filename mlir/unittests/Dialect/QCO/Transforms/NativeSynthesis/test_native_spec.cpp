/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>

using namespace mlir::qco::decomposition;
using namespace mlir::qco::native_synth;

TEST(NativeSpecTest, ResolveIbmBasicCx) {
  const auto spec = resolveNativeGatesSpec("x,sx,rz,cx");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->allowedGates.contains(NativeGateKind::Cx));
  EXPECT_TRUE(spec->allowedGates.contains(NativeGateKind::X));
  EXPECT_FALSE(spec->allowRzz);
}

TEST(NativeSpecTest, ResolveRejectsUnknownToken) {
  EXPECT_FALSE(resolveNativeGatesSpec("x,sx,rz,not-a-gate").has_value());
}

TEST(NativeSpecTest, ResolveEmptyOrWhitespaceOnlyReturnsNullopt) {
  EXPECT_FALSE(resolveNativeGatesSpec("").has_value());
  EXPECT_FALSE(resolveNativeGatesSpec("   \t  ").has_value());
  EXPECT_FALSE(resolveNativeGatesSpec(",,,").has_value());
}

TEST(NativeSpecTest, PhaseAliasPMatchesRzInIbmStyleMenu) {
  const auto pMenu = resolveNativeGatesSpec("x,sx,p,cx");
  const auto rzMenu = resolveNativeGatesSpec("x,sx,rz,cx");
  ASSERT_TRUE(pMenu);
  ASSERT_TRUE(rzMenu);
  EXPECT_EQ(pMenu->allowedGates, rzMenu->allowedGates);
}

TEST(NativeSpecTest, EmitterEulerBasisForAxisPair) {
  EXPECT_EQ(emitterEulerBasis(SingleQubitEmitterSpec{
                .mode = SingleQubitMode::AxisPair, .axisPair = AxisPair::RxRz}),
            EulerBasis::XZX);
  EXPECT_EQ(emitterEulerBasis(SingleQubitEmitterSpec{
                .mode = SingleQubitMode::AxisPair, .axisPair = AxisPair::RxRy}),
            EulerBasis::XYX);
  EXPECT_EQ(emitterEulerBasis(SingleQubitEmitterSpec{
                .mode = SingleQubitMode::AxisPair, .axisPair = AxisPair::RyRz}),
            EulerBasis::ZYZ);
}

TEST(NativeSpecTest, EmitterEulerBasisForPrimaryModes) {
  EXPECT_EQ(
      emitterEulerBasis(SingleQubitEmitterSpec{.mode = SingleQubitMode::U3}),
      EulerBasis::U);
  EXPECT_EQ(
      emitterEulerBasis(SingleQubitEmitterSpec{.mode = SingleQubitMode::ZSXX}),
      EulerBasis::ZSXX);
  EXPECT_EQ(
      emitterEulerBasis(SingleQubitEmitterSpec{.mode = SingleQubitMode::R}),
      EulerBasis::R);
}

TEST(NativeSpecTest, RzzSetsAllowRzzFlag) {
  const auto spec = resolveNativeGatesSpec("u,cx,rzz");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->allowRzz);
  EXPECT_TRUE(spec->allowedGates.contains(NativeGateKind::Rzz));
}
