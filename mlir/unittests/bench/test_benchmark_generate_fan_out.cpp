/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/FanOut.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/DDAdapter.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <utility>

namespace mqt::bench {

TEST(GenerateProgramTest, FanOutMatchesSequentialPhasesAndCleansAncillas) {
  for (const size_t controls : {1U, 3U, 5U, 257U}) {
    SCOPED_TRACE(controls);
    auto program = test::generateQCO(
        llvm::cantFail(FanOut::create({.controls = controls})));
    ASSERT_TRUE(mlir::succeeded(program));
    dd::Package package(0);
    auto state = mlir::qco::simulateStatevector(
        mlir::mqt::getEntryPoint(program->module()), package);
    ASSERT_TRUE(mlir::succeeded(state));
    ASSERT_EQ(package.qubits(), 2U * controls);

    // The sequential reference never touches the clean ancillas.
    auto expected = dd::makeZeroState(2U * controls, package);
    const auto target = static_cast<dd::Qubit>(controls);
    expected = package.applyOperation(
        mlir::qco::makeGateDD(package, mlir::qco::HOp::getUnitaryMatrix(),
                              {target}),
        expected);
    for (dd::Qubit control = 0; control < target; ++control) {
      expected = package.applyOperation(
          mlir::qco::makeGateDD(package, mlir::qco::HOp::getUnitaryMatrix(),
                                {control}),
          expected);
      expected = package.applyOperation(
          mlir::qco::makeGateDD(package, mlir::qco::SOp::getUnitaryMatrix(),
                                {target}, {{control}}),
          expected);
    }
    expected = package.applyOperation(
        mlir::qco::makeGateDD(package, mlir::qco::HOp::getUnitaryMatrix(),
                              {target}),
        expected);
    EXPECT_NEAR(package.fidelity(*state, expected), 1., 1e-10);
    package.decRef(expected);
    package.decRef(*state);
  }
}

TEST(GenerateProgramTest, SamplesFanOutAgainstReference) {
  test::expectSamplingMatchesReference(
      llvm::cantFail(FanOut::create({.controls = 3})));
}

TEST(GenerateProgramTest, SerializesTheLargestFanOutWithCompactLoops) {
  auto program = generate(llvm::cantFail(
      FanOut::create({.controls = FanOutOptions::MAX_CONTROLS})));
  ASSERT_TRUE(mlir::succeeded(program));
  // Generation must not expand a gate list with one operation per qubit.
  EXPECT_LT(test::countOperations(program->module()), 150U);
  test::expectJeffRoundTrip(std::move(*program));
}

} // namespace mqt::bench
