/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "qc_programs.h"
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCTOpTest, QCTest,
    testing::Values(
        QCTestCase{"T", MQT_NAMED_BUILDER(t_), MQT_NAMED_BUILDER(t_)},
        QCTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT),
                   MQT_NAMED_BUILDER(singleControlledT)},
        QCTestCase{"MultipleControlledT",
                   MQT_NAMED_BUILDER(multipleControlledT),
                   MQT_NAMED_BUILDER(multipleControlledT)},
        QCTestCase{"NestedControlledT", MQT_NAMED_BUILDER(nestedControlledT),
                   MQT_NAMED_BUILDER(multipleControlledT)},
        QCTestCase{"TrivialControlledT", MQT_NAMED_BUILDER(trivialControlledT),
                   MQT_NAMED_BUILDER(t_)},
        QCTestCase{"InverseT", MQT_NAMED_BUILDER(inverseT),
                   MQT_NAMED_BUILDER(tdg)},
        QCTestCase{"InverseMultipleControlledT",
                   MQT_NAMED_BUILDER(inverseMultipleControlledT),
                   MQT_NAMED_BUILDER(multipleControlledTdg)}));
