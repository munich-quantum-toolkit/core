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
    QCRYOpTest, QCTest,
    testing::Values(
        QCTestCase{"RY", MQT_NAMED_BUILDER(ry), MQT_NAMED_BUILDER(ry)},
        QCTestCase{"SingleControlledRY", MQT_NAMED_BUILDER(singleControlledRy),
                   MQT_NAMED_BUILDER(singleControlledRy)},
        QCTestCase{"MultipleControlledRY",
                   MQT_NAMED_BUILDER(multipleControlledRy),
                   MQT_NAMED_BUILDER(multipleControlledRy)},
        QCTestCase{"NestedControlledRY", MQT_NAMED_BUILDER(nestedControlledRy),
                   MQT_NAMED_BUILDER(multipleControlledRy)},
        QCTestCase{"TrivialControlledRY",
                   MQT_NAMED_BUILDER(trivialControlledRy),
                   MQT_NAMED_BUILDER(ry)},
        QCTestCase{"InverseRY", MQT_NAMED_BUILDER(inverseRy),
                   MQT_NAMED_BUILDER(ry)},
        QCTestCase{"InverseMultipleControlledRY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledRy),
                   MQT_NAMED_BUILDER(multipleControlledRy)}));
