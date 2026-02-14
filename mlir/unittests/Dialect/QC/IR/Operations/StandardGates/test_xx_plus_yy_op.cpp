/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qc_programs.h"
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCXXPlusYYOpTest, QCTest,
    testing::Values(
        QCTestCase{"XXPlusYY", MQT_NAMED_BUILDER(xxPlusYY),
                   MQT_NAMED_BUILDER(xxPlusYY)},
        QCTestCase{"SingleControlledXXPlusYY",
                   MQT_NAMED_BUILDER(singleControlledXxPlusYY),
                   MQT_NAMED_BUILDER(singleControlledXxPlusYY)},
        QCTestCase{"MultipleControlledXXPlusYY",
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCTestCase{"NestedControlledXXPlusYY",
                   MQT_NAMED_BUILDER(nestedControlledXxPlusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCTestCase{"TrivialControlledXXPlusYY",
                   MQT_NAMED_BUILDER(trivialControlledXxPlusYY),
                   MQT_NAMED_BUILDER(xxPlusYY)},
        QCTestCase{"InverseXXPlusYY", MQT_NAMED_BUILDER(inverseXxPlusYY),
                   MQT_NAMED_BUILDER(xxPlusYY)},
        QCTestCase{"InverseMultipleControlledXXPlusYY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledXxPlusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxPlusYY)}));
