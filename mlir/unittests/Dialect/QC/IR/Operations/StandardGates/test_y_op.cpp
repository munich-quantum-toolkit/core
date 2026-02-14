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
    QCYOpTest, QCTest,
    testing::Values(
        QCTestCase{"Y", MQT_NAMED_BUILDER(y), MQT_NAMED_BUILDER(y)},
        QCTestCase{"SingleControlledY", MQT_NAMED_BUILDER(singleControlledY),
                   MQT_NAMED_BUILDER(singleControlledY)},
        QCTestCase{"MultipleControlledY",
                   MQT_NAMED_BUILDER(multipleControlledY),
                   MQT_NAMED_BUILDER(multipleControlledY)},
        QCTestCase{"NestedControlledY", MQT_NAMED_BUILDER(nestedControlledY),
                   MQT_NAMED_BUILDER(multipleControlledY)},
        QCTestCase{"TrivialControlledY", MQT_NAMED_BUILDER(trivialControlledY),
                   MQT_NAMED_BUILDER(y)},
        QCTestCase{"InverseY", MQT_NAMED_BUILDER(inverseY),
                   MQT_NAMED_BUILDER(y)},
        QCTestCase{"InverseMultipleControlledY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledY),
                   MQT_NAMED_BUILDER(multipleControlledY)}));
