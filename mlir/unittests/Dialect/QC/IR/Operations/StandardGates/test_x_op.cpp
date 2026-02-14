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
    QCXOpTest, QCTest,
    testing::Values(
        QCTestCase{"X", MQT_NAMED_BUILDER(x), MQT_NAMED_BUILDER(x)},
        QCTestCase{"SingleControlledX", MQT_NAMED_BUILDER(singleControlledX),
                   MQT_NAMED_BUILDER(singleControlledX)},
        QCTestCase{"MultipleControlledX",
                   MQT_NAMED_BUILDER(multipleControlledX),
                   MQT_NAMED_BUILDER(multipleControlledX)},
        QCTestCase{"NestedControlledX", MQT_NAMED_BUILDER(nestedControlledX),
                   MQT_NAMED_BUILDER(multipleControlledX)},
        QCTestCase{"TrivialControlledX", MQT_NAMED_BUILDER(trivialControlledX),
                   MQT_NAMED_BUILDER(x)},
        QCTestCase{"InverseX", MQT_NAMED_BUILDER(inverseX),
                   MQT_NAMED_BUILDER(x)},
        QCTestCase{"InverseMultipleControlledX",
                   MQT_NAMED_BUILDER(inverseMultipleControlledX),
                   MQT_NAMED_BUILDER(multipleControlledX)}));
