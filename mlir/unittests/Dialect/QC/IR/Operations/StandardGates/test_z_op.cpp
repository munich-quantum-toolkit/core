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
    QCZOpTest, QCTest,
    testing::Values(
        QCTestCase{"Z", MQT_NAMED_BUILDER(z), MQT_NAMED_BUILDER(z)},
        QCTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ),
                   MQT_NAMED_BUILDER(singleControlledZ)},
        QCTestCase{"MultipleControlledZ",
                   MQT_NAMED_BUILDER(multipleControlledZ),
                   MQT_NAMED_BUILDER(multipleControlledZ)},
        QCTestCase{"NestedControlledZ", MQT_NAMED_BUILDER(nestedControlledZ),
                   MQT_NAMED_BUILDER(multipleControlledZ)},
        QCTestCase{"TrivialControlledZ", MQT_NAMED_BUILDER(trivialControlledZ),
                   MQT_NAMED_BUILDER(z)},
        QCTestCase{"InverseZ", MQT_NAMED_BUILDER(inverseZ),
                   MQT_NAMED_BUILDER(z)},
        QCTestCase{"InverseMultipleControlledZ",
                   MQT_NAMED_BUILDER(inverseMultipleControlledZ),
                   MQT_NAMED_BUILDER(multipleControlledZ)}));
