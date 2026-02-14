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
    QCU2OpTest, QCTest,
    testing::Values(
        QCTestCase{"U2", MQT_NAMED_BUILDER(u2), MQT_NAMED_BUILDER(u2)},
        QCTestCase{"SingleControlledU2", MQT_NAMED_BUILDER(singleControlledU2),
                   MQT_NAMED_BUILDER(singleControlledU2)},
        QCTestCase{"MultipleControlledU2",
                   MQT_NAMED_BUILDER(multipleControlledU2),
                   MQT_NAMED_BUILDER(multipleControlledU2)},
        QCTestCase{"NestedControlledU2", MQT_NAMED_BUILDER(nestedControlledU2),
                   MQT_NAMED_BUILDER(multipleControlledU2)},
        QCTestCase{"TrivialControlledU2",
                   MQT_NAMED_BUILDER(trivialControlledU2),
                   MQT_NAMED_BUILDER(u2)},
        QCTestCase{"InverseU2", MQT_NAMED_BUILDER(inverseU2),
                   MQT_NAMED_BUILDER(u2)},
        QCTestCase{"InverseMultipleControlledU2",
                   MQT_NAMED_BUILDER(inverseMultipleControlledU2),
                   MQT_NAMED_BUILDER(multipleControlledU2)}));
