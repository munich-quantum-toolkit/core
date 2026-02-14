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
    QCSXOpTest, QCTest,
    testing::Values(
        QCTestCase{"SX", MQT_NAMED_BUILDER(sx), MQT_NAMED_BUILDER(sx)},
        QCTestCase{"SingleControlledSX", MQT_NAMED_BUILDER(singleControlledSx),
                   MQT_NAMED_BUILDER(singleControlledSx)},
        QCTestCase{"MultipleControlledSX",
                   MQT_NAMED_BUILDER(multipleControlledSx),
                   MQT_NAMED_BUILDER(multipleControlledSx)},
        QCTestCase{"NestedControlledSX", MQT_NAMED_BUILDER(nestedControlledSx),
                   MQT_NAMED_BUILDER(multipleControlledSx)},
        QCTestCase{"TrivialControlledSX",
                   MQT_NAMED_BUILDER(trivialControlledSx),
                   MQT_NAMED_BUILDER(sx)},
        QCTestCase{"InverseSX", MQT_NAMED_BUILDER(inverseSx),
                   MQT_NAMED_BUILDER(sxdg)},
        QCTestCase{"InverseMultipleControlledSX",
                   MQT_NAMED_BUILDER(inverseMultipleControlledSx),
                   MQT_NAMED_BUILDER(multipleControlledSxdg)}));
