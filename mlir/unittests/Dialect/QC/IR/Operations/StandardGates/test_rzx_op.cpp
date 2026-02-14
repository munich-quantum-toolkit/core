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
    QCRZXOpTest, QCTest,
    testing::Values(QCTestCase{"RZX", MQT_NAMED_BUILDER(rzx),
                               MQT_NAMED_BUILDER(rzx)},
                    QCTestCase{"SingleControlledRZX",
                               MQT_NAMED_BUILDER(singleControlledRzx),
                               MQT_NAMED_BUILDER(singleControlledRzx)},
                    QCTestCase{"MultipleControlledRZX",
                               MQT_NAMED_BUILDER(multipleControlledRzx),
                               MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCTestCase{"NestedControlledRZX",
                               MQT_NAMED_BUILDER(nestedControlledRzx),
                               MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCTestCase{"TrivialControlledRZX",
                               MQT_NAMED_BUILDER(trivialControlledRzx),
                               MQT_NAMED_BUILDER(rzx)},
                    QCTestCase{"InverseRZX", MQT_NAMED_BUILDER(inverseRzx),
                               MQT_NAMED_BUILDER(rzx)},
                    QCTestCase{"InverseMultipleControlledRZX",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRzx),
                               MQT_NAMED_BUILDER(multipleControlledRzx)}));
