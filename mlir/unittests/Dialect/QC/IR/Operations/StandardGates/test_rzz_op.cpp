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
    QCRZZOpTest, QCTest,
    testing::Values(QCTestCase{"RZZ", MQT_NAMED_BUILDER(rzz),
                               MQT_NAMED_BUILDER(rzz)},
                    QCTestCase{"SingleControlledRZZ",
                               MQT_NAMED_BUILDER(singleControlledRzz),
                               MQT_NAMED_BUILDER(singleControlledRzz)},
                    QCTestCase{"MultipleControlledRZZ",
                               MQT_NAMED_BUILDER(multipleControlledRzz),
                               MQT_NAMED_BUILDER(multipleControlledRzz)},
                    QCTestCase{"NestedControlledRZZ",
                               MQT_NAMED_BUILDER(nestedControlledRzz),
                               MQT_NAMED_BUILDER(multipleControlledRzz)},
                    QCTestCase{"TrivialControlledRZZ",
                               MQT_NAMED_BUILDER(trivialControlledRzz),
                               MQT_NAMED_BUILDER(rzz)},
                    QCTestCase{"InverseRZZ", MQT_NAMED_BUILDER(inverseRzz),
                               MQT_NAMED_BUILDER(rzz)},
                    QCTestCase{"InverseMultipleControlledRZZ",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRzz),
                               MQT_NAMED_BUILDER(multipleControlledRzz)}));
