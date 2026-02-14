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
    QCRYYOpTest, QCTest,
    testing::Values(QCTestCase{"RYY", MQT_NAMED_BUILDER(ryy),
                               MQT_NAMED_BUILDER(ryy)},
                    QCTestCase{"SingleControlledRYY",
                               MQT_NAMED_BUILDER(singleControlledRyy),
                               MQT_NAMED_BUILDER(singleControlledRyy)},
                    QCTestCase{"MultipleControlledRYY",
                               MQT_NAMED_BUILDER(multipleControlledRyy),
                               MQT_NAMED_BUILDER(multipleControlledRyy)},
                    QCTestCase{"NestedControlledRYY",
                               MQT_NAMED_BUILDER(nestedControlledRyy),
                               MQT_NAMED_BUILDER(multipleControlledRyy)},
                    QCTestCase{"TrivialControlledRYY",
                               MQT_NAMED_BUILDER(trivialControlledRyy),
                               MQT_NAMED_BUILDER(ryy)},
                    QCTestCase{"InverseRYY", MQT_NAMED_BUILDER(inverseRyy),
                               MQT_NAMED_BUILDER(ryy)},
                    QCTestCase{"InverseMultipleControlledRYY",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRyy),
                               MQT_NAMED_BUILDER(multipleControlledRyy)}));
