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
    QCXXMinusYYOpTest, QCTest,
    testing::Values(
        QCTestCase{"XXMinusYY", MQT_NAMED_BUILDER(xxMinusYY),
                   MQT_NAMED_BUILDER(xxMinusYY)},
        QCTestCase{"SingleControlledXXMinusYY",
                   MQT_NAMED_BUILDER(singleControlledXxMinusYY),
                   MQT_NAMED_BUILDER(singleControlledXxMinusYY)},
        QCTestCase{"MultipleControlledXXMinusYY",
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCTestCase{"NestedControlledXXMinusYY",
                   MQT_NAMED_BUILDER(nestedControlledXxMinusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCTestCase{"TrivialControlledXXMinusYY",
                   MQT_NAMED_BUILDER(trivialControlledXxMinusYY),
                   MQT_NAMED_BUILDER(xxMinusYY)},
        QCTestCase{"InverseXXMinusYY", MQT_NAMED_BUILDER(inverseXxMinusYY),
                   MQT_NAMED_BUILDER(xxMinusYY)},
        QCTestCase{"InverseMultipleControlledXXMinusYY",
                   MQT_NAMED_BUILDER(inverseMultipleControlledXxMinusYY),
                   MQT_NAMED_BUILDER(multipleControlledXxMinusYY)}));
