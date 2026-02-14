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
    QCRZOpTest, QCTest,
    testing::Values(
        QCTestCase{"RZ", MQT_NAMED_BUILDER(rz), MQT_NAMED_BUILDER(rz)},
        QCTestCase{"SingleControlledRZ", MQT_NAMED_BUILDER(singleControlledRz),
                   MQT_NAMED_BUILDER(singleControlledRz)},
        QCTestCase{"MultipleControlledRZ",
                   MQT_NAMED_BUILDER(multipleControlledRz),
                   MQT_NAMED_BUILDER(multipleControlledRz)},
        QCTestCase{"NestedControlledRZ", MQT_NAMED_BUILDER(nestedControlledRz),
                   MQT_NAMED_BUILDER(multipleControlledRz)},
        QCTestCase{"TrivialControlledRZ",
                   MQT_NAMED_BUILDER(trivialControlledRz),
                   MQT_NAMED_BUILDER(rz)},
        QCTestCase{"InverseRZ", MQT_NAMED_BUILDER(inverseRz),
                   MQT_NAMED_BUILDER(rz)},
        QCTestCase{"InverseMultipleControlledRZ",
                   MQT_NAMED_BUILDER(inverseMultipleControlledRz),
                   MQT_NAMED_BUILDER(multipleControlledRz)}));
