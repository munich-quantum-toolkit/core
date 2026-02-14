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
    QCRXOpTest, QCTest,
    testing::Values(
        QCTestCase{"RX", MQT_NAMED_BUILDER(rx), MQT_NAMED_BUILDER(rx)},
        QCTestCase{"SingleControlledRX", MQT_NAMED_BUILDER(singleControlledRx),
                   MQT_NAMED_BUILDER(singleControlledRx)},
        QCTestCase{"MultipleControlledRX",
                   MQT_NAMED_BUILDER(multipleControlledRx),
                   MQT_NAMED_BUILDER(multipleControlledRx)},
        QCTestCase{"NestedControlledRX", MQT_NAMED_BUILDER(nestedControlledRx),
                   MQT_NAMED_BUILDER(multipleControlledRx)},
        QCTestCase{"TrivialControlledRX",
                   MQT_NAMED_BUILDER(trivialControlledRx),
                   MQT_NAMED_BUILDER(rx)},
        QCTestCase{"InverseRX", MQT_NAMED_BUILDER(inverseRx),
                   MQT_NAMED_BUILDER(rx)},
        QCTestCase{"InverseMultipleControlledRX",
                   MQT_NAMED_BUILDER(inverseMultipleControlledRx),
                   MQT_NAMED_BUILDER(multipleControlledRx)}));
