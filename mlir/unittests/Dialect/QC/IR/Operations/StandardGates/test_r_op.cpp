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
    QCROpTest, QCTest,
    testing::Values(
        QCTestCase{"R", MQT_NAMED_BUILDER(r), MQT_NAMED_BUILDER(r)},
        QCTestCase{"SingleControlledR", MQT_NAMED_BUILDER(singleControlledR),
                   MQT_NAMED_BUILDER(singleControlledR)},
        QCTestCase{"MultipleControlledR",
                   MQT_NAMED_BUILDER(multipleControlledR),
                   MQT_NAMED_BUILDER(multipleControlledR)},
        QCTestCase{"NestedControlledR", MQT_NAMED_BUILDER(nestedControlledR),
                   MQT_NAMED_BUILDER(multipleControlledR)},
        QCTestCase{"TrivialControlledR", MQT_NAMED_BUILDER(trivialControlledR),
                   MQT_NAMED_BUILDER(r)},
        QCTestCase{"InverseR", MQT_NAMED_BUILDER(inverseR),
                   MQT_NAMED_BUILDER(r)},
        QCTestCase{"InverseMultipleControlledR",
                   MQT_NAMED_BUILDER(inverseMultipleControlledR),
                   MQT_NAMED_BUILDER(multipleControlledR)}));
