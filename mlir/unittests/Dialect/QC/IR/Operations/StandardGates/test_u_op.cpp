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
    QCUOpTest, QCTest,
    testing::Values(
        QCTestCase{"U", MQT_NAMED_BUILDER(u), MQT_NAMED_BUILDER(u)},
        QCTestCase{"SingleControlledU", MQT_NAMED_BUILDER(singleControlledU),
                   MQT_NAMED_BUILDER(singleControlledU)},
        QCTestCase{"MultipleControlledU",
                   MQT_NAMED_BUILDER(multipleControlledU),
                   MQT_NAMED_BUILDER(multipleControlledU)},
        QCTestCase{"NestedControlledU", MQT_NAMED_BUILDER(nestedControlledU),
                   MQT_NAMED_BUILDER(multipleControlledU)},
        QCTestCase{"TrivialControlledU", MQT_NAMED_BUILDER(trivialControlledU),
                   MQT_NAMED_BUILDER(u)},
        QCTestCase{"InverseU", MQT_NAMED_BUILDER(inverseU),
                   MQT_NAMED_BUILDER(u)},
        QCTestCase{"InverseMultipleControlledU",
                   MQT_NAMED_BUILDER(inverseMultipleControlledU),
                   MQT_NAMED_BUILDER(multipleControlledU)}));
