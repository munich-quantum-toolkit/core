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
    QCHOpTest, QCTest,
    testing::Values(
        QCTestCase{"H", MQT_NAMED_BUILDER(h), MQT_NAMED_BUILDER(h)},
        QCTestCase{"SingleControlledH", MQT_NAMED_BUILDER(singleControlledH),
                   MQT_NAMED_BUILDER(singleControlledH)},
        QCTestCase{"MultipleControlledH",
                   MQT_NAMED_BUILDER(multipleControlledH),
                   MQT_NAMED_BUILDER(multipleControlledH)},
        QCTestCase{"NestedControlledH", MQT_NAMED_BUILDER(nestedControlledH),
                   MQT_NAMED_BUILDER(multipleControlledH)},
        QCTestCase{"TrivialControlledH", MQT_NAMED_BUILDER(trivialControlledH),
                   MQT_NAMED_BUILDER(h)},
        QCTestCase{"InverseH", MQT_NAMED_BUILDER(inverseH),
                   MQT_NAMED_BUILDER(h)},
        QCTestCase{"InverseMultipleControlledH",
                   MQT_NAMED_BUILDER(inverseMultipleControlledH),
                   MQT_NAMED_BUILDER(multipleControlledH)}));
