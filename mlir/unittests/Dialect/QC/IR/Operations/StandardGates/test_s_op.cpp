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
    QCSOpTest, QCTest,
    testing::Values(
        QCTestCase{"S", MQT_NAMED_BUILDER(s), MQT_NAMED_BUILDER(s)},
        QCTestCase{"SingleControlledS", MQT_NAMED_BUILDER(singleControlledS),
                   MQT_NAMED_BUILDER(singleControlledS)},
        QCTestCase{"MultipleControlledS",
                   MQT_NAMED_BUILDER(multipleControlledS),
                   MQT_NAMED_BUILDER(multipleControlledS)},
        QCTestCase{"NestedControlledS", MQT_NAMED_BUILDER(nestedControlledS),
                   MQT_NAMED_BUILDER(multipleControlledS)},
        QCTestCase{"TrivialControlledS", MQT_NAMED_BUILDER(trivialControlledS),
                   MQT_NAMED_BUILDER(s)},
        QCTestCase{"InverseS", MQT_NAMED_BUILDER(inverseS),
                   MQT_NAMED_BUILDER(sdg)},
        QCTestCase{"InverseMultipleControlledS",
                   MQT_NAMED_BUILDER(inverseMultipleControlledS),
                   MQT_NAMED_BUILDER(multipleControlledSdg)}));
