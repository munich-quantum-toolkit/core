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
    QCPOpTest, QCTest,
    testing::Values(
        QCTestCase{"P", MQT_NAMED_BUILDER(p), MQT_NAMED_BUILDER(p)},
        QCTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP),
                   MQT_NAMED_BUILDER(singleControlledP)},
        QCTestCase{"MultipleControlledP",
                   MQT_NAMED_BUILDER(multipleControlledP),
                   MQT_NAMED_BUILDER(multipleControlledP)},
        QCTestCase{"NestedControlledP", MQT_NAMED_BUILDER(nestedControlledP),
                   MQT_NAMED_BUILDER(multipleControlledP)},
        QCTestCase{"TrivialControlledP", MQT_NAMED_BUILDER(trivialControlledP),
                   MQT_NAMED_BUILDER(p)},
        QCTestCase{"InverseP", MQT_NAMED_BUILDER(inverseP),
                   MQT_NAMED_BUILDER(p)},
        QCTestCase{"InverseMultipleControlledP",
                   MQT_NAMED_BUILDER(inverseMultipleControlledP),
                   MQT_NAMED_BUILDER(multipleControlledP)}));
