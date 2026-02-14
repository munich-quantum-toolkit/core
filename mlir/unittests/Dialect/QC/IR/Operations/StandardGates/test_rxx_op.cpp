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
    QCRXXOpTest, QCTest,
    testing::Values(QCTestCase{"RXX", MQT_NAMED_BUILDER(rxx),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"SingleControlledRXX",
                               MQT_NAMED_BUILDER(singleControlledRxx),
                               MQT_NAMED_BUILDER(singleControlledRxx)},
                    QCTestCase{"MultipleControlledRXX",
                               MQT_NAMED_BUILDER(multipleControlledRxx),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"NestedControlledRXX",
                               MQT_NAMED_BUILDER(nestedControlledRxx),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"TrivialControlledRXX",
                               MQT_NAMED_BUILDER(trivialControlledRxx),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"InverseRXX", MQT_NAMED_BUILDER(inverseRxx),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"InverseMultipleControlledRXX",
                               MQT_NAMED_BUILDER(inverseMultipleControlledRxx),
                               MQT_NAMED_BUILDER(multipleControlledRxx)}));
