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
    QCDCXOpTest, QCTest,
    testing::Values(QCTestCase{"DCX", MQT_NAMED_BUILDER(dcx),
                               MQT_NAMED_BUILDER(dcx)},
                    QCTestCase{"SingleControlledDCX",
                               MQT_NAMED_BUILDER(singleControlledDcx),
                               MQT_NAMED_BUILDER(singleControlledDcx)},
                    QCTestCase{"MultipleControlledDCX",
                               MQT_NAMED_BUILDER(multipleControlledDcx),
                               MQT_NAMED_BUILDER(multipleControlledDcx)},
                    QCTestCase{"NestedControlledDCX",
                               MQT_NAMED_BUILDER(nestedControlledDcx),
                               MQT_NAMED_BUILDER(multipleControlledDcx)},
                    QCTestCase{"TrivialControlledDCX",
                               MQT_NAMED_BUILDER(trivialControlledDcx),
                               MQT_NAMED_BUILDER(dcx)},
                    QCTestCase{"InverseDCX", MQT_NAMED_BUILDER(inverseDcx),
                               MQT_NAMED_BUILDER(dcx)},
                    QCTestCase{"InverseMultipleControlledDCX",
                               MQT_NAMED_BUILDER(inverseMultipleControlledDcx),
                               MQT_NAMED_BUILDER(multipleControlledDcx)}));
