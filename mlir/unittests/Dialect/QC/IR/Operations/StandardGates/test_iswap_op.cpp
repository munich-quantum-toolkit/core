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
    QCiSWAPOpTest, QCTest,
    testing::Values(
        QCTestCase{"iSWAP", MQT_NAMED_BUILDER(iswap), MQT_NAMED_BUILDER(iswap)},
        QCTestCase{"SingleControllediSWAP",
                   MQT_NAMED_BUILDER(singleControlledIswap),
                   MQT_NAMED_BUILDER(singleControlledIswap)},
        QCTestCase{"MultipleControllediSWAP",
                   MQT_NAMED_BUILDER(multipleControlledIswap),
                   MQT_NAMED_BUILDER(multipleControlledIswap)},
        QCTestCase{"NestedControllediSWAP",
                   MQT_NAMED_BUILDER(nestedControlledIswap),
                   MQT_NAMED_BUILDER(multipleControlledIswap)},
        QCTestCase{"TrivialControllediSWAP",
                   MQT_NAMED_BUILDER(trivialControlledIswap),
                   MQT_NAMED_BUILDER(iswap)},
        QCTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(inverseIswap),
                   MQT_NAMED_BUILDER(inverseIswap)},
        QCTestCase{"InverseMultipleControllediSWAP",
                   MQT_NAMED_BUILDER(inverseMultipleControlledIswap),
                   MQT_NAMED_BUILDER(inverseMultipleControlledIswap)}));
