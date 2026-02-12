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
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"RX", qco::rx, qc::rx},
        QCOToQCTestCase{"SingleControlledRX", qco::singleControlledRx,
                        qc::singleControlledRx},
        QCOToQCTestCase{"MultipleControlledRX", qco::multipleControlledRx,
                        qc::multipleControlledRx},
        QCOToQCTestCase{"NestedControlledRX", qco::nestedControlledRx,
                        qc::multipleControlledRx},
        QCOToQCTestCase{"TrivialControlledRX", qco::trivialControlledRx,
                        qc::rx},
        QCOToQCTestCase{"InverseRX", qco::inverseRx, qc::rx},
        QCOToQCTestCase{"InverseMultipleControlledRX",
                        qco::inverseMultipleControlledRx,
                        qc::multipleControlledRx}),
    printTestName);
