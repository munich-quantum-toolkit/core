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
    QCORZXOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"RZX", qco::rzx, qc::rzx},
        QCOToQCTestCase{"SingleControlledRZX", qco::singleControlledRzx,
                        qc::singleControlledRzx},
        QCOToQCTestCase{"MultipleControlledRZX", qco::multipleControlledRzx,
                        qc::multipleControlledRzx},
        QCOToQCTestCase{"NestedControlledRZX", qco::nestedControlledRzx,
                        qc::multipleControlledRzx},
        QCOToQCTestCase{"TrivialControlledRZX", qco::trivialControlledRzx,
                        qc::rzx},
        QCOToQCTestCase{"InverseRZX", qco::inverseRzx, qc::rzx},
        QCOToQCTestCase{"InverseMultipleControlledRZX",
                        qco::inverseMultipleControlledRzx,
                        qc::multipleControlledRzx}),
    printTestName);
