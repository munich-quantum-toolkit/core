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
    QCOSXOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SX", qco::sx, qc::sx},
        QCOToQCTestCase{"SingleControlledSX", qco::singleControlledSx,
                        qc::singleControlledSx},
        QCOToQCTestCase{"MultipleControlledSX", qco::multipleControlledSx,
                        qc::multipleControlledSx},
        QCOToQCTestCase{"NestedControlledSX", qco::nestedControlledSx,
                        qc::multipleControlledSx},
        QCOToQCTestCase{"TrivialControlledSX", qco::trivialControlledSx,
                        qc::sx},
        QCOToQCTestCase{"InverseSX", qco::inverseSx, qc::sxdg},
        QCOToQCTestCase{"InverseMultipleControlledSX",
                        qco::inverseMultipleControlledSx,
                        qc::multipleControlledSxdg}),
    printTestName);
