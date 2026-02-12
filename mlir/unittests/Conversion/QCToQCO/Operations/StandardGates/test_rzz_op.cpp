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
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCRZZOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RZZ", qc::rzz, qco::rzz},
        QCToQCOTestCase{"SingleControlledRZZ", qc::singleControlledRzz,
                        qco::singleControlledRzz},
        QCToQCOTestCase{"MultipleControlledRZZ", qc::multipleControlledRzz,
                        qco::multipleControlledRzz},
        QCToQCOTestCase{"NestedControlledRZZ", qc::nestedControlledRzz,
                        qco::multipleControlledRzz},
        QCToQCOTestCase{"TrivialControlledRZZ", qc::trivialControlledRzz,
                        qco::rzz},
        QCToQCOTestCase{"InverseRZZ", qc::inverseRzz, qco::rzz},
        QCToQCOTestCase{"InverseMultipleControlledRZZ",
                        qc::inverseMultipleControlledRzz,
                        qco::multipleControlledRzz}),
    printTestName);
