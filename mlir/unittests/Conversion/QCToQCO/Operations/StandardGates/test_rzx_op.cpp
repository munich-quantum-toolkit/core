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
    QCRZXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RZX", qc::rzx, qco::rzx},
        QCToQCOTestCase{"SingleControlledRZX", qc::singleControlledRzx,
                        qco::singleControlledRzx},
        QCToQCOTestCase{"MultipleControlledRZX", qc::multipleControlledRzx,
                        qco::multipleControlledRzx},
        QCToQCOTestCase{"NestedControlledRZX", qc::nestedControlledRzx,
                        qco::multipleControlledRzx},
        QCToQCOTestCase{"TrivialControlledRZX", qc::trivialControlledRzx,
                        qco::rzx},
        QCToQCOTestCase{"InverseRZX", qc::inverseRzx, qco::rzx},
        QCToQCOTestCase{"InverseMultipleControlledRZX",
                        qc::inverseMultipleControlledRzx,
                        qco::multipleControlledRzx}),
    printTestName);
