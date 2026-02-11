/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qco_programs.h"
#include "test_qco_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOU2OpTest, QCOTest,
    testing::Values(QCOTestCase{"U2", u2, u2},
                    QCOTestCase{"SingleControlledU2", singleControlledU2,
                                singleControlledU2},
                    QCOTestCase{"MultipleControlledU2", multipleControlledU2,
                                multipleControlledU2},
                    QCOTestCase{"NestedControlledU2", nestedControlledU2,
                                multipleControlledU2},
                    QCOTestCase{"TrivialControlledU2", trivialControlledU2, u2},
                    QCOTestCase{"InverseU2", inverseU2, u2},
                    QCOTestCase{"InverseMultipleControlledU2",
                                inverseMultipleControlledU2,
                                multipleControlledU2}),
    printTestName);
