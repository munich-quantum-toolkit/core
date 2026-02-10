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

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, QCOTest,
    testing::Values(QCOTestCase{"RY", ry, ry},
                    QCOTestCase{"SingleControlledRY", singleControlledRy,
                                singleControlledRy},
                    QCOTestCase{"MultipleControlledRY", multipleControlledRy,
                                multipleControlledRy},
                    QCOTestCase{"NestedControlledRY", nestedControlledRy,
                                multipleControlledRy},
                    QCOTestCase{"TrivialControlledRY", trivialControlledRy, ry},
                    QCOTestCase{"InverseRY", inverseRy, ry},
                    QCOTestCase{"InverseMultipleControlledRY",
                                inverseMultipleControlledRy,
                                multipleControlledRy}),
    printTestName);
