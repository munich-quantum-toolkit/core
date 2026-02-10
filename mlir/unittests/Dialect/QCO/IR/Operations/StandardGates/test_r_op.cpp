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
    QCOROpTest, QCOTest,
    testing::Values(
        QCOTestCase{"R", r, r},
        QCOTestCase{"SingleControlledR", singleControlledR, singleControlledR},
        QCOTestCase{"MultipleControlledR", multipleControlledR,
                    multipleControlledR},
        QCOTestCase{"NestedControlledR", nestedControlledR,
                    multipleControlledR},
        QCOTestCase{"TrivialControlledR", trivialControlledR, r},
        QCOTestCase{"InverseR", inverseR, r},
        QCOTestCase{"InverseMultipleControlledR", inverseMultipleControlledR,
                    multipleControlledR}),
    printTestName);
