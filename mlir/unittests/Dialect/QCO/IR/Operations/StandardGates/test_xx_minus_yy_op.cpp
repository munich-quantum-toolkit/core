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
    QCOXXMinusYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"XXMinusYY", xxMinusYY, xxMinusYY},
        QCOTestCase{"SingleControlledXXMinusYY", singleControlledXxMinusYY,
                    singleControlledXxMinusYY},
        QCOTestCase{"MultipleControlledXXMinusYY", multipleControlledXxMinusYY,
                    multipleControlledXxMinusYY},
        QCOTestCase{"NestedControlledXXMinusYY", nestedControlledXxMinusYY,
                    multipleControlledXxMinusYY},
        QCOTestCase{"TrivialControlledXXMinusYY", trivialControlledXxMinusYY,
                    xxMinusYY},
        QCOTestCase{"InverseXXMinusYY", inverseXxMinusYY, xxMinusYY},
        QCOTestCase{"InverseMultipleControlledXXMinusYY",
                    inverseMultipleControlledXxMinusYY,
                    multipleControlledXxMinusYY}),
    printTestName);
