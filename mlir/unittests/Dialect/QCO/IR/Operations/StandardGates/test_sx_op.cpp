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
    QCOSXOpTest, QCOTest,
    testing::Values(QCOTestCase{"SX", sx, sx},
                    QCOTestCase{"SingleControlledSX", singleControlledSx,
                                singleControlledSx},
                    QCOTestCase{"MultipleControlledSX", multipleControlledSx,
                                multipleControlledSx},
                    QCOTestCase{"NestedControlledSX", nestedControlledSx,
                                multipleControlledSx},
                    QCOTestCase{"TrivialControlledSX", trivialControlledSx, sx},
                    QCOTestCase{"InverseSX", inverseSx, sxdg},
                    QCOTestCase{"InverseMultipleControlledSX",
                                inverseMultipleControlledSx,
                                inverseMultipleControlledSx}),
    printTestName);
