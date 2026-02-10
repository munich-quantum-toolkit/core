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
    QCORZOpTest, QCOTest,
    testing::Values(QCOTestCase{"RZ", rz, rz},
                    QCOTestCase{"SingleControlledRZ", singleControlledRz,
                                singleControlledRz},
                    QCOTestCase{"MultipleControlledRZ", multipleControlledRz,
                                multipleControlledRz},
                    QCOTestCase{"NestedControlledRZ", nestedControlledRz,
                                multipleControlledRz},
                    QCOTestCase{"TrivialControlledRZ", trivialControlledRz, rz},
                    QCOTestCase{"InverseRZ", inverseRz, rz},
                    QCOTestCase{"InverseMultipleControlledRZ",
                                inverseMultipleControlledRz,
                                inverseMultipleControlledRz}),
    printTestName);
