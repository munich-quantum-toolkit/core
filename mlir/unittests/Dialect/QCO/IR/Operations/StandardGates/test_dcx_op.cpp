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
    QCODCXOpTest, QCOTest,
    testing::Values(QCOTestCase{"DCX", dcx, dcx},
                    QCOTestCase{"SingleControlledDCX", singleControlledDcx,
                                singleControlledDcx},
                    QCOTestCase{"MultipleControlledDCX", multipleControlledDcx,
                                multipleControlledDcx},
                    QCOTestCase{"NestedControlledDCX", nestedControlledDcx,
                                multipleControlledDcx},
                    QCOTestCase{"TrivialControlledDCX", trivialControlledDcx,
                                dcx},
                    QCOTestCase{"InverseDCX", inverseDcx, dcx},
                    QCOTestCase{"InverseMultipleControlledDCX",
                                inverseMultipleControlledDcx,
                                inverseMultipleControlledDcx}),
    printTestName);
