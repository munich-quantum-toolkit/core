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
    QCOSWAPOpTest, QCOTest,
    testing::Values(QCOTestCase{"SWAP", swap, swap},
                    QCOTestCase{"SingleControlledSWAP", singleControlledSwap,
                                singleControlledSwap},
                    QCOTestCase{"MultipleControlledSWAP",
                                multipleControlledSwap, multipleControlledSwap},
                    QCOTestCase{"NestedControlledSWAP", nestedControlledSwap,
                                multipleControlledSwap},
                    QCOTestCase{"TrivialControlledSWAP", trivialControlledSwap,
                                swap},
                    QCOTestCase{"InverseSWAP", inverseSwap, swap},
                    QCOTestCase{"InverseMultipleControlledSWAP",
                                inverseMultipleControlledSwap,
                                inverseMultipleControlledSwap}),
    printTestName);
