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
#include "test_qc_ir.h"

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCSWAPOpTest, QCTest,
    testing::Values(
        QCTestCase{"SWAP", swap, swap},
        QCTestCase{"SingleControlledSWAP", singleControlledSwap,
                   singleControlledSwap},
        QCTestCase{"MultipleControlledSWAP", multipleControlledSwap,
                   multipleControlledSwap},
        QCTestCase{"NestedControlledSWAP", nestedControlledSwap,
                   multipleControlledSwap},
        QCTestCase{"TrivialControlledSWAP", trivialControlledSwap, swap},
        QCTestCase{"InverseSWAP", inverseSwap, swap},
        QCTestCase{"InverseMultipleControlledSWAP",
                   inverseMultipleControlledSwap, multipleControlledSwap}),
    printTestName);
