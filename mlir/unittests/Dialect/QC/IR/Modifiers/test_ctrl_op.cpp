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
    QCCtrlOpTest, QCTest,
    testing::Values(
        QCTestCase{"TrivialCtrl", trivialCtrl, rxx},
        QCTestCase{"NestedCtrl", nestedCtrl, multipleControlledRxx},
        QCTestCase{"TripleNestedCtrl", tripleNestedCtrl, tripleControlledRxx},
        QCTestCase{"CtrlInvSandwich", ctrlInvSandwich, multipleControlledRxx}),
    printTestName);
