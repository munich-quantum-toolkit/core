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
    QCUOpTest, QCTest,
    testing::Values(
        QCTestCase{"U", u, u},
        QCTestCase{"SingleControlledU", singleControlledU, singleControlledU},
        QCTestCase{"MultipleControlledU", multipleControlledU,
                   multipleControlledU},
        QCTestCase{"NestedControlledU", nestedControlledU, multipleControlledU},
        QCTestCase{"TrivialControlledU", trivialControlledU, u},
        QCTestCase{"InverseU", inverseU, u},
        QCTestCase{"InverseMultipleControlledU", inverseMultipleControlledU,
                   multipleControlledU}),
    printTestName);
