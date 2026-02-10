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
    QCU2OpTest, QCTest,
    testing::Values(QCTestCase{"U2", u2, u2},
                    QCTestCase{"SingleControlledU2", singleControlledU2,
                               singleControlledU2},
                    QCTestCase{"MultipleControlledU2", multipleControlledU2,
                               multipleControlledU2},
                    QCTestCase{"NestedControlledU2", nestedControlledU2,
                               multipleControlledU2},
                    QCTestCase{"TrivialControlledU2", trivialControlledU2, u2},
                    QCTestCase{"InverseU2", inverseU2, u2},
                    QCTestCase{"InverseMultipleControlledU2",
                               inverseMultipleControlledU2,
                               multipleControlledU2}),
    printTestName);
