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
    QCSXdgOpTest, QCTest,
    testing::Values(
        QCTestCase{"SXdg", sxdg, sxdg},
        QCTestCase{"SingleControlledSXdg", singleControlledSxdg,
                   singleControlledSxdg},
        QCTestCase{"MultipleControlledSXdg", multipleControlledSxdg,
                   multipleControlledSxdg},
        QCTestCase{"NestedControlledSXdg", nestedControlledSxdg,
                   multipleControlledSxdg},
        QCTestCase{"TrivialControlledSXdg", trivialControlledSxdg, sxdg},
        QCTestCase{"InverseSXdg", inverseSxdg, sx},
        QCTestCase{"InverseMultipleControlledSXdg",
                   inverseMultipleControlledSxdg, multipleControlledSx}),
    printTestName);
