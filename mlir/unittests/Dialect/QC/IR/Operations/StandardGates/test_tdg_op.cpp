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

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCTdgOpTest, QCTest,
    testing::Values(
        QCTestCase{"Tdg", tdg, tdg},
        QCTestCase{"SingleControlledTdg", singleControlledTdg,
                   singleControlledTdg},
        QCTestCase{"MultipleControlledTdg", multipleControlledTdg,
                   multipleControlledTdg},
        QCTestCase{"NestedControlledTdg", nestedControlledTdg,
                   multipleControlledTdg},
        QCTestCase{"TrivialControlledTdg", trivialControlledTdg, tdg},
        QCTestCase{"InverseTdg", inverseTdg, t_},
        QCTestCase{"InverseMultipleControlledTdg", inverseMultipleControlledTdg,
                   multipleControlledT}),
    printTestName);
