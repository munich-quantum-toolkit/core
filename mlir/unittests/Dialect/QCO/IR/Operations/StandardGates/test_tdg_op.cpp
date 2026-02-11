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

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Tdg", tdg, tdg},
        QCOTestCase{"SingleControlledTdg", singleControlledTdg,
                    singleControlledTdg},
        QCOTestCase{"MultipleControlledTdg", multipleControlledTdg,
                    multipleControlledTdg},
        QCOTestCase{"NestedControlledTdg", nestedControlledTdg,
                    multipleControlledTdg},
        QCOTestCase{"TrivialControlledTdg", trivialControlledTdg, tdg},
        QCOTestCase{"InverseTdg", inverseTdg, t_},
        QCOTestCase{"InverseMultipleControlledTdg",
                    inverseMultipleControlledTdg, multipleControlledT}),
    printTestName);
