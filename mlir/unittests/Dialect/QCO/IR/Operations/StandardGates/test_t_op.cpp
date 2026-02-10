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
    QCOTOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"T", t_, t_},
        QCOTestCase{"SingleControlledT", singleControlledT, singleControlledT},
        QCOTestCase{"MultipleControlledT", multipleControlledT,
                    multipleControlledT},
        QCOTestCase{"NestedControlledT", nestedControlledT,
                    multipleControlledT},
        QCOTestCase{"TrivialControlledT", trivialControlledT, t_},
        QCOTestCase{"InverseT", inverseT, tdg},
        QCOTestCase{"InverseMultipleControlledT", inverseMultipleControlledT,
                    inverseMultipleControlledT}),
    printTestName);
