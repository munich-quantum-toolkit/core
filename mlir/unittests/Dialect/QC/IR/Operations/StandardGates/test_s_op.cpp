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
    QCSOpTest, QCTest,
    testing::Values(
        QCTestCase{"S", s, s},
        QCTestCase{"SingleControlledS", singleControlledS, singleControlledS},
        QCTestCase{"MultipleControlledS", multipleControlledS,
                   multipleControlledS},
        QCTestCase{"NestedControlledS", nestedControlledS, multipleControlledS},
        QCTestCase{"TrivialControlledS", trivialControlledS, s},
        QCTestCase{"InverseS", inverseS, sdg},
        QCTestCase{"InverseMultipleControlledS", inverseMultipleControlledS,
                   multipleControlledSdg}),
    printTestName);
