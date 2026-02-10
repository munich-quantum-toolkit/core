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
    QCRZOpTest, QCTest,
    testing::Values(QCTestCase{"RZ", rz, rz},
                    QCTestCase{"SingleControlledRZ", singleControlledRz,
                               singleControlledRz},
                    QCTestCase{"MultipleControlledRZ", multipleControlledRz,
                               multipleControlledRz},
                    QCTestCase{"NestedControlledRZ", nestedControlledRz,
                               multipleControlledRz},
                    QCTestCase{"TrivialControlledRZ", trivialControlledRz, rz},
                    QCTestCase{"InverseRZ", inverseRz, rz},
                    QCTestCase{"InverseMultipleControlledRZ",
                               inverseMultipleControlledRz,
                               multipleControlledRz}),
    printTestName);
