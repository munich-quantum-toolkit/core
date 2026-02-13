/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir_programs.h"
#include "test_qir_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qir;

INSTANTIATE_TEST_SUITE_P(
    QIRXXMinusYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"XXMinusYY", xxMinusYY, xxMinusYY},
                    QIRTestCase{"SingleControlledXXMinusYY",
                                singleControlledXxMinusYY,
                                singleControlledXxMinusYY},
                    QIRTestCase{"MultipleControlledXXMinusYY",
                                multipleControlledXxMinusYY,
                                multipleControlledXxMinusYY}),
    printTestName);
