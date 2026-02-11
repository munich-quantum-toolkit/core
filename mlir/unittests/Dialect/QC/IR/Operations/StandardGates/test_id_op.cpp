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
    QCIDOpTest, QCTest,
    testing::Values(QCTestCase{"Identity", identity, identity},
                    QCTestCase{"SingleControlledIdentity",
                               singleControlledIdentity, identity},
                    QCTestCase{"MultipleControlledIdentity",
                               multipleControlledIdentity, identity},
                    QCTestCase{"NestedControlledIdentity",
                               nestedControlledIdentity, identity},
                    QCTestCase{"TrivialControlledIdentity",
                               trivialControlledIdentity, identity},
                    QCTestCase{"InverseIdentity", inverseIdentity, identity},
                    QCTestCase{"InverseMultipleControlledIdentity",
                               inverseMultipleControlledIdentity, identity}),
    printTestName);
