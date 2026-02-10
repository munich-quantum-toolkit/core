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
    QCOIDOpTest, QCOTest,
    testing::Values(QCOTestCase{"Identity", identity, emptyQCO},
                    QCOTestCase{"SingleControlledIdentity",
                                singleControlledIdentity, emptyQCO},
                    QCOTestCase{"MultipleControlledIdentity",
                                multipleControlledIdentity, emptyQCO},
                    QCOTestCase{"NestedControlledIdentity",
                                nestedControlledIdentity, emptyQCO},
                    QCOTestCase{"TrivialControlledIdentity",
                                trivialControlledIdentity, emptyQCO},
                    QCOTestCase{"InverseIdentity", inverseIdentity, emptyQCO},
                    QCOTestCase{"InverseMultipleControlledIdentity",
                                inverseMultipleControlledIdentity, emptyQCO}),
    printTestName);
