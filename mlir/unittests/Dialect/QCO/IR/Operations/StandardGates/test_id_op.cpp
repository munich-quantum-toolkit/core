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
    testing::Values(QCOTestCase{"Identity", identity, identity},
                    QCOTestCase{"SingleControlledIdentity",
                                singleControlledIdentity, identity},
                    QCOTestCase{"MultipleControlledIdentity",
                                multipleControlledIdentity, identity},
                    QCOTestCase{"NestedControlledIdentity",
                                nestedControlledIdentity, identity},
                    QCOTestCase{"TrivialControlledIdentity",
                                trivialControlledIdentity, identity},
                    QCOTestCase{"InverseIdentity", inverseIdentity, identity},
                    QCOTestCase{"InverseMultipleControlledIdentity",
                                inverseMultipleControlledIdentity,
                                inverseMultipleControlledIdentity}),
    printTestName);
