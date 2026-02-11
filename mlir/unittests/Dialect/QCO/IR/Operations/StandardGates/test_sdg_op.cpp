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
    QCOSdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Sdg", sdg, sdg},
        QCOTestCase{"SingleControlledSdg", singleControlledSdg,
                    singleControlledSdg},
        QCOTestCase{"MultipleControlledSdg", multipleControlledSdg,
                    multipleControlledSdg},
        QCOTestCase{"NestedControlledSdg", nestedControlledSdg,
                    multipleControlledSdg},
        QCOTestCase{"TrivialControlledSdg", trivialControlledSdg, sdg},
        QCOTestCase{"InverseSdg", inverseSdg, s},
        QCOTestCase{"InverseMultipleControlledSdg",
                    inverseMultipleControlledSdg, multipleControlledS}),
    printTestName);
