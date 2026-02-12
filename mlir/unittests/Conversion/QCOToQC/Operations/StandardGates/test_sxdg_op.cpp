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
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SXdg", qco::sxdg, qc::sxdg},
        QCOToQCTestCase{"SingleControlledSXdg", qco::singleControlledSxdg,
                        qc::singleControlledSxdg},
        QCOToQCTestCase{"MultipleControlledSXdg", qco::multipleControlledSxdg,
                        qc::multipleControlledSxdg},
        QCOToQCTestCase{"NestedControlledSXdg", qco::nestedControlledSxdg,
                        qc::multipleControlledSxdg},
        QCOToQCTestCase{"TrivialControlledSXdg", qco::trivialControlledSxdg,
                        qc::sxdg},
        QCOToQCTestCase{"InverseSXdg", qco::inverseSxdg, qc::sx},
        QCOToQCTestCase{"InverseMultipleControlledSXdg",
                        qco::inverseMultipleControlledSxdg,
                        qc::multipleControlledSx}),
    printTestName);
