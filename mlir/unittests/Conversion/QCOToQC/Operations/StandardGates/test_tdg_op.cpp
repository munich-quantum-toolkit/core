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
    QCOTdgOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"Tdg", qco::tdg, qc::tdg},
        QCOToQCTestCase{"SingleControlledTdg", qco::singleControlledTdg,
                        qc::singleControlledTdg},
        QCOToQCTestCase{"MultipleControlledTdg", qco::multipleControlledTdg,
                        qc::multipleControlledTdg},
        QCOToQCTestCase{"NestedControlledTdg", qco::nestedControlledTdg,
                        qc::multipleControlledTdg},
        QCOToQCTestCase{"TrivialControlledTdg", qco::trivialControlledTdg,
                        qc::tdg},
        QCOToQCTestCase{"InverseTdg", qco::inverseTdg, qc::t_},
        QCOToQCTestCase{"InverseMultipleControlledTdg",
                        qco::inverseMultipleControlledTdg,
                        qc::multipleControlledT}),
    printTestName);
