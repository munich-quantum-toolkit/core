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
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCRZOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RZ", qc::rz, qco::rz},
                    QCToQCOTestCase{"SingleControlledRZ",
                                    qc::singleControlledRz,
                                    qco::singleControlledRz},
                    QCToQCOTestCase{"MultipleControlledRZ",
                                    qc::multipleControlledRz,
                                    qco::multipleControlledRz}),
    printTestName);
