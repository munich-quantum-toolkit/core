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
    QCU2OpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"U2", qc::u2, qco::u2},
                    QCToQCOTestCase{"SingleControlledU2",
                                    qc::singleControlledU2,
                                    qco::singleControlledU2},
                    QCToQCOTestCase{"MultipleControlledU2",
                                    qc::multipleControlledU2,
                                    qco::multipleControlledU2}),
    printTestName);
