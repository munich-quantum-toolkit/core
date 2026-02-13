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
#include "qir_programs.h"
#include "test_qc_to_qir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRSdgOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"Sdg", qc::sdg, qir::sdg},
                    QCToQIRTestCase{"SingleControlledSdg",
                                    qc::singleControlledSdg,
                                    qir::singleControlledSdg},
                    QCToQIRTestCase{"MultipleControlledSdg",
                                    qc::multipleControlledSdg,
                                    qir::multipleControlledSdg}),
    printTestName);
