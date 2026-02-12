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
    QCCtrlOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"TrivialCtrl", qc::trivialCtrl, qco::rxx},
                    QCToQCOTestCase{"NestedCtrl", qc::nestedCtrl,
                                    qco::multipleControlledRxx},
                    QCToQCOTestCase{"TripleNestedCtrl", qc::tripleNestedCtrl,
                                    qco::tripleControlledRxx},
                    QCToQCOTestCase{"CtrlInvSandwich", qc::ctrlInvSandwich,
                                    qco::multipleControlledRxx}),
    printTestName);

/// TODO: Add a test that has nested controls where each of the control
///   modifiers has at least two qubits.
