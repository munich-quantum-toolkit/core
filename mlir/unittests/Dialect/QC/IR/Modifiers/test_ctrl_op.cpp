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
    QCCtrlOpTest, QCTest,
    testing::Values(QCTestCase{"TrivialCtrl", MQT_NAMED_BUILDER(trivialCtrl),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"NestedCtrl", MQT_NAMED_BUILDER(nestedCtrl),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"TripleNestedCtrl",
                               MQT_NAMED_BUILDER(tripleNestedCtrl),
                               MQT_NAMED_BUILDER(tripleControlledRxx)},
                    QCTestCase{"CtrlInvSandwich",
                               MQT_NAMED_BUILDER(ctrlInvSandwich),
                               MQT_NAMED_BUILDER(multipleControlledRxx)},
                    QCTestCase{"DoubleNestedCtrlTwoQubits",
                               MQT_NAMED_BUILDER(doubleNestedCtrlTwoQubits),
                               MQT_NAMED_BUILDER(fourControlledRxx)}));
