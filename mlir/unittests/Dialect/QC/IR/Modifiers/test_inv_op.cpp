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
    QCInvOpTest, QCTest,
    testing::Values(QCTestCase{"NestedInv", MQT_NAMED_BUILDER(nestedInv),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"TripleNestedInv",
                               MQT_NAMED_BUILDER(tripleNestedInv),
                               MQT_NAMED_BUILDER(rxx)},
                    QCTestCase{"InvControlSandwich",
                               MQT_NAMED_BUILDER(invCtrlSandwich),
                               MQT_NAMED_BUILDER(singleControlledRxx)}));
