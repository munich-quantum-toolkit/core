/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "qir_programs.h"
#include "test_qir_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qir;

INSTANTIATE_TEST_SUITE_P(
    QIRRZXOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZX", MQT_NAMED_BUILDER(rzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QIRTestCase{"SingleControlledRZX",
                                MQT_NAMED_BUILDER(singleControlledRzx),
                                MQT_NAMED_BUILDER(singleControlledRzx)},
                    QIRTestCase{"MultipleControlledRZX",
                                MQT_NAMED_BUILDER(multipleControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)}));
