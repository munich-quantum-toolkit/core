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
    QIRRZZOpTest, QIRTest,
    testing::Values(QIRTestCase{"RZZ", MQT_NAMED_BUILDER(rzz),
                                MQT_NAMED_BUILDER(rzz)},
                    QIRTestCase{"SingleControlledRZZ",
                                MQT_NAMED_BUILDER(singleControlledRzz),
                                MQT_NAMED_BUILDER(singleControlledRzz)},
                    QIRTestCase{"MultipleControlledRZZ",
                                MQT_NAMED_BUILDER(multipleControlledRzz),
                                MQT_NAMED_BUILDER(multipleControlledRzz)}));
