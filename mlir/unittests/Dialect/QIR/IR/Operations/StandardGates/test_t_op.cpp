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
    QIRTOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"T", MQT_NAMED_BUILDER(t_), MQT_NAMED_BUILDER(t_)},
        QIRTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT),
                    MQT_NAMED_BUILDER(singleControlledT)},
        QIRTestCase{"MultipleControlledT",
                    MQT_NAMED_BUILDER(multipleControlledT),
                    MQT_NAMED_BUILDER(multipleControlledT)}));
