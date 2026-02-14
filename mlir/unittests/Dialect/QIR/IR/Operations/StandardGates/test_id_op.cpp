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
    QIRIDOpTest, QIRTest,
    testing::Values(QIRTestCase{"Identity", MQT_NAMED_BUILDER(identity),
                                MQT_NAMED_BUILDER(identity)},
                    QIRTestCase{"SingleControlledIdentity",
                                MQT_NAMED_BUILDER(singleControlledIdentity),
                                MQT_NAMED_BUILDER(singleControlledIdentity)},
                    QIRTestCase{
                        "MultipleControlledIdentity",
                        MQT_NAMED_BUILDER(multipleControlledIdentity),
                        MQT_NAMED_BUILDER(multipleControlledIdentity)}));
