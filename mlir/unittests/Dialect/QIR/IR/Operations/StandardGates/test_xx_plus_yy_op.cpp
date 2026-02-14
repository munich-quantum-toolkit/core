/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir_programs.h"
#include "test_qir_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qir;

INSTANTIATE_TEST_SUITE_P(
    QIRXXPlusYYOpTest, QIRTest,
    testing::Values(QIRTestCase{"XXPlusYY", MQT_NAMED_BUILDER(xxPlusYY),
                                MQT_NAMED_BUILDER(xxPlusYY)},
                    QIRTestCase{"SingleControlledXXPlusYY",
                                MQT_NAMED_BUILDER(singleControlledXxPlusYY),
                                MQT_NAMED_BUILDER(singleControlledXxPlusYY)},
                    QIRTestCase{
                        "MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(multipleControlledXxPlusYY)}));
