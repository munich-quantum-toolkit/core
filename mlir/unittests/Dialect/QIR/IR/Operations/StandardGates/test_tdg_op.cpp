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
    QIRTdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"Tdg", MQT_NAMED_BUILDER(tdg),
                                MQT_NAMED_BUILDER(tdg)},
                    QIRTestCase{"SingleControlledTdg",
                                MQT_NAMED_BUILDER(singleControlledTdg),
                                MQT_NAMED_BUILDER(singleControlledTdg)},
                    QIRTestCase{"MultipleControlledTdg",
                                MQT_NAMED_BUILDER(multipleControlledTdg),
                                MQT_NAMED_BUILDER(multipleControlledTdg)}));
