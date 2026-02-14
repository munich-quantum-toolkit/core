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
    QIRSXdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg),
                                MQT_NAMED_BUILDER(sxdg)},
                    QIRTestCase{"SingleControlledSXdg",
                                MQT_NAMED_BUILDER(singleControlledSxdg),
                                MQT_NAMED_BUILDER(singleControlledSxdg)},
                    QIRTestCase{"MultipleControlledSXdg",
                                MQT_NAMED_BUILDER(multipleControlledSxdg),
                                MQT_NAMED_BUILDER(multipleControlledSxdg)}));
