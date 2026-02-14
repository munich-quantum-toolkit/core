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
    QIRSdgOpTest, QIRTest,
    testing::Values(QIRTestCase{"Sdg", MQT_NAMED_BUILDER(sdg),
                                MQT_NAMED_BUILDER(sdg)},
                    QIRTestCase{"SingleControlledSdg",
                                MQT_NAMED_BUILDER(singleControlledSdg),
                                MQT_NAMED_BUILDER(singleControlledSdg)},
                    QIRTestCase{"MultipleControlledSdg",
                                MQT_NAMED_BUILDER(multipleControlledSdg),
                                MQT_NAMED_BUILDER(multipleControlledSdg)}));
