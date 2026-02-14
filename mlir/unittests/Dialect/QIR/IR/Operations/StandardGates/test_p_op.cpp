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
    QIRPOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"P", MQT_NAMED_BUILDER(p), MQT_NAMED_BUILDER(p)},
        QIRTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP),
                    MQT_NAMED_BUILDER(singleControlledP)},
        QIRTestCase{"MultipleControlledP",
                    MQT_NAMED_BUILDER(multipleControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)}));
