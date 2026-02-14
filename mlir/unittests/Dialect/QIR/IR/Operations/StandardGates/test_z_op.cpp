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
    QIRZOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"Z", MQT_NAMED_BUILDER(z), MQT_NAMED_BUILDER(z)},
        QIRTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ),
                    MQT_NAMED_BUILDER(singleControlledZ)},
        QIRTestCase{"MultipleControlledZ",
                    MQT_NAMED_BUILDER(multipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)}));
