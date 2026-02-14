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
    QIRQubitManagementTest, QIRTest,
    testing::Values(QIRTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubit),
                                MQT_NAMED_BUILDER(allocQubit)},
                    QIRTestCase{"AllocQubitRegister",
                                MQT_NAMED_BUILDER(allocQubitRegister),
                                MQT_NAMED_BUILDER(allocQubitRegister)},
                    QIRTestCase{"AllocMultipleQubitRegisters",
                                MQT_NAMED_BUILDER(allocMultipleQubitRegisters),
                                MQT_NAMED_BUILDER(allocMultipleQubitRegisters)},
                    QIRTestCase{"AllocLargeRegister",
                                MQT_NAMED_BUILDER(allocLargeRegister),
                                MQT_NAMED_BUILDER(allocLargeRegister)},
                    QIRTestCase{"StaticQubits", MQT_NAMED_BUILDER(staticQubits),
                                MQT_NAMED_BUILDER(staticQubits)}));
