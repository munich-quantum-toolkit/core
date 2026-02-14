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
#include "qc_programs.h"
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCQubitManagementTest, QCTest,
    testing::Values(
        QCTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubit),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocQubitRegister", MQT_NAMED_BUILDER(allocQubitRegister),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocMultipleQubitRegisters",
                   MQT_NAMED_BUILDER(allocMultipleQubitRegisters),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocLargeRegister", MQT_NAMED_BUILDER(allocLargeRegister),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"StaticQubits", MQT_NAMED_BUILDER(staticQubits),
                   MQT_NAMED_BUILDER(emptyQC)},
        QCTestCase{"AllocDeallocPair", MQT_NAMED_BUILDER(allocDeallocPair),
                   MQT_NAMED_BUILDER(emptyQC)}));
