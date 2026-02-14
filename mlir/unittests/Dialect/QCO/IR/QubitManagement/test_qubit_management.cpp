/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qco_programs.h"
#include "test_qco_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOQubitManagementTest, QCOTest,
    testing::Values(
        QCOTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubit),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocQubitRegister", MQT_NAMED_BUILDER(allocQubitRegister),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocMultipleQubitRegisters",
                    MQT_NAMED_BUILDER(allocMultipleQubitRegisters),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocLargeRegister", MQT_NAMED_BUILDER(allocLargeRegister),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"StaticQubits", MQT_NAMED_BUILDER(staticQubits),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"AllocDeallocPair", MQT_NAMED_BUILDER(allocDeallocPair),
                    MQT_NAMED_BUILDER(emptyQCO)}));
