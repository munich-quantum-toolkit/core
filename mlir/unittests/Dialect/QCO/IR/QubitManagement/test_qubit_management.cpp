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
        QCOTestCase{"AllocQubit", allocQubit, emptyQCO},
        QCOTestCase{"AllocQubitRegister", allocQubitRegister, emptyQCO},
        QCOTestCase{"AllocMultipleQubitRegisters", allocMultipleQubitRegisters,
                    emptyQCO},
        QCOTestCase{"AllocLargeRegister", allocLargeRegister, emptyQCO},
        QCOTestCase{"StaticQubits", staticQubits, emptyQCO},
        QCOTestCase{"AllocDeallocPair", allocDeallocPair, emptyQCO}),
    printTestName);
