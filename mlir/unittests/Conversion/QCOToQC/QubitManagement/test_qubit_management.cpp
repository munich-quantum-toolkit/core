/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qc_programs.h"
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOQubitManagementTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"AllocQubit", qco::allocQubit, emptyQC},
        QCOToQCTestCase{"AllocQubitRegister", qco::allocQubitRegister, emptyQC},
        QCOToQCTestCase{"AllocMultipleQubitRegisters",
                        qco::allocMultipleQubitRegisters, emptyQC},
        QCOToQCTestCase{"AllocLargeRegister", qco::allocLargeRegister, emptyQC},
        QCOToQCTestCase{"StaticQubits", qco::staticQubits, emptyQC},
        QCOToQCTestCase{"AllocDeallocPair", qco::allocDeallocPair, emptyQC}),
    printTestName);
