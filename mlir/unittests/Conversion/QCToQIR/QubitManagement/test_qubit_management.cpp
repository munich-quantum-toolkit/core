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
#include "qir_programs.h"
#include "test_qc_to_qir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;
using namespace mlir::qir;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRQubitManagementTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"AllocQubit", MQT_NAMED_BUILDER(qc::allocQubit),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"AllocQubitRegister",
                        MQT_NAMED_BUILDER(qc::allocQubitRegister),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"AllocMultipleQubitRegisters",
                        MQT_NAMED_BUILDER(qc::allocMultipleQubitRegisters),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"AllocLargeRegister",
                        MQT_NAMED_BUILDER(qc::allocLargeRegister),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"StaticQubits", MQT_NAMED_BUILDER(qc::staticQubits),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"AllocDeallocPair",
                        MQT_NAMED_BUILDER(qc::allocDeallocPair),
                        MQT_NAMED_BUILDER(emptyQIR)}));
