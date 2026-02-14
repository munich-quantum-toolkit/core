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
    QIRSWAPOpTest, QIRTest,
    testing::Values(QIRTestCase{"SWAP", MQT_NAMED_BUILDER(swap),
                                MQT_NAMED_BUILDER(swap)},
                    QIRTestCase{"SingleControlledSWAP",
                                MQT_NAMED_BUILDER(singleControlledSwap),
                                MQT_NAMED_BUILDER(singleControlledSwap)},
                    QIRTestCase{"MultipleControlledSWAP",
                                MQT_NAMED_BUILDER(multipleControlledSwap),
                                MQT_NAMED_BUILDER(multipleControlledSwap)}));
