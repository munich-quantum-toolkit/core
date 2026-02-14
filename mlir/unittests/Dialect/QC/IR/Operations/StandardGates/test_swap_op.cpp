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
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCSWAPOpTest, QCTest,
    testing::Values(QCTestCase{"SWAP", MQT_NAMED_BUILDER(swap),
                               MQT_NAMED_BUILDER(swap)},
                    QCTestCase{"SingleControlledSWAP",
                               MQT_NAMED_BUILDER(singleControlledSwap),
                               MQT_NAMED_BUILDER(singleControlledSwap)},
                    QCTestCase{"MultipleControlledSWAP",
                               MQT_NAMED_BUILDER(multipleControlledSwap),
                               MQT_NAMED_BUILDER(multipleControlledSwap)},
                    QCTestCase{"NestedControlledSWAP",
                               MQT_NAMED_BUILDER(nestedControlledSwap),
                               MQT_NAMED_BUILDER(multipleControlledSwap)},
                    QCTestCase{"TrivialControlledSWAP",
                               MQT_NAMED_BUILDER(trivialControlledSwap),
                               MQT_NAMED_BUILDER(swap)},
                    QCTestCase{"InverseSWAP", MQT_NAMED_BUILDER(inverseSwap),
                               MQT_NAMED_BUILDER(swap)},
                    QCTestCase{"InverseMultipleControlledSWAP",
                               MQT_NAMED_BUILDER(inverseMultipleControlledSwap),
                               MQT_NAMED_BUILDER(multipleControlledSwap)}));
