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
    QCGPhaseOpTest, QCTest,
    testing::Values(
        QCTestCase{"GlobalPhase", MQT_NAMED_BUILDER(globalPhase),
                   MQT_NAMED_BUILDER(globalPhase)},
        QCTestCase{"SingleControlledGlobalPhase",
                   MQT_NAMED_BUILDER(singleControlledGlobalPhase),
                   MQT_NAMED_BUILDER(p)},
        QCTestCase{"MultipleControlledGlobalPhase",
                   MQT_NAMED_BUILDER(multipleControlledGlobalPhase),
                   MQT_NAMED_BUILDER(multipleControlledP)},
        QCTestCase{"NestedControlledGlobalPhase",
                   MQT_NAMED_BUILDER(nestedControlledGlobalPhase),
                   MQT_NAMED_BUILDER(singleControlledP)},
        QCTestCase{"TrivialControlledGlobalPhase",
                   MQT_NAMED_BUILDER(trivialControlledGlobalPhase),
                   MQT_NAMED_BUILDER(globalPhase)},
        QCTestCase{"InverseGlobalPhase", MQT_NAMED_BUILDER(inverseGlobalPhase),
                   MQT_NAMED_BUILDER(globalPhase)},
        QCTestCase{"InverseMultipleControlledGlobalPhase",
                   MQT_NAMED_BUILDER(inverseMultipleControlledGlobalPhase),
                   MQT_NAMED_BUILDER(multipleControlledP)}));
