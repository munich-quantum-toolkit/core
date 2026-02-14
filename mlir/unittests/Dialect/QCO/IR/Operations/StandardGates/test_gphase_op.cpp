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
    QCOGPhaseOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"GlobalPhase", MQT_NAMED_BUILDER(globalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"SingleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(singleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"MultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"InverseGlobalPhase", MQT_NAMED_BUILDER(inverseGlobalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"InverseMultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(inverseMultipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase)}));
