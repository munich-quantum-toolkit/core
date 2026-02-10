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

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCGPhaseOpTest, QCTest,
    testing::Values(
        QCTestCase{"GlobalPhase", globalPhase, globalPhase},
        QCTestCase{"SingleControlledGlobalPhase", singleControlledGlobalPhase,
                   p},
        QCTestCase{"MultipleControlledGlobalPhase",
                   multipleControlledGlobalPhase, multipleControlledP},
        QCTestCase{"NestedControlledGlobalPhase", nestedControlledGlobalPhase,
                   singleControlledP},
        QCTestCase{"TrivialControlledGlobalPhase", trivialControlledGlobalPhase,
                   globalPhase},
        QCTestCase{"InverseGlobalPhase", inverseGlobalPhase, globalPhase},
        QCTestCase{"InverseMultipleControlledGlobalPhase",
                   inverseMultipleControlledGlobalPhase, multipleControlledP}),
    printTestName);
