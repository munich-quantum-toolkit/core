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
    QCOBarrierOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"Barrier", qco::barrier, qc::barrier},
        QCOToQCTestCase{"BarrierTwoQubits", qco::barrierTwoQubits,
                        qc::barrierTwoQubits},
        QCOToQCTestCase{"BarrierMultipleQubits", qco::barrierMultipleQubits,
                        qc::barrierMultipleQubits},
        QCOToQCTestCase{"SingleControlledBarrier", qco::singleControlledBarrier,
                        qc::barrier},
        QCOToQCTestCase{"InverseBarrier", qco::inverseBarrier, qc::barrier}),
    printTestName);
