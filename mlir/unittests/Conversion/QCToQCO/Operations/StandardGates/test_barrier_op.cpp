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
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCBarrierOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"Barrier", qc::barrier, qco::barrier},
        QCToQCOTestCase{"BarrierTwoQubits", qc::barrierTwoQubits,
                        qco::barrierTwoQubits},
        QCToQCOTestCase{"BarrierMultipleQubits", qc::barrierMultipleQubits,
                        qco::barrierMultipleQubits},
        QCToQCOTestCase{"SingleControlledBarrier", qc::singleControlledBarrier,
                        qco::barrier},
        QCToQCOTestCase{"InverseBarrier", qc::inverseBarrier, qco::barrier}),
    printTestName);
