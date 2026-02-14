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
    QCiSWAPOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap),
                        MQT_NAMED_BUILDER(qco::iswap)},
        QCToQCOTestCase{"SingleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledIswap),
                        MQT_NAMED_BUILDER(qco::singleControlledIswap)},
        QCToQCOTestCase{"MultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledIswap),
                        MQT_NAMED_BUILDER(qco::multipleControlledIswap)}));
