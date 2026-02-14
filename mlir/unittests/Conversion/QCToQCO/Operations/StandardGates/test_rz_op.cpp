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
#include "qco_programs.h"
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCRZOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                                    MQT_NAMED_BUILDER(qco::rz)},
                    QCToQCOTestCase{"SingleControlledRZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledRz),
                                    MQT_NAMED_BUILDER(qco::singleControlledRz)},
                    QCToQCOTestCase{
                        "MultipleControlledRZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRz),
                        MQT_NAMED_BUILDER(qco::multipleControlledRz)}));
