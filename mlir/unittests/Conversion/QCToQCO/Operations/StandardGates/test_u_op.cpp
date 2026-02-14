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
    QCUOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                                    MQT_NAMED_BUILDER(qco::u)},
                    QCToQCOTestCase{"SingleControlledU",
                                    MQT_NAMED_BUILDER(qc::singleControlledU),
                                    MQT_NAMED_BUILDER(qco::singleControlledU)},
                    QCToQCOTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qc::multipleControlledU),
                        MQT_NAMED_BUILDER(qco::multipleControlledU)}));
