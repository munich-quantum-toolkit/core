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
    QCOTdgOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Tdg", MQT_NAMED_BUILDER(qco::tdg),
                                    MQT_NAMED_BUILDER(qc::tdg)},
                    QCOToQCTestCase{"SingleControlledTdg",
                                    MQT_NAMED_BUILDER(qco::singleControlledTdg),
                                    MQT_NAMED_BUILDER(qc::singleControlledTdg)},
                    QCOToQCTestCase{
                        "MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg)}));
