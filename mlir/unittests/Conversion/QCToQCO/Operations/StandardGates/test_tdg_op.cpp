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
    QCTdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                        MQT_NAMED_BUILDER(qco::tdg)},
        QCToQCOTestCase{"SingleControlledTdg",
                        MQT_NAMED_BUILDER(qc::singleControlledTdg),
                        MQT_NAMED_BUILDER(qco::singleControlledTdg)},
        QCToQCOTestCase{"MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledTdg)}));
