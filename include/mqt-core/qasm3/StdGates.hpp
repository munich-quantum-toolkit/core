/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Gate.hpp"
#include "ir/operations/OpType.hpp"

#include <map>
#include <memory>
#include <string>

namespace qasm3 {
// Non-natively supported gates from
// https://github.com/Qiskit/qiskit/blob/main/qiskit/qasm/libs/stdgates.inc
const std::string STDGATES =
    "// four parameter controlled-U gate with relative phase\n"
    "gate cu(theta, phi, lambda, gamma) c, t { p(gamma) c; ctrl @ U(theta, "
    "phi, lambda) c, t; }\n";

// Non-natively supported gates from
// https://github.com/Qiskit/qiskit/blob/main/qiskit/qasm/libs/qelib1.inc
const std::string QE1LIB = "gate rccx a, b, c {\n"
                           "  u2(0, pi) c; u1(pi/4) c; \n"
                           "  cx b, c; u1(-pi/4) c; \n"
                           "  cx a, c; u1(pi/4) c; \n"
                           "  cx b, c; u1(-pi/4) c; \n"
                           "  u2(0, pi) c; \n"
                           "}\n"
                           "gate rc3x a,b,c,d {\n"
                           "  u2(0,pi) d; u1(pi/4) d; \n"
                           "  cx c,d; u1(-pi/4) d; u2(0,pi) d; \n"
                           "  cx a,d; u1(pi/4) d; \n"
                           "  cx b,d; u1(-pi/4) d; \n"
                           "  cx a,d; u1(pi/4) d; \n"
                           "  cx b,d; u1(-pi/4) d; \n"
                           "  u2(0,pi) d; u1(pi/4) d; \n"
                           "  cx c,d; u1(-pi/4) d; \n"
                           "  u2(0,pi) d; \n"
                           "}\n";

const std::map<std::string, std::shared_ptr<Gate>> STANDARD_GATES = {
    // gates from which all other gates can be constructed.
    {"gphase",
     std::make_shared<StandardGate>(StandardGate({.nControls = 0,
                                                  .nTargets = 0,
                                                  .nParameters = 1,
                                                  .type = qc::GPhase}))},
    {"U",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 3, .type = qc::U}))},

    // natively supported gates
    {"p",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 1, .type = qc::P}))},
    {"u1",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 1, .type = qc::P}))},
    {"phase",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 1, .type = qc::P}))},
    {"cphase",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 1, .type = qc::P}))},
    {"cp",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 1, .type = qc::P}))},

    {"id",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::I}))},
    {"u2",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 2, .type = qc::U2}))},
    {"u3",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 3, .type = qc::U}))},
    {"u",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 3, .type = qc::U}))},

    {"x",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::X}))},
    {"cx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 0, .type = qc::X}))},
    {"CX",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 0, .type = qc::X}))},
    {"ccx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 2, .nTargets = 1, .nParameters = 0, .type = qc::X}))},
    {"c3x",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 3, .nTargets = 1, .nParameters = 0, .type = qc::X}))},
    {"c4x",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 4, .nTargets = 1, .nParameters = 0, .type = qc::X}))},

    {"rx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 1, .type = qc::RX}))},
    {"crx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 1, .type = qc::RX}))},

    {"y",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::Y}))},
    {"cy",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 0, .type = qc::Y}))},

    {"ry",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 1, .type = qc::RY}))},
    {"cry",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 1, .type = qc::RY}))},

    {"z",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::Z}))},
    {"cz",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 0, .type = qc::Z}))},

    {"rz",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 1, .type = qc::RZ}))},
    {"crz",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 1, .type = qc::RZ}))},

    {"r",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 2, .type = qc::R}))},
    {"prx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 2, .type = qc::R}))},
    {"cr",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 2, .type = qc::R}))},

    {"h",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::H}))},
    {"ch",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 1, .nParameters = 0, .type = qc::H}))},

    {"s",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::S}))},
    {"sdg",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::Sdg}))},

    {"t",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::T}))},
    {"tdg",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::Tdg}))},

    {"sx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::SX}))},
    {"sxdg",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 1, .nParameters = 0, .type = qc::SXdg}))},
    {"c3sqrtx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 3, .nTargets = 1, .nParameters = 0, .type = qc::SXdg}))},

    {"swap",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 2, .nParameters = 0, .type = qc::SWAP}))},
    {"cswap",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 1, .nTargets = 2, .nParameters = 0, .type = qc::SWAP}))},

    {"iswap",
     std::make_shared<StandardGate>(StandardGate({.nControls = 0,
                                                  .nTargets = 2,
                                                  .nParameters = 0,
                                                  .type = qc::iSWAP}))},
    {"iswapdg",
     std::make_shared<StandardGate>(StandardGate({.nControls = 0,
                                                  .nTargets = 2,
                                                  .nParameters = 0,
                                                  .type = qc::iSWAPdg}))},

    {"rxx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 2, .nParameters = 1, .type = qc::RXX}))},
    {"ryy",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 2, .nParameters = 1, .type = qc::RYY}))},
    {"rzz",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 2, .nParameters = 1, .type = qc::RZZ}))},
    {"rzx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 2, .nParameters = 1, .type = qc::RZX}))},
    {"dcx",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 2, .nParameters = 0, .type = qc::DCX}))},
    {"ecr",
     std::make_shared<StandardGate>(StandardGate(
         {.nControls = 0, .nTargets = 2, .nParameters = 0, .type = qc::ECR}))},
    {"xx_minus_yy",
     std::make_shared<StandardGate>(StandardGate({.nControls = 0,
                                                  .nTargets = 2,
                                                  .nParameters = 2,
                                                  .type = qc::XXminusYY}))},
    {"xx_plus_yy",
     std::make_shared<StandardGate>(StandardGate({.nControls = 0,
                                                  .nTargets = 2,
                                                  .nParameters = 2,
                                                  .type = qc::XXplusYY}))},
};
} // namespace qasm3
