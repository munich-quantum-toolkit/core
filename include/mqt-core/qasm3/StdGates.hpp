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
     std::make_shared<StandardGate>(StandardGate({0, 0, 1, qc::GPhase}))},
    {"U", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::U}))},

    // natively supported gates
    {"p", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::P}))},
    {"u1", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::P}))},
    {"phase", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::P}))},
    {"cphase", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::P}))},
    {"cp", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::P}))},

    {"id", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::I}))},
    {"u2", std::make_shared<StandardGate>(StandardGate({0, 1, 2, qc::U2}))},
    {"u3", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::U}))},
    {"u", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::U}))},

    {"x", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::X}))},
    {"cx", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::X}))},
    {"CX", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::X}))},
    {"ccx", std::make_shared<StandardGate>(StandardGate({2, 1, 0, qc::X}))},
    {"c3x", std::make_shared<StandardGate>(StandardGate({3, 1, 0, qc::X}))},
    {"c4x", std::make_shared<StandardGate>(StandardGate({4, 1, 0, qc::X}))},

    {"rx", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::RX}))},
    {"crx", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::RX}))},

    {"y", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Y}))},
    {"cy", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::Y}))},

    {"ry", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::RY}))},
    {"cry", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::RY}))},

    {"z", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Z}))},
    {"cz", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::Z}))},

    {"rz", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::RZ}))},
    {"crz", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::RZ}))},

    {"h", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::H}))},
    {"ch", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::H}))},

    {"s", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::S}))},
    {"sdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Sdg}))},

    {"t", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::T}))},
    {"tdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Tdg}))},

    {"sx", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::SX}))},
    {"sxdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::SXdg}))},
    {"c3sqrtx",
     std::make_shared<StandardGate>(StandardGate({3, 1, 0, qc::SXdg}))},

    {"swap", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::SWAP}))},
    {"cswap",
     std::make_shared<StandardGate>(StandardGate({1, 2, 0, qc::SWAP}))},

    {"iswap",
     std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::iSWAP}))},
    {"iswapdg",
     std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::iSWAPdg}))},

    {"rxx", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RXX}))},
    {"ryy", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RYY}))},
    {"rzz", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RZZ}))},
    {"rzx", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RZX}))},
    {"dcx", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::DCX}))},
    {"ecr", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::ECR}))},
    {"xx_minus_yy",
     std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::XXminusYY}))},
    {"xx_plus_yy",
     std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::XXplusYY}))},
};
} // namespace qasm3
