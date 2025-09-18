/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

HANDLE_OP_TYPE(0, None, 0, "none")

// Standard Operations
HANDLE_OP_TYPE(1, GPhase, 0, "gphase")
HANDLE_OP_TYPE(2, I, OpTypeDiag, "i")
HANDLE_OP_TYPE(3, Barrier, OpTypeDiag, "barrier")
HANDLE_OP_TYPE(4, H, OpTypeNone, "h")
HANDLE_OP_TYPE(5, X, OpTypeNone, "x")
HANDLE_OP_TYPE(6, Y, OpTypeNone, "y")
HANDLE_OP_TYPE(7, Z, OpTypeDiag, "z")
HANDLE_OP_TYPE(8, S, OpTypeDiag, "s")
HANDLE_OP_TYPE(8, Sdg, OpTypeDiag | OpTypeInv, "sdg")
HANDLE_OP_TYPE(9, T, OpTypeDiag, "t")
HANDLE_OP_TYPE(9, Tdg, OpTypeDiag | OpTypeInv, "tdg")
HANDLE_OP_TYPE(10, V, OpTypeNone, "v")
HANDLE_OP_TYPE(10, Vdg, OpTypeInv, "vdg")
HANDLE_OP_TYPE(11, U, OpTypeNone, "u")
HANDLE_OP_TYPE(12, U2, OpTypeNone, "u2")
HANDLE_OP_TYPE(13, P, OpTypeDiag, "p")
HANDLE_OP_TYPE(14, SX, OpTypeNone, "sx")
HANDLE_OP_TYPE(14, SXdg, OpTypeInv, "sxdg")
HANDLE_OP_TYPE(15, RX, OpTypeNone, "rx")
HANDLE_OP_TYPE(16, RY, OpTypeNone, "ry")
HANDLE_OP_TYPE(17, RZ, OpTypeDiag, "rz")
HANDLE_OP_TYPE(18, SWAP, OpTypeNone, "swap")
HANDLE_OP_TYPE(19, iSWAP, OpTypeNone,   "iswap")
HANDLE_OP_TYPE(19, iSWAPdg, OpTypeInv, "iswapdg")
HANDLE_OP_TYPE(20, Peres, OpTypeNone, "peres")
HANDLE_OP_TYPE(20, Peresdg, OpTypeInv, "peresdg")
HANDLE_OP_TYPE(21, DCX, OpTypeNone, "dcx")
HANDLE_OP_TYPE(22, ECR, OpTypeNone, "ecr")
HANDLE_OP_TYPE(23, RXX, OpTypeNone, "rxx")
HANDLE_OP_TYPE(24, RYY, OpTypeNone, "ryy")
HANDLE_OP_TYPE(25, RZZ, OpTypeDiag, "rzz")
HANDLE_OP_TYPE(26, RZX, OpTypeNone, "rzx")
HANDLE_OP_TYPE(27, XXminusYY, OpTypeNone, "xx_minus_yy")
HANDLE_OP_TYPE(28, XXplusYY, OpTypeNone, "xx_plus_yy")

// Compound Operation
HANDLE_OP_TYPE(29, Compound, OpTypeNone, "compound")

// Non Unitary Operations
HANDLE_OP_TYPE(30, Measure, OpTypeNone, "measure")
HANDLE_OP_TYPE(31, Reset, OpTypeNone, "reset")

// Control flow operations
HANDLE_OP_TYPE(32, IfElse, OpTypeNone, "if_else")

// Noise operations
HANDLE_OP_TYPE(33, ATrue, 0, "a_true")
HANDLE_OP_TYPE(34, AFalse, 0, "a_false")
HANDLE_OP_TYPE(35, MultiATrue, 0, "multi_a_true")
HANDLE_OP_TYPE(36, MultiAFalse, 0, "multi_a_false")

// Neutral atom shuttling operations
HANDLE_OP_TYPE(37, Move, 0, "move")
HANDLE_OP_TYPE(38, Bridge, 0, "bridge")
HANDLE_OP_TYPE(39, AodActivate, 0, "aod_activate")
HANDLE_OP_TYPE(40, AodDeactivate, 0, "aod_deactivate")
HANDLE_OP_TYPE(41, AodMove, 0, "aod_move")

LAST_OP_TYPE(42)


#undef OpTypeInv
#undef OpTypeDiag
