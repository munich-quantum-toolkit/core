/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qasm_programs.h"

#include <string>

// NOLINTBEGIN(readability-identifier-naming)
namespace mlir::qasm {

const std::string allocQubit = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit q;
)qasm";

const std::string allocQubitRegister = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
)qasm";

const std::string allocMultipleQubitRegisters = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q0;
qubit[3] q1;
)qasm";

const std::string allocLargeRegister = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[100] q;
)qasm";

const std::string singleMeasurementToSingleBit = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
measure q[0] -> c[0];
)qasm";

const std::string repeatedMeasurementToSameBit = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
measure q[0] -> c[0];
measure q[0] -> c[0];
measure q[0] -> c[0];
)qasm";

const std::string repeatedMeasurementToDifferentBits = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[3] c;
measure q[0] -> c[0];
measure q[0] -> c[1];
measure q[0] -> c[2];
)qasm";

const std::string multipleClassicalRegistersAndMeasurements =
    R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[1] c0;
bit[2] c1;
measure q[0] -> c0[0];
measure q[1] -> c1[0];
measure q[2] -> c1[1];
)qasm";

const std::string resetQubitAfterSingleOp = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
reset q[0];
)qasm";

const std::string resetMultipleQubitsAfterSingleOp = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
reset q[0];
h q[1];
reset q[1];
)qasm";

const std::string repeatedResetAfterSingleOp = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
reset q[0];
reset q[0];
reset q[0];
)qasm";

const std::string globalPhase = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
gphase(0.123);
)qasm";

const std::string identity = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
id q[0];
)qasm";

const std::string singleControlledIdentity = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ id q[0], q[1];
)qasm";

const std::string multipleControlledIdentity = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ id q[0], q[1], q[2];
)qasm";

const std::string x = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
x q[0];
)qasm";

const std::string singleControlledX = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ x q[0], q[1];
)qasm";

const std::string multipleControlledX = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ x q[0], q[1], q[2];
)qasm";

const std::string y = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
y q[0];
)qasm";

const std::string singleControlledY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ y q[0], q[1];
)qasm";

const std::string multipleControlledY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ y q[0], q[1], q[2];
)qasm";

const std::string z = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
z q[0];
)qasm";

const std::string singleControlledZ = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ z q[0], q[1];
)qasm";

const std::string multipleControlledZ = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ z q[0], q[1], q[2];
)qasm";

const std::string h = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
)qasm";

const std::string singleControlledH = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ h q[0], q[1];
)qasm";

const std::string multipleControlledH = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ h q[0], q[1], q[2];
)qasm";

const std::string s = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
s q[0];
)qasm";

const std::string singleControlledS = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ s q[0], q[1];
)qasm";

const std::string multipleControlledS = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ s q[0], q[1], q[2];
)qasm";

const std::string sdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
sdg q[0];
)qasm";

const std::string singleControlledSdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ sdg q[0], q[1];
)qasm";

const std::string multipleControlledSdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ sdg q[0], q[1], q[2];
)qasm";

const std::string t_ = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
t q[0];
)qasm";

const std::string singleControlledT = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ t q[0], q[1];
)qasm";

const std::string multipleControlledT = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ t q[0], q[1], q[2];
)qasm";

const std::string tdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
tdg q[0];
)qasm";

const std::string singleControlledTdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ tdg q[0], q[1];
)qasm";

const std::string multipleControlledTdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ tdg q[0], q[1], q[2];
)qasm";

const std::string sx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
sx q[0];
)qasm";

const std::string singleControlledSx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ sx q[0], q[1];
)qasm";

const std::string multipleControlledSx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ sx q[0], q[1], q[2];
)qasm";

const std::string sxdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
sxdg q[0];
)qasm";

const std::string singleControlledSxdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ sxdg q[0], q[1];
)qasm";

const std::string multipleControlledSxdg = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ sxdg q[0], q[1], q[2];
)qasm";

const std::string rx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
rx(0.123) q[0];
)qasm";

const std::string singleControlledRx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ rx(0.123) q[0], q[1];
)qasm";

const std::string multipleControlledRx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ rx(0.123) q[0], q[1], q[2];
)qasm";

const std::string ry = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
ry(0.456) q[0];
)qasm";

const std::string singleControlledRy = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ ry(0.456) q[0], q[1];
)qasm";

const std::string multipleControlledRy = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ ry(0.456) q[0], q[1], q[2];
)qasm";

const std::string rz = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
rz(0.789) q[0];
)qasm";

const std::string singleControlledRz = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ rz(0.789) q[0], q[1];
)qasm";

const std::string multipleControlledRz = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ rz(0.789) q[0], q[1], q[2];
)qasm";

const std::string p = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
p(0.123) q[0];
)qasm";

const std::string singleControlledP = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ p(0.123) q[0], q[1];
)qasm";

const std::string multipleControlledP = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ p(0.123) q[0], q[1], q[2];
)qasm";

const std::string r = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
r(0.123, 0.456) q[0];
)qasm";

const std::string singleControlledR = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ r(0.123, 0.456) q[0], q[1];
)qasm";

const std::string multipleControlledR = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ r(0.123, 0.456) q[0], q[1], q[2];
)qasm";

const std::string u2 = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
u2(0.234, 0.567) q[0];
)qasm";

const std::string singleControlledU2 = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ u2(0.234, 0.567) q[0], q[1];
)qasm";

const std::string multipleControlledU2 = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ u2(0.234, 0.567) q[0], q[1], q[2];
)qasm";

const std::string u = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
u(0.1, 0.2, 0.3) q[0];
)qasm";

const std::string singleControlledU = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ u(0.1, 0.2, 0.3) q[0], q[1];
)qasm";

const std::string multipleControlledU = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ u(0.1, 0.2, 0.3) q[0], q[1], q[2];
)qasm";

const std::string swap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
swap q[0], q[1];
)qasm";

const std::string singleControlledSwap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ swap q[0], q[1], q[2];
)qasm";

const std::string multipleControlledSwap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ swap q[0], q[1], q[2], q[3];
)qasm";

const std::string iswap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
iswap q[0], q[1];
)qasm";

const std::string singleControlledIswap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ iswap q[0], q[1], q[2];
)qasm";

const std::string multipleControlledIswap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ iswap q[0], q[1], q[2], q[3];
)qasm";

const std::string inverseIswap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
inv @ iswap q[0], q[1];
)qasm";

const std::string inverseMultipleControlledIswap = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
inv @ ctrl(2) @ iswap q[0], q[1], q[2], q[3];
)qasm";

const std::string dcx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
dcx q[0], q[1];
)qasm";

const std::string singleControlledDcx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ dcx q[0], q[1], q[2];
)qasm";

const std::string multipleControlledDcx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ dcx q[0], q[1], q[2], q[3];
)qasm";

const std::string ecr = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ecr q[0], q[1];
)qasm";

const std::string singleControlledEcr = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ ecr q[0], q[1], q[2];
)qasm";

const std::string multipleControlledEcr = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ ecr q[0], q[1], q[2], q[3];
)qasm";

const std::string rxx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
rxx(0.123) q[0], q[1];
)qasm";

const std::string singleControlledRxx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ rxx(0.123) q[0], q[1], q[2];
)qasm";

const std::string multipleControlledRxx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ rxx(0.123) q[0], q[1], q[2], q[3];
)qasm";

const std::string tripleControlledRxx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
ctrl(3) @ rxx(0.123) q[0], q[1], q[2], q[3], q[4];
)qasm";

const std::string ryy = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ryy(0.123) q[0], q[1];
)qasm";

const std::string singleControlledRyy = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ ryy(0.123) q[0], q[1], q[2];
)qasm";

const std::string multipleControlledRyy = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ ryy(0.123) q[0], q[1], q[2], q[3];
)qasm";

const std::string rzx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
rzx(0.123) q[0], q[1];
)qasm";

const std::string singleControlledRzx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ rzx(0.123) q[0], q[1], q[2];
)qasm";

const std::string multipleControlledRzx = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ rzx(0.123) q[0], q[1], q[2], q[3];
)qasm";

const std::string rzz = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
rzz(0.123) q[0], q[1];
)qasm";

const std::string singleControlledRzz = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ rzz(0.123) q[0], q[1], q[2];
)qasm";

const std::string multipleControlledRzz = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ rzz(0.123) q[0], q[1], q[2], q[3];
)qasm";

const std::string xxPlusYY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
xx_plus_yy(0.123, 0.456) q[0], q[1];
)qasm";

const std::string singleControlledXxPlusYY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ xx_plus_yy(0.123, 0.456) q[0], q[1], q[2];
)qasm";

const std::string multipleControlledXxPlusYY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ xx_plus_yy(0.123, 0.456) q[0], q[1], q[2], q[3];
)qasm";

const std::string xxMinusYY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
xx_minus_yy(0.123, 0.456) q[0], q[1];
)qasm";

const std::string singleControlledXxMinusYY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ xx_minus_yy(0.123, 0.456) q[0], q[1], q[2];
)qasm";

const std::string multipleControlledXxMinusYY = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ xx_minus_yy(0.123, 0.456) q[0], q[1], q[2], q[3];
)qasm";

const std::string barrier = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
barrier q[0];
)qasm";

const std::string barrierTwoQubits = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
barrier q[0], q[1];
)qasm";

const std::string barrierMultipleQubits = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
barrier q[0], q[1], q[2];
)qasm";

const std::string simpleIf = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
bit c = measure q[0];
if (c) {
  x q[0];
}
)qasm";

const std::string ifElse = R"qasm(OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
bit c = measure q[0];
if (c) {
  x q[0];
} else {
  z q[0];
}
)qasm";

} // namespace mlir::qasm
// NOLINTEND(readability-identifier-naming)
