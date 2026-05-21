OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ctrl(2) @ swap q[0], q[1], q[2], q[3];
