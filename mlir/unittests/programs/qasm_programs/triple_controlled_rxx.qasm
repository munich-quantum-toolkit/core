OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
ctrl(3) @ rxx(0.123) q[0], q[1], q[2], q[3], q[4];
