OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ u2(0.234, 0.567) q[0], q[1], q[2];
