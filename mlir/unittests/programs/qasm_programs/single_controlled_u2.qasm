OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ u2(0.234, 0.567) q[0], q[1];
