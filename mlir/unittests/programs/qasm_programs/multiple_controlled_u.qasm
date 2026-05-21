OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ u(0.1, 0.2, 0.3) q[0], q[1], q[2];
