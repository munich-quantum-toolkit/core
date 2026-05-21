OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ xx_minus_yy(0.123, 0.456) q[0], q[1], q[2];
