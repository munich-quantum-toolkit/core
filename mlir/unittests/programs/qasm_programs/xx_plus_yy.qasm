OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
xx_plus_yy(0.123, 0.456) q[0], q[1];
