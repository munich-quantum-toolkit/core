OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ r(0.123, 0.456) q[0], q[1];
