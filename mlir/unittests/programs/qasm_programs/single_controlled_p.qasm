OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ p(0.123) q[0], q[1];
