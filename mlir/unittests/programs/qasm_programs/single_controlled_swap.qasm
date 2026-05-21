OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ swap q[0], q[1], q[2];
