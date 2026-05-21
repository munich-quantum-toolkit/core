OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ ry(0.456) q[0], q[1];
