OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ sx q[0], q[1], q[2];
