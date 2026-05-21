OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ sx q[0], q[1];
