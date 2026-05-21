OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ t q[0], q[1];
