OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ rxx(0.123) q[0], q[1], q[2];
