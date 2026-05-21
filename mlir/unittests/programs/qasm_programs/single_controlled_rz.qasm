OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ rz(0.789) q[0], q[1];
