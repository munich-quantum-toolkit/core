OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl(2) @ rz(0.789) q[0], q[1], q[2];
