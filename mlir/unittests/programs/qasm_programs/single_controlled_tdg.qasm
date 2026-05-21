OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ctrl @ tdg q[0], q[1];
