OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
reset q[0];
h q[1];
reset q[1];
