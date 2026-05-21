OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
barrier q[0], q[1], q[2];
